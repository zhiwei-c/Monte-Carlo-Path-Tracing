#pragma once

#include "../core/integrator_base.h"

NAMESPACE_BEGIN(raytracer)

//路径点
struct PathVertex
{
    Intersection its;     //路径点对应的交点
    Vector3 wi;           //在路径中的当前点，光线入射方向
    Vector3 wo;           //在路径中的当前点，光线出射方向
    Float pdf;            //在路径中的当前点，光线入射并出射的概率
    Spectrum attenuation; //在路径中的当前点，光线入射并出射对应的光能衰减系数
    Spectrum L;           //在路径中的当前点，光线沿出射方向传递能量的数学期望

    PathVertex(Intersection its, Vector3 wi, Vector3 wo)
        : its(its), wi(wi), wo(wo), pdf(-1), attenuation(Spectrum(1)), L(Spectrum(0))
    {
    }
};

//双向路径追踪派生类
class BdptIntegrator : public Integrator
{
public:
    ///\brief 双向路径追踪
    ///\param max_depth 递归地追踪光线最大深度
    ///\param rr_depth 最小的光线追踪深度，超过该深度后进行俄罗斯轮盘赌抽样控制光线追踪深度
    BdptIntegrator(int max_depth, int rr_depth) : Integrator(IntegratorType::kBdpt, max_depth, rr_depth) {}

    ///\brief 着色
    ///\param eye_pos 观察点的位置
    ///\param look_dir 观察方向
    ///\return 观察点来源于给定观察方向的辐射亮度
    Spectrum Shade(const Vector3 &eye_pos, const Vector3 &look_dir) const override
    {
        const Vector3 &wo = -look_dir;
        auto its = Intersection();
        if (!this->bvh_ || !this->bvh_->Intersect(Ray(eye_pos, look_dir), its))
        { //原初光线源于环境
            if (envmap_)
                return envmap_->radiance(look_dir);
            else
                return Spectrum(0);
        }
        else
        {
            if (its.absorb())
            { //单面材质物体的背面，只吸收而不反射或折射光线
                return Spectrum(0);
            }
            else if (its.HasEmission())
            { //原初光线源于发光物体
                return its.radiance();
            }
            else //处理双向路径追踪算法
            {
                std::vector<PathVertex> emitter_path = CreateEmitterPath();
                std::vector<PathVertex> camera_path = CreateCameraPath(its, wo);
                for (int c_idx = camera_path.size() - 1; c_idx >= 0; --c_idx)
                {
                    const PathVertex &c = camera_path[c_idx];

                    //直接光照
                    Spectrum L_direct = EmitterEnv2OneV(c, !emitter_path.empty() ? &emitter_path[0].its : nullptr);

                    auto L_indirects = std::vector<Spectrum>();
                    auto pdfs_indirects = std::vector<Float>();

                    auto L_indirect_bsdf = Spectrum(0);
                    //来自前一个照相机路径的间接光照，是根据 BSDF 抽样产生的间接光照
                    if (c_idx < camera_path.size() - 1)
                    {
                        L_indirect_bsdf = camera_path[c_idx + 1].L * (c.attenuation / c.pdf);
                        Float pdf_indirect = c.pdf;
                        if (c_idx > rr_depth_)
                        {
                            L_indirect_bsdf /= pdf_rr_;
                            pdf_indirect *= pdf_rr_;
                        }
                        L_indirects.push_back(L_indirect_bsdf);
                        pdfs_indirects.push_back(pdf_indirect);
                    }

                    if (c.its.HarshLobe())
                    { //当前照相机路径点总光照
                        camera_path[c_idx].L = L_direct + L_indirect_bsdf;
                    }
                    else
                    {
                        //来自光源路径的间接光照，是主动地抽样其它景物表面产生的间接光照
                        for (int e_idx = 1; e_idx < emitter_path.size(); e_idx++)
                        {
                            if (emitter_path[e_idx].its.HarshLobe())
                                continue;

                            auto [L_indirect, pdf_indirect] = PrepareOtherEmitter2OneC(emitter_path, e_idx, c);
                            if (pdf_indirect > 0)
                            {
                                if (c_idx > rr_depth_)
                                {
                                    L_indirect /= pdf_rr_;
                                    pdf_indirect *= pdf_rr_;
                                }
                                L_indirects.push_back(L_indirect);
                                pdfs_indirects.push_back(pdf_indirect);
                            }
                        }
                        //多重重要抽样
                        Spectrum L_indirect = WeightPowerHeuristic(L_indirects, pdfs_indirects);
                        //当前照相机路径点总光照
                        camera_path[c_idx].L = L_direct + L_indirect;
                    }
                }
                return camera_path[0].L;
            }
        }
    }

private:
    ///\brief 从光源出发，创建路径点
    std::vector<PathVertex> CreateEmitterPath() const
    {
        auto emitter_path = std::vector<PathVertex>();
        //从发光物体上直接采样，生成第一个光源路径点
        auto its = Intersection();
        if (!SampleEmitterDirectIts(its))
            return emitter_path;
        //采样光线从第一个光源路径点射出的方向
        auto wo = Vector3(0);
        SampleHemisCos(its.normal(), wo);
        wo = -wo;
        emitter_path.push_back({its, Vector3(0), wo});

        //生成第二个及之后光源路径点
        size_t depth = 0;
        while (depth < static_cast<size_t>(max_depth_) && (depth < rr_depth_ || UniformFloat() < pdf_rr_))
        {
            const PathVertex &e = emitter_path.back();
            its = Intersection();
            if (!this->bvh_->Intersect(Ray(e.its.pos(), e.wo), its) || its.absorb() || its.HasEmission())
                break;
            emitter_path.push_back({its, e.wo, Vector3(0)});
            depth += 1;

            PathVertex &e_next = emitter_path.back();
            SamplingRecord bs_next = e_next.its.Sample(-e_next.wi, false);
            if (bs_next.type == ScatteringType::kNone)
                break;
            bs_next = e_next.its.Eval(e_next.wi, -bs_next.wi);
            if (bs_next.type == ScatteringType::kNone)
                break;
            e_next.wo = bs_next.wo;
            e_next.pdf = bs_next.pdf;
            e_next.attenuation = bs_next.attenuation;
        }

        //计算第1个光源路径点向第2个光源路径点传递的辐射亮度的数学期望
        PathVertex &e_first = emitter_path[0];
        e_first.L = e_first.its.radiance();

        //计算第2个光源路径点向第3个光源路径点传递的辐射亮度的数学期望
        if (emitter_path.size() > 2)
        {
            PathVertex &e_second = emitter_path[1];
            e_second.L = EmitterEnv2OneV(e_second);
        }

        //计算第3个及之后的光源路径点向下一个光源路径点传递辐射亮度的数学期望
        for (int i = 2; i < emitter_path.size() - 1; i++)
        {
            //计算来自第前一个光源路径点的间接光照
            PathVertex &e_pre = emitter_path[i - 1];
            PathVertex &e = emitter_path[i];
            Spectrum L_indirect = e_pre.L * (e.attenuation / e.pdf);
            if (i > rr_depth_)
                L_indirect /= pdf_rr_;

            //计算直接光照
            Spectrum L_direct_env = EmitterEnv2OneV(e, &e_first.its);

            e.L = L_indirect + L_direct_env;
        }
        return emitter_path;
    }

    ///\brief 从相机出发，创建路径点
    std::vector<PathVertex> CreateCameraPath(const Intersection &its_first, const Vector3 &wo_first) const
    {
        auto camera_path = std::vector<PathVertex>();
        camera_path.push_back({its_first, Vector3(0), wo_first});
        size_t depth = 0;
        auto its_pre = Intersection();
        while (depth < static_cast<size_t>(max_depth_) && (depth < rr_depth_ || UniformFloat() < pdf_rr_))
        {
            PathVertex &c = camera_path.back();
            SamplingRecord b_rec = c.its.Sample(c.wo);
            if (b_rec.type == ScatteringType::kNone)
                break;

            its_pre = Intersection();
            if (!this->bvh_->Intersect(Ray(b_rec.pos, -b_rec.wi), its_pre) || its_pre.absorb() || its_pre.HasEmission())
                break;

            c.wi = b_rec.wi;
            c.pdf = b_rec.pdf;
            c.attenuation = b_rec.attenuation;

            camera_path.push_back({its_pre, Vector3(0), b_rec.wi});

            depth += 1;
        }
        return camera_path;
    }

    ///\brief 光源与环境光 -> 某个路径点，计算直接光照的辐射亮度（光亮度）的数学期望
    Spectrum EmitterEnv2OneV(const PathVertex &v, const Intersection *its_emitter_ptr = nullptr) const
    {
        Vector3 wo = v.wo,
                normal = v.its.normal();
        auto L_emitter = Spectrum(0),
             L_env = Spectrum(0);

        //按面积直接采样光源，并按多重重要性采样合并
        if (!v.its.HarshLobe())
        {
            if (its_emitter_ptr)
            {
                if (!EmitterDirectArea(v.its, wo, L_emitter, its_emitter_ptr))
                    EmitterDirectArea(v.its, wo, L_emitter);
            }
            else
                EmitterDirectArea(v.its, wo, L_emitter);
        }

        SamplingRecord b_rec = v.its.Sample(wo);
        if (b_rec.type == ScatteringType::kNone)
            return L_emitter;
        Spectrum local_attenuation = b_rec.attenuation / b_rec.pdf;

        auto its_pre = Intersection();
        if (!this->bvh_->Intersect(Ray(b_rec.pos, -b_rec.wi), its_pre))
        { //按 BSDF 采样环境光
            if (envmap_ != nullptr)
            {
                L_env = envmap_->radiance(-b_rec.wi) * local_attenuation;
            }
        }
        else if (!its_pre.absorb() && its_pre.HasEmission())
        { //按 BSDF 采样来自面光源的直接光照
            if (v.its.HarshLobe())
                L_emitter += its_pre.radiance() * local_attenuation;
            else
            {
                Float pdf_direct = PdfEmitterDirect(its_pre, b_rec.wi),
                      weight_bsdf = MisWeight(b_rec.pdf, pdf_direct);
                L_emitter += weight_bsdf * its_pre.radiance() * local_attenuation;
            }
        }
        auto L_direct = L_emitter + L_env;
        return L_direct;
    }

    ///\brief 第二个或之后的某个光源路径点 -> 某个相机路径点 -> 下一个相机路径点，计算辐射亮度（光亮度）的数学期望及概率
    std::pair<Spectrum, Float> PrepareOtherEmitter2OneC(const std::vector<PathVertex> &emitter_path, const int e_index,
                                                        PathVertex c) const
    {
        const PathVertex &e_first = emitter_path[0];         //光源路径中的第一个点
        const PathVertex &e_pre = emitter_path[e_index - 1]; //当前光源路径点的前一个光源路径点
        PathVertex e = emitter_path[e_index];                //当前光源路径点

        if (!Visible(e.its, c.its))
        { //当前光源路径点与相机路径点之间不可见
            return {Spectrum(0), 0};
        }

        //计算从当前光源路径点向相机路径点传播光线的方向
        Vector3 ray_dir_vec = c.its.pos() - e.its.pos();
        e.wo = glm::normalize(ray_dir_vec);
        c.wi = e.wo;

        SamplingRecord rec_now = c.its.Eval(c.wi, c.wo);
        if (rec_now.type == ScatteringType::kNone)
            return {Spectrum(0), 0};
        Spectrum local_attenutation = rec_now.attenuation / rec_now.pdf;

        auto L_pre = Spectrum(0); //光源路径点向相机路径点传递辐射亮度的数学期望

        //计算光源路径点接收直接光照，并向相机路径点传递辐射亮度的数学期望
        L_pre += (e_index == 1) ? EmitterEnv2OneV(e) : EmitterEnv2OneV(e, &e_first.its);

        if (e_index > 1)
        { //计算光源路径点接收间接光照，并向相机路径点传递辐射亮度的数学期望
            SamplingRecord rec_pre = e.its.Eval(e.wi, e.wo);
            if (rec_pre.pdf > kEpsilonPdf2)
            {
                Vector3 L_pre_indirect = e_pre.L * (rec_pre.attenuation / rec_pre.pdf);
                if (e_index - 1 > rr_depth_)
                    L_pre_indirect /= pdf_rr_;
                L_pre += L_pre_indirect;
            }
        }
        if (L_pre.r + L_pre.g + L_pre.b < kEpsilon)
        { //光源路径点对相机路径点没有贡献辐射亮度
            return {Spectrum(0), 0};
        }

        auto L_indirect = L_pre * local_attenutation; //当前点接收间接光照的数学期望
        Float pdf_indirect = rec_now.pdf;
        if (e_index > rr_depth_)
        {
            L_indirect /= pdf_rr_;
            pdf_indirect *= pdf_rr_;
        }
        return {L_indirect, pdf_indirect};
    }
};

NAMESPACE_END(raytracer)