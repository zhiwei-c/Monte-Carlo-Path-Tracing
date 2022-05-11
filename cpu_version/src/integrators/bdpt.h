#pragma once

#include "../core/integrator_base.h"

NAMESPACE_BEGIN(raytracer)

//路径点
struct PathVertex
{
    Intersection its;    //路径点对应的交点
    Vector3 wi;          //在路径中的当前点，光线入射方向
    Vector3 wo;          //在路径中的当前点，光线出射方向
    Float cos_theta_abs; //在路径中的当前点，光线入射方向和当前点法线夹角余弦的绝对值
    Float pdf;           //在路径中的当前点，光线入射并出射的概率
    Spectrum bsdf;       //在路径中的当前点，光线入射并出射对应的 BSDF 数值
    Spectrum L;          //在路径中的当前点，光线沿出射方向传递能量的数学期望

    PathVertex(Intersection its, Vector3 wi, Vector3 wo)
        : its(its), wi(wi), wo(wo), cos_theta_abs(2), pdf(-1), bsdf(Spectrum(0)), L(Spectrum(0))
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
    BdptIntegrator(int max_depth, int rr_depth) : Integrator(max_depth, rr_depth) {}

    ///\brief 着色
    ///\param eye_pos 观察点的位置
    ///\param look_dir 观察方向
    ///\return 观察点来源于给定观察方向的辐射亮度
    Spectrum Shade(const Vector3 &eye_pos, const Vector3 &look_dir) const override
    {
        auto its = Intersection();
        if (!this->bvh_ || !this->bvh_->Intersect(Ray(eye_pos, look_dir), its)) //原初光线源于环境
            return envmap_ ? envmap_->radiance(look_dir) : Spectrum(0);
        else if (its.absorb()) //单面材质物体的背面，只吸收而不反射或折射光线
            return Spectrum(0);
        else if (its.HasEmission()) //原初光线源于发光物体
            return its.radiance();
        else //处理双向路径追踪算法
        {
            std::vector<PathVertex> emitter_path = CreateEmitterPath();
            std::vector<PathVertex> camera_path = CreateCameraPath(its, -look_dir);
            for (int c_idx = camera_path.size() - 1; c_idx >= 0; --c_idx)
            {
                const PathVertex &c = camera_path[c_idx];
                //直接光照
                Spectrum L_direct = EmitterEnv2OneV(c, !emitter_path.empty() ? &emitter_path[0].its : nullptr);
                //来自前一个照相机路径的间接光照
                auto L_indirect_pre = Spectrum(0);
                Float pdf_pre = 0;
                if (c_idx < camera_path.size() - 1)
                {
                    L_indirect_pre = camera_path[c_idx + 1].L * c.bsdf * (c.cos_theta_abs / c.pdf);
                    pdf_pre = c.pdf;
                    if (c_idx > rr_depth_)
                    {
                        L_indirect_pre /= pdf_rr_;
                        pdf_pre *= pdf_rr_;
                    }
                }

                if (c.its.HarshLobe())
                    camera_path[c_idx].L = L_direct + L_indirect_pre; //当前照相机路径点总光照
                else
                {
                    auto L_indirects = std::vector<Spectrum>();
                    auto pdfs = std::vector<Float>();
                    if (L_indirect_pre.r + L_indirect_pre.g + L_indirect_pre.b > kEpsilon)
                    {
                        L_indirects.push_back(L_indirect_pre);
                        pdfs.push_back(pdf_pre);
                    }
                    //来自光源路径的间接光照
                    for (int e_idx = 1; e_idx < emitter_path.size(); e_idx++)
                    {
                        if (max_depth_ > 0 && c_idx + e_idx + 2 > max_depth_)
                            break;

                        auto [L_temp, pdf_temp] = PrepareOtherEmitter2OneC(emitter_path, e_idx, c);
                        if (pdf_temp > kEpsilonPdf)
                        {
                            L_indirects.push_back(L_temp);
                            pdfs.push_back(pdf_temp);
                        }
                    }
                    //多重重要抽样
                    Spectrum L_indirect = WeightPowerHeuristic(L_indirects, pdfs);
                    //当前照相机路径点总光照
                    camera_path[c_idx].L = L_indirect + L_direct;
                }
            }
            return camera_path[0].L;
        }
    }

private:
    ///\brief 从光源出发，创建路径点
    std::vector<PathVertex> CreateEmitterPath() const
    {
        auto emitter_path = std::vector<PathVertex>();
        //从发光物体上直接采样，生成第一个光源路径点
        auto its_first = Intersection();
        if (!SampleEmitterDirectIts(its_first))
            return emitter_path;
        //采样光线从第一个光源路径点射出的方向
        auto wo_first = Vector3(0);
        SampleHemisCos(its_first.normal(), wo_first);
        wo_first = -wo_first;
        emitter_path.push_back({its_first, Vector3(0), wo_first});
        //生成第二个及之后光源路径点
        int depth = 1;
        auto its_next = Intersection();
        while (max_depth_ > 0 && emitter_path.size() < max_depth_ ||
               max_depth_ <= 0 && (depth <= rr_depth_ || UniformFloat() < pdf_rr_))
        {
            PathVertex &e = emitter_path.back();
            its_next = Intersection();
            if (!this->bvh_->Intersect(Ray(e.its.pos(), e.wo), its_next) ||
                its_next.absorb() ||
                its_next.HasEmission())
                break;
            emitter_path.push_back({its_next, e.wo, Vector3(0)});
            depth += 1;
            PathVertex &e_next = emitter_path.back();
            e_next.cos_theta_abs = std::fabs(glm::dot(e_next.wi, e_next.its.normal()));
            std::unique_ptr<BsdfSampling> bs_next = e_next.its.Sample(-e_next.wi, false);
            if (!bs_next)
                break;
            Vector3 wo_next = -bs_next->wi;
            Float pdf_next = its_next.Pdf(e_next.wi, wo_next);
            if (pdf_next < kEpsilonPdf2)
                break;
            e_next.wo = wo_next;
            e_next.pdf = pdf_next;
            e_next.bsdf = its_next.Eval(e_next.wi, e_next.wo);
        }
        PathVertex &e_first = emitter_path[0];
        e_first.L = e_first.its.radiance();
        //第一个光源路径点 -> 第二个光源路径点 -> 第三个光源路径点，预计算传递的辐射亮度（光亮度）期望
        if (emitter_path.size() > 2)
        {
            PathVertex &e_second = emitter_path[1];
            e_second.L = EmitterEnv2OneV(e_second);
        }
        //第二个及之后的光源路径点 -> 下一个光源路径点 -> 下一个光源路径点，预计算传递的辐射亮度（光亮度）期望
        for (int i = 2; i < emitter_path.size() - 1; i++)
        {
            PathVertex &e_pre = emitter_path[i - 1];
            PathVertex &e = emitter_path[i];
            Spectrum L_indirect = e_pre.L * e.bsdf * (e.cos_theta_abs / e.pdf);
            if (i > rr_depth_)
                L_indirect /= pdf_rr_;
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
        int depth = 1;
        auto its_pre = Intersection();
        while (max_depth_ > 0 && camera_path.size() < max_depth_ ||
               max_depth_ <= 0 && (depth <= rr_depth_ || UniformFloat() < pdf_rr_))
        {
            PathVertex &c = camera_path.back();
            std::unique_ptr<BsdfSampling> b_rec = c.its.Sample(c.wo);
            if (!b_rec)
                break;
            its_pre = Intersection();
            if (!this->bvh_->Intersect(Ray(c.its.pos(), -b_rec->wi), its_pre) ||
                its_pre.absorb() || its_pre.HasEmission())
                break;
            c.wi = b_rec->wi;
            c.cos_theta_abs = std::fabs(glm::dot(b_rec->wi, c.its.normal()));
            c.pdf = b_rec->pdf;
            c.bsdf = b_rec->attenuation;
            camera_path.push_back({its_pre, Vector3(0), b_rec->wi});
            depth += 1;
        }
        return camera_path;
    }

    ///\brief 光源与环境光 -> 某个路径点，计算直接光照的辐射亮度（光亮度）的数学期望
    Spectrum EmitterEnv2OneV(const PathVertex &v, const Intersection *its_emitter_ptr = nullptr) const
    {
        Vector3 wo = v.wo, normal = v.its.normal();
        auto L_emitter = Spectrum(0), L_env = Spectrum(0);
        //直接采样光源，并按多重重要性采样合并
        if (!v.its.HarshLobe())
        {
            if (its_emitter_ptr)
            {
                if (!EmitterDirectArea(v.its, wo, L_emitter, nullptr, its_emitter_ptr))
                    EmitterDirectArea(v.its, wo, L_emitter);
            }
            else
                EmitterDirectArea(v.its, wo, L_emitter);
        }
        std::unique_ptr<BsdfSampling> b_rec = v.its.Sample(wo);
        if (!b_rec)
            return L_emitter;
        Float cos_theta = std::abs(glm::dot(b_rec->wi, v.its.normal()));
        auto its_pre = Intersection();

        if (!this->bvh_->Intersect(Ray(v.its.pos(), -b_rec->wi), its_pre))
        {
            if (envmap_ != nullptr) //按 BSDF 采样环境光
                L_env = envmap_->radiance(-b_rec->wi) * b_rec->attenuation * (cos_theta / b_rec->pdf);
        }
        else if (!its_pre.absorb() && its_pre.HasEmission()) //按 BSDF 采样来自面光源的直接光照
        {
            if (v.its.HarshLobe())
                L_emitter += its_pre.radiance() * b_rec->attenuation * (cos_theta / b_rec->pdf);
            else
            {
                Float pdf_direct = PdfEmitterDirect(its_pre, b_rec->wi),
                      weight_bsdf = MisWeight(b_rec->pdf, pdf_direct);
                L_emitter += weight_bsdf * its_pre.radiance() * b_rec->attenuation * (cos_theta / b_rec->pdf);
            }
        }
        auto res = L_emitter + L_env;
        return res;
    }

    ///\brief 第二个及之后的某个光源路径点 -> 某个相机路径点，计算辐射亮度（光亮度）的数学期望及概率
    std::pair<Spectrum, Float> PrepareOtherEmitter2OneC(const std::vector<PathVertex> &emitter_path, const int e_index, PathVertex c) const
    {
        const PathVertex &e_first = emitter_path[0];
        const PathVertex &e_pre = emitter_path[e_index - 1];
        PathVertex e = emitter_path[e_index];
        if (!Visible(e.its, c.its))
            return {Spectrum(0), 0};
        c.wi = glm::normalize(c.its.pos() - e.its.pos());
        if (Perpendicular(-c.wi, c.its.normal()))
            return {Spectrum(0), 0};
        e.wo = c.wi;
        if (Perpendicular(e.wo, e.its.normal()))
            return {Spectrum(0), 0};
        c.pdf = c.its.Pdf(c.wi, c.wo);
        if (c.pdf < kEpsilonPdf2)
            return {Spectrum(0), 0};
        //上一个点接收的直接光照
        auto L_pre = Spectrum(0);
        if (!e.its.HarshLobe())
            L_pre += (e_index == 1) ? EmitterEnv2OneV(e) : EmitterEnv2OneV(e, &e_first.its);
        //上一个点接收的间接光照
        if (e_index > 1)
        {
            e.pdf = e.its.Pdf(e.wi, e.wo);
            if (e.pdf > kEpsilonPdf2)
            {
                e.bsdf = e.its.Eval(e.wi, e.wo);
                auto L_pre_indirect = e_pre.L * e.bsdf * (e.cos_theta_abs / e.pdf);
                if (e_index - 1 > rr_depth_)
                    L_pre_indirect /= pdf_rr_;
                L_pre += L_pre_indirect;
            }
        }
        if (L_pre.r + L_pre.g + L_pre.b < kEpsilon)
            return {Spectrum(0), 0};
        c.bsdf = c.its.Eval(c.wi, c.wo);
        c.cos_theta_abs = std::abs(glm::dot(c.wi, c.its.normal()));
        //当前点接收的间接光照
        auto L_indirect = L_pre * c.bsdf * (c.cos_theta_abs / c.pdf);
        if (e_index > rr_depth_)
        {
            L_indirect /= pdf_rr_;
            c.pdf *= pdf_rr_;
        }
        return {L_indirect, c.pdf};
    }
};

NAMESPACE_END(raytracer)