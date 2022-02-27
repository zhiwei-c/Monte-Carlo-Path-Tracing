#include "bdpt.h"

NAMESPACE_BEGIN(simple_renderer)

Spectrum BdptIntegrator::Shade(const Vector3 &eye_pos, const Vector3 &look_dir) const
{
    if (this->bvh_ != nullptr)
    {
        auto its = this->bvh_->Intersect(Ray(eye_pos, look_dir));
        if (its.valid())
        {
            if (its.HasEmission())
                return its.radiance();
            else
                return ProcessBdpt(its, -look_dir);
        }
    }

    if (envmap_ != nullptr)
        return envmap_->GetLe(look_dir);

    return Spectrum(0);
}

Spectrum BdptIntegrator::ProcessBdpt(const Intersection &its, const Vector3 &wo) const
{
    auto emitter_path = CreateEmitterPath();
    auto camera_path = CreateCameraPath(its, wo);

    for (int c_idx = camera_path.size() - 1; c_idx >= 0; c_idx--)
    {
        const auto &c = camera_path[c_idx];

        //直接光照
        auto L_direct = EmitterEnv2OneV(c, !emitter_path.empty() ? &emitter_path[0].its : nullptr);

        //来自光源路径的间接光照
        std::vector<Spectrum> L_indirects;
        std::vector<Float> pdfs;
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
        //来自照相机路径的间接光照
        if (c_idx < camera_path.size() - 1)
        {
            auto L_temp = camera_path[c_idx + 1].L * c.bsdf * (c.cos_theta_abs / c.pdf);
            auto pdf_temp = c.pdf;
            if (c_idx > rr_depth_)
            {
                L_temp /= pdf_rr_;
                pdf_temp *= pdf_rr_;
            }
            if (L_temp.r + L_temp.g + L_temp.b > kEpsilon)
            {
                L_indirects.push_back(L_temp);
                pdfs.push_back(pdf_temp);
            }
        }
        //总间接光照
        auto L_indirect = WeightPowerHeuristic(L_indirects, pdfs);

        camera_path[c_idx].L = L_indirect + L_direct;
    }

    return camera_path[0].L;
}

///\brief 创建光源路径
std::vector<PathVertex> BdptIntegrator::CreateEmitterPath() const
{
    std::vector<PathVertex> emitter_path;

    //从发光物体上直接采样，生成第一个光源路径点
    auto its_first = Intersection();
    if (!SampleEmitterDirectIts(its_first))
        return emitter_path;

    //采样光线从第一个光源路径点射出的方向
    auto wo_first = Vector3(0);
    std::tie(wo_first, std::ignore) = HemisCos();
    wo_first = ToWorld(wo_first, its_first.normal());
    emitter_path.push_back({its_first, Vector3(0), wo_first});

    //生成第二个及之后光源路径点
    int depth = 1;
    while (max_depth_ > 0 && emitter_path.size() < max_depth_ ||
           max_depth_ <= 0 && (depth <= rr_depth_ || UniformFloat() < pdf_rr_))
    {
        auto &e = emitter_path.back();

        auto its_next = this->bvh_->Intersect(Ray(e.its.pos(), e.wo));
        if (!its_next.valid() || its_next.HasEmission())
            break;
        emitter_path.push_back({its_next, e.wo, Vector3(0)});
        depth += 1;
        auto &e_next = emitter_path.back();
        e_next.cos_theta_abs = std::fabs(glm::dot(e_next.wi, e_next.its.normal()));

        auto bs_next = e_next.its.Sample(-e_next.wi, false);
        auto wo_next = -bs_next.wi;
        auto pdf_next_pseudo = bs_next.pdf;
        if (pdf_next_pseudo < kEpsilonPdf)
            break;

        auto pdf_next = its_next.Pdf(e_next.wi, wo_next);
        if (pdf_next < kEpsilonPdf2)
            break;

        e_next.wo = wo_next;
        e_next.pdf = pdf_next;
        e_next.bsdf = its_next.Eval(e_next.wi, e_next.wo);
    }

    auto &e_first = emitter_path[0];
    e_first.L = e_first.its.radiance();

    //第一个光源路径点 -> 第二个光源路径点 -> 第三个光源路径点，预计算传递的辐射亮度（光亮度）期望
    if (emitter_path.size() > 2)
    {
        auto &e_second = emitter_path[1];
        e_second.L = EmitterEnv2OneV(e_second);
    }

    //第二个及之后的光源路径点 -> 下一个光源路径点 -> 下一个光源路径点，预计算传递的辐射亮度（光亮度）期望
    for (int i = 2; i < emitter_path.size() - 1; i++)
    {
        auto &e_pre = emitter_path[i - 1];
        auto &e = emitter_path[i];

        auto L_indirect = e_pre.L * e.bsdf * (e.cos_theta_abs / e.pdf);
        if (i > rr_depth_)
            L_indirect /= pdf_rr_;

        auto L_direct_env = EmitterEnv2OneV(e, &e_first.its);
        e.L = L_indirect + L_direct_env;
    }
    return emitter_path;
}

///\brief 创建照相机路径
std::vector<PathVertex> BdptIntegrator::CreateCameraPath(const Intersection &its_first, const Vector3 &wo_first) const
{
    std::vector<PathVertex> camera_path;
    camera_path.push_back({its_first, Vector3(0), wo_first});
    int depth = 1;
    while (max_depth_ > 0 && camera_path.size() < max_depth_ ||
           max_depth_ <= 0 && (depth <= rr_depth_ || UniformFloat() < pdf_rr_))
    {
        auto &c = camera_path.back();
        auto b_rec = c.its.Sample(c.wo);
        if (b_rec.pdf < kEpsilonPdf2)
            break;

        auto its_pre = this->bvh_->Intersect(Ray(c.its.pos(), -b_rec.wi));
        if (!its_pre.valid() || its_pre.HasEmission())
            break;

        c.wi = b_rec.wi;
        c.cos_theta_abs = std::fabs(glm::dot(b_rec.wi, c.its.normal()));
        c.pdf = b_rec.pdf;
        c.bsdf = b_rec.weight;
        camera_path.push_back({its_pre, Vector3(0), b_rec.wi});
        depth += 1;
    }
    return camera_path;
}

///\brief 光源与环境光 -> 某个路径点 -> 下一个点
Spectrum BdptIntegrator::EmitterEnv2OneV(const PathVertex &v, const Intersection *its_emitter_ptr) const
{
    auto wo = v.wo;
    auto normal = v.its.normal();

    auto L_emitter = Spectrum(0),
         L_env = Spectrum(0);

    //直接采样光源，并按多重重要性采样合并
    if (its_emitter_ptr)
    {
        if (!EmitterDirectArea(v.its, wo, L_emitter, its_emitter_ptr))
            EmitterDirectArea(v.its, wo, L_emitter);
    }
    else
        EmitterDirectArea(v.its, wo, L_emitter);

    auto b_rec = v.its.Sample(wo);
    auto cos_theta = std::fabs(glm::dot(b_rec.wi, v.its.normal()));
    auto its_pre = this->bvh_->Intersect(Ray(v.its.pos(), -b_rec.wi));
    if (!its_pre.valid())
    {
        //按 BSDF 采样环境光
        if (envmap_ != nullptr)
            L_env = envmap_->GetLe(-b_rec.wi) * b_rec.weight * (cos_theta / b_rec.pdf);
    }
    else if (its_pre.HasEmission())
    {
        //按 BSDF 采样光源，并按多重重要性采样合并
        auto pdf_direct = PdfEmitterDirect(its_pre, b_rec.wi);
        auto weight_bsdf = MisWeight(b_rec.pdf, pdf_direct);
        L_emitter += weight_bsdf * its_pre.radiance() * b_rec.weight * (cos_theta / b_rec.pdf);
    }

    auto res = L_emitter + L_env;
    return res;
}

///\brief 第二个及之后的某个光源路径点 -> 某个相机路径点 -> 下一个点
std::pair<Spectrum, Float> BdptIntegrator::PrepareOtherEmitter2OneC(const std::vector<PathVertex> &emitter_path, const int e_index, PathVertex v) const
{

    const auto &e_first = emitter_path[0];
    const auto &e_pre = emitter_path[e_index - 1];
    auto e = emitter_path[e_index];

    if (!Visible(e.its, v.its))
        return {Spectrum(0), 0};

    v.wi = glm::normalize(v.its.pos() - e.its.pos());
    if (Perpendicular(-v.wi, v.its.normal()))
        return {Spectrum(0), 0};

    e.wo = v.wi;
    if (Perpendicular(e.wo, e.its.normal()))
        return {Spectrum(0), 0};

    v.pdf = v.its.Pdf(v.wi, v.wo);
    if (v.pdf < kEpsilonPdf2)
        return {Spectrum(0), 0};

    Spectrum L_pre = e_index == 1 ? EmitterEnv2OneV(e) : EmitterEnv2OneV(e, &e_first.its);
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

    v.bsdf = v.its.Eval(v.wi, v.wo);
    v.cos_theta_abs = std::abs(glm::dot(v.wi, v.its.normal()));
    auto L_indirect = L_pre * v.bsdf * (v.cos_theta_abs / v.pdf);
    if (e_index > rr_depth_)
    {
        L_indirect /= pdf_rr_;
        v.pdf *= pdf_rr_;
    }
    return {L_indirect, v.pdf};
}

NAMESPACE_END(simple_renderer)