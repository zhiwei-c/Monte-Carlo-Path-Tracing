#include "bdpt.h"

NAMESPACE_BEGIN(simple_renderer)

Spectrum BdptIntegrator::Shade(const Vector3 &eye_pos, const Vector3 &look_dir) const
{
    if (this->bvh_ != nullptr)
    {
        auto hit = this->bvh_->Intersect(Ray(eye_pos, look_dir));
        if (hit.valid())
        {
            if (hit.HasEmission())
            {
                return hit.radiance();
            }
            else
            {
                return ProcessBdpt(hit, -look_dir);
            }
        }
    }

    if (envmap_ != nullptr)
        return envmap_->GetLe(look_dir);

    return Spectrum(0);
}

Spectrum BdptIntegrator::ProcessBdpt(const Intersection &start_obj, const Vector3 &start_wo) const
{
    auto emitter_path = CreateEmitterPath();
    auto camera_path = CreateCameraPath(start_obj, start_wo);

    for (int c_idx = camera_path.size() - 1; c_idx >= 0; c_idx--)
    {
        const auto &c = camera_path[c_idx];

        auto L_direct = Spectrum(0);
        if (!emitter_path.empty())
            L_direct = PrepareFisrtEmitter2OneV(emitter_path[0], c);

        std::vector<Spectrum> L_indirects;
        std::vector<Float> pdfs;
        for (int e_idx = 1; e_idx < emitter_path.size(); e_idx++)
        {
            auto [L_temp, pdf_temp] = PrepareOtherEmitter2OneC(emitter_path, e_idx, c);
            if (pdf_temp > kEpsilonPdf)
            {
                L_indirects.push_back(L_temp);
                pdfs.push_back(pdf_temp);
            }
        }
        if (c_idx < camera_path.size() - 1)
        {
            auto L_temp = camera_path[c_idx + 1].L * c.bsdf * (c.cos_theta_abs / c.pdf);
            L_indirects.push_back(L_temp);
            pdfs.push_back(c.pdf);
        }
        auto L_indirect = WeightPowerHeuristic(L_indirects, pdfs);

        camera_path[c_idx].L = L_indirect + L_direct;
    }
    return camera_path[0].L;
}

std::vector<PathVertex> BdptIntegrator::CreateCameraPath(const Intersection &its_first, const Vector3 &start_wo) const
{
    std::vector<PathVertex> camera_path;
    camera_path.push_back({its_first, Vector3(0), start_wo});

    while (max_depth_ > 0 && camera_path.size() < max_depth_ ||
           max_depth_ <= 0 && UniformFloat() < pdf_rr_)
    {
        auto &c = camera_path.back();
        auto b_rec = c.its.Sample(c.wo);
        if (b_rec.pdf > kEpsilonPdf2)
        {
            auto its_pre = this->bvh_->Intersect(Ray(c.its.pos(), -b_rec.wi));
            if (!its_pre.valid() || its_pre.HasEmission())
                break;

            c.wi = b_rec.wi;
            c.cos_theta_abs = std::fabs(glm::dot(c.wi, c.its.normal()));
            c.pdf = b_rec.pdf;
            c.bsdf = b_rec.weight;
            camera_path.push_back({its_pre, Vector3(0), b_rec.wi});
        }
    }
    return camera_path;
}

std::vector<PathVertex> BdptIntegrator::CreateEmitterPath() const
{
    std::vector<PathVertex> emitter_path;

    auto its_first = Intersection();
    if (!SampleEmitterDirectIts(its_first))
        return emitter_path;

    auto wo_first = Vector3(0);
    std::tie(wo_first, std::ignore) = HemisUniform();
    wo_first = ToWorld(wo_first, its_first.normal());
    emitter_path.push_back({its_first, Vector3(0), wo_first});

    while (max_depth_ > 0 && emitter_path.size() < max_depth_ ||
           max_depth_ <= 0 && UniformFloat() < pdf_rr_)
    {
        auto &e = emitter_path.back();
        auto its_next = this->bvh_->Intersect(Ray(e.pos(), e.wo));
        if (!its_next.valid() || its_next.HasEmission())
            break;

        emitter_path.push_back({its_next, e.wo, Vector3(0)});
        auto &e_next = emitter_path.back();
        e_next.cos_theta_abs = std::fabs(glm::dot(e_next.wi, e_next.normal()));

        auto wo_next = Vector3(0);
        Float pdf_next_pseudo = 0;
        std::tie(wo_next, pdf_next_pseudo) = e_next.SampleWo(e_next.wi);
        if (pdf_next_pseudo < kEpsilonPdf)
            break;

        auto pdf_next = its_next.Pdf(e_next.wi, wo_next);
        if (pdf_next < kEpsilonPdf2)
            break;

        e_next.wo = wo_next;
        e_next.pdf = pdf_next;
        e_next.bsdf = its_next.Eval(e_next.wi, e_next.wo);
    }

    emitter_path[0].L = emitter_path[0].its.radiance();

    if (emitter_path.size() > 2)
    {
        auto &e_first = emitter_path[0];
        auto &e = emitter_path[1];

        Spectrum L_direct(0);
        EmitterDirectArea(e.its, e.wo, L_direct);
        Float distance_sqr = std::pow(glm::length(e_first.pos() - e.pos()), 2);
        auto pdf_direct = PdfEmitterDirect(e_first.its, e.wi, &distance_sqr);
        auto weight_bsdf = MisWeight(e.pdf, pdf_direct);
        L_direct += weight_bsdf * e_first.L * e.bsdf * (e.cos_theta_abs / e.pdf);

        Spectrum L_env(0);
        if (envmap_ != nullptr)
        {
            BsdfSampling b_rec = e.SampleWi(e.wo);
            auto its_pre = this->bvh_->Intersect(Ray(e.pos(), -b_rec.wi));
            if (!its_pre.valid())
            {
                auto cos_theta = std::fabs(glm::dot(b_rec.wi, e.normal()));
                L_env = envmap_->GetLe(-b_rec.wi) * b_rec.weight * (cos_theta / b_rec.pdf);
            }
        }

        e.L = L_direct + L_env;
    }

    for (int i = 2; i < emitter_path.size() - 1; i++)
    {
        auto &e_pre = emitter_path[i - 1];
        auto &e = emitter_path[i];

        auto L_indirect = e_pre.L * e.bsdf * (e.cos_theta_abs / e.pdf);

        Spectrum L_direct(0);
        Spectrum L_env(0);

        EmitterDirectArea(e.its, e.wo, L_direct);

        auto b_rec = e.SampleWi(e.wo);
        auto cos_theta = std::fabs(glm::dot(b_rec.wi, e.normal()));
        auto its_pre = this->bvh_->Intersect(Ray(e.pos(), -b_rec.wi));
        if (!its_pre.valid())
        {
            if (envmap_ != nullptr)
                L_env = envmap_->GetLe(-b_rec.wi) * b_rec.weight * (cos_theta / b_rec.pdf);
        }
        else if (its_pre.HasEmission())
        {
            auto pdf_direct = PdfEmitterDirect(its_pre, b_rec.wi);
            auto weight_bsdf = MisWeight(b_rec.pdf, pdf_direct);
            L_direct += weight_bsdf * its_pre.radiance() * b_rec.weight * (cos_theta / b_rec.pdf);
        }

        e.L = L_indirect + L_direct + L_env;
    }
    return emitter_path;
}

/**
 * \brief 第一个光源路径点（光源）或环境光 -> 某个路径点
 */
Spectrum BdptIntegrator::PrepareFisrtEmitter2OneV(const PathVertex &e_first, const PathVertex &v) const
{
    auto wo = v.wo;
    auto normal = v.normal();

    auto L_emitter = Spectrum(0),
         L_env = Spectrum(0);

    bool flag = false;
    if (Visible(e_first.its, v.its))
    {
        auto wi = glm::normalize(v.pos() - e_first.pos());
        auto pdf_bsdf = v.Pdf(wi, wo);
        if (pdf_bsdf > kEpsilonPdf2)
        {
            auto cos_theta = std::abs(glm::dot(wi, normal));
            auto bsdf = v.Eval(wi, wo);
            Float distance_sqr = std::pow(glm::length(e_first.pos() - v.pos()), 2);
            auto pdf_direct = PdfEmitterDirect(e_first.its, wi, &distance_sqr);
            if (pdf_direct > 0)
            {
                auto weight_direct = MisWeight(pdf_direct, pdf_bsdf);
                L_emitter += weight_direct * e_first.L * bsdf * (cos_theta / pdf_direct);
                flag = true;
            }
        }
    }
    if (!flag)
        EmitterDirectArea(v.its, wo, L_emitter);

    auto b_rec = v.SampleWi(wo);
    auto cos_theta = std::fabs(glm::dot(b_rec.wi, v.normal()));
    auto its_pre = this->bvh_->Intersect(Ray(v.pos(), -b_rec.wi));
    if (!its_pre.valid())
    {
        if (envmap_ != nullptr)
            L_env = envmap_->GetLe(-b_rec.wi) * b_rec.weight * (cos_theta / b_rec.pdf);
    }
    else if (its_pre.HasEmission())
    {
        auto pdf_direct = PdfEmitterDirect(its_pre, b_rec.wi);
        auto weight_bsdf = MisWeight(b_rec.pdf, pdf_direct);
        L_emitter += weight_bsdf * its_pre.radiance() * b_rec.weight * (cos_theta / b_rec.pdf);
    }

    auto res = L_emitter + L_env;
    return res;
}

/**
 * \brief 第二个及之后的某个光源路径点 -> 某个相机路径点
 */
std::pair<Spectrum, Float> BdptIntegrator::PrepareOtherEmitter2OneC(const std::vector<PathVertex> &emitter_path, const int e_index, const PathVertex &c) const
{
    const auto &e_first = emitter_path[0];
    const auto &e_pre = emitter_path[e_index - 1];
    const auto &e = emitter_path[e_index];

    auto wo = c.wo;
    auto normal = c.normal();

    if (Visible(e.its, c.its))
    {
        auto wi = glm::normalize(c.pos() - e.pos());
        auto pdf = c.Pdf(wi, wo);
        if (pdf > kEpsilonPdf2)
        {
            auto e_second_tmp = e;
            e_second_tmp.wo = wi;

            Spectrum L_pre = PrepareFisrtEmitter2OneV(e_first, e_second_tmp);
            if (e_index > 1)
            {
                auto pdf_e = e.Pdf(e.wi, wi);
                if (pdf_e > kEpsilonPdf2)
                {
                    auto bsdf_e = e.Eval(e.wi, wi);
                    L_pre += e_pre.L * bsdf_e * (e.cos_theta_abs / pdf_e);
                }
            }
            auto bsdf = c.Eval(wi, wo);
            auto cos_theta = std::abs(glm::dot(wi, normal));
            auto L_indirect = L_pre * bsdf * (cos_theta / pdf);
            return {L_indirect, pdf};
        }
    }
    return {Spectrum(0), 0};
}

NAMESPACE_END(simple_renderer)