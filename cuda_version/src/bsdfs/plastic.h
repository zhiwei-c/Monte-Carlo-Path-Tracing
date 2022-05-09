#pragma once

#include "../core/material_base.h"

class Plastic : public Material
{
public:
    __device__ Plastic(uint idx,
                       bool twosided,
                       Texture *bump_map,
                       Texture *opacity_map,
                       vec3 eta,
                       Texture *diffuse_reflectance,
                       Texture *specular_reflectance,
                       bool nonlinear)
        : Material(idx, kPlastic, twosided, bump_map, opacity_map),
          eta_inv_d_(1.0 / eta.x),
          diffuse_reflectance_(diffuse_reflectance),
          specular_reflectance_(specular_reflectance),
          nonlinear_(nonlinear),
          fdr_int_(AverageFresnel(eta.x)) {}

    __device__ void Sample(BsdfSampling &bs, const vec3 &sample) const override
    {
        auto specular_sampling_weight = SpecularSamplingWeight(bs.texcoord);

        auto kr_o = Fresnel(-bs.wo, bs.normal, eta_inv_d_);
        auto pdf_specular = kr_o * specular_sampling_weight,
             pdf_diffuse = (1.0 - kr_o) * (1.0 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);

        bs.pdf = 0;
        auto kr_i = static_cast<Float>(0);
        auto specular = false;
        if (sample.x < pdf_specular)
        {
            bs.wi = -Reflect(-bs.wo, bs.normal);
            kr_i = kr_o;
            bs.pdf += pdf_specular;
            specular = true;
        }
        else
        {
            auto wi_local = vec3(0);
            auto pdf = static_cast<Float>(0);
            HemisCos(sample.y, sample.z, wi_local, pdf);
            bs.wi = -ToWorld(wi_local, bs.normal);
            kr_i = Fresnel(bs.wi, bs.normal, eta_inv_d_);
            pdf_specular = kr_i * specular_sampling_weight;
            pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);
            pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        }
        pdf_diffuse = 1.0 - pdf_specular;
        bs.pdf += pdf_diffuse * PdfHemisCos(ToLocal(bs.wo, bs.normal));
        if (bs.pdf < kEpsilonPdf)
            return;

        auto diffuse_reflectance = diffuse_reflectance_ ? diffuse_reflectance_->Color(bs.texcoord) : vec3(0.5);
        if (nonlinear_)
            bs.attenuation = diffuse_reflectance / (vec3(1) - diffuse_reflectance * fdr_int_);
        else
            bs.attenuation = diffuse_reflectance / (1.0 - fdr_int_);
        bs.attenuation *= eta_inv_d_ * eta_inv_d_ * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;

        if (specular)
        {
            auto attenuation = vec3(kr_i);
            if (specular_reflectance_)
                attenuation *= specular_reflectance_->Color(bs.texcoord);
            bs.attenuation += attenuation;
        }

        bs.valid = true;
    }

    __device__ vec3 Eval(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const override
    {
        auto albedo = vec3(0);
        auto diffuse_reflectance = diffuse_reflectance_ ? diffuse_reflectance_->Color(texcoord) : vec3(0.5);
        if (nonlinear_)
            albedo = diffuse_reflectance / (vec3(1) - diffuse_reflectance * fdr_int_);
        else
            albedo = diffuse_reflectance / (1.0 - fdr_int_);

        auto kr_i = Fresnel(wi, normal, eta_inv_d_);
        auto kr_o = Fresnel(-wo, normal, eta_inv_d_);
        albedo *= eta_inv_d_ * eta_inv_d_ * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;

        if (SameDirection(Reflect(wi, normal), wo))
        {
            auto attenuation = vec3(kr_i);
            if (specular_reflectance_)
                attenuation *= specular_reflectance_->Color(texcoord);
            albedo += attenuation;
        }

        return albedo;
    }

    __device__ Float Pdf(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const override
    {
        // 表面法线方向，光线入射和出射需在介质同侧
        if (NotSameHemis(wo, normal))
            return 0;

        auto kr = Fresnel(wi, normal, eta_inv_d_);
        auto specular_sampling_weight = SpecularSamplingWeight(texcoord);

        auto pdf_specular = kr * specular_sampling_weight,
             pdf_diffuse = (1.0 - kr) * (1.0 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        pdf_diffuse = 1.0 - pdf_specular;

        auto wo_local = ToLocal(wo, normal);
        auto local_pdf_diffuse = PdfHemisCos(wo_local);
        auto pdf = pdf_diffuse * local_pdf_diffuse;

        if (SameDirection(wo, Reflect(wi, normal)))
        {
            pdf += pdf_specular;
        }
        return pdf;
    }

    __device__ bool Transparent(const vec2 &texcoord, const vec2 &sample) const override
    {
        if (Material::Transparent(texcoord, sample))
            return true;

        if (diffuse_reflectance_)
        {
            if (diffuse_reflectance_->type() == kBitmap &&
                diffuse_reflectance_->Transparent(texcoord, sample.y))
                return true;
        }
        return false;
    }

private:
    Float eta_inv_d_;
    Texture *diffuse_reflectance_;
    Texture *specular_reflectance_;
    bool nonlinear_;
    Float fdr_int_;

    __device__ Float SpecularSamplingWeight(const vec2 &texcoord) const
    {
        if (!diffuse_reflectance_ && !specular_reflectance_)
            return 2.0 / 3.0;
        else if (!diffuse_reflectance_)
        {
            auto ks = specular_reflectance_->Color(texcoord);
            auto s_sum = ks.x + ks.y + ks.z;
            return s_sum / (1.5 + s_sum);
        }
        else if (!specular_reflectance_)
        {
            auto kd = diffuse_reflectance_->Color(texcoord);
            auto d_sum = kd.x + kd.y + kd.z;
            return 3.0 / (d_sum + 3.0);
        }
        else
        {
            auto ks = specular_reflectance_->Color(texcoord);
            auto s_sum = ks.x + ks.y + ks.z;
            auto kd = diffuse_reflectance_->Color(texcoord);
            auto d_sum = kd.x + kd.y + kd.z;
            return s_sum / (d_sum + s_sum);
        }
    }
};

__device__ inline void InitPlastic(uint m_idx,
                                   MaterialInfo *material_info_list,
                                   Texture *texture_list,
                                   Material **&material_list)
{
    auto bump_map = static_cast<Texture *>(nullptr);
    if (material_info_list[m_idx].bump_map_idx != kUintMax)
        bump_map = texture_list + material_info_list[m_idx].bump_map_idx;

    auto opacity_map = static_cast<Texture *>(nullptr);
    if (material_info_list[m_idx].opacity_idx != kUintMax)
        opacity_map = texture_list + material_info_list[m_idx].opacity_idx;

    auto diffuse_reflectance = static_cast<Texture *>(nullptr);
    if (material_info_list[m_idx].diffuse_reflectance_idx != kUintMax)
        diffuse_reflectance = texture_list + material_info_list[m_idx].diffuse_reflectance_idx;

    auto specular_reflectance = static_cast<Texture *>(nullptr);
    if (material_info_list[m_idx].specular_reflectance_idx != kUintMax)
        specular_reflectance = texture_list + material_info_list[m_idx].specular_reflectance_idx;

    material_list[m_idx] = new Plastic(m_idx,
                                       material_info_list[m_idx].twosided,
                                       bump_map,
                                       opacity_map,
                                       material_info_list[m_idx].eta,
                                       diffuse_reflectance,
                                       specular_reflectance,
                                       material_info_list[m_idx].nonlinear);
}