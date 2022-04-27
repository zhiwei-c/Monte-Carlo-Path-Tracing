#pragma once

#include "materials/materials.h"

__device__ Float Material::SpecularSamplingWeight(const vec2 &texcoord) const
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

__device__ bool Material::HasEmission() const
{
    switch (type_)
    {
    case kAreaLight:
        return true;
        break;
    }
    return false;
}

__device__ vec3 Material::radiance() const
{
    switch (type_)
    {
    case kAreaLight:
        return radiance_->Color(vec2(0));
        break;
    }
    return vec3(0);
}

__device__ void Material::Sample(BsdfSampling &bs, const vec3 &sample) const
{
    switch (type_)
    {
    case kDiffuse:
        SampleDiffuse(bs, sample);
        break;
    case kDielectric:
        SampleDielectric(bs, sample);
        break;
    case kRoughDielectric:
        SampleRoughDielectric(bs, sample);
        break;
    case kThinDielectric:
        SampleThinDielectric(bs, sample);
        break;
    case kConductor:
        SampleConductor(bs, sample);
        break;
    case kRoughConductor:
        SampleRoughConductor(bs, sample);
        break;
    case kPlastic:
        SamplePlastic(bs, sample);
        break;
    case kRoughPlastic:
        SampleRoughPlastic(bs, sample);
        break;
    }
}

__device__ vec3 Material::Eval(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, bool inside) const
{
    switch (type_)
    {
    case kDiffuse:
        return EvalDiffuse(wi, wo, normal, texcoord, inside);
        break;
    case kDielectric:
        EvalDielectric(wi, wo, normal, texcoord, inside);
        break;
    case kRoughDielectric:
        EvalRoughDielectric(wi, wo, normal, texcoord, inside);
        break;
    case kThinDielectric:
        EvalThinDielectric(wi, wo, normal, texcoord, inside);
        break;
    case kConductor:
        EvalConductor(wi, wo, normal, texcoord, inside);
        break;
    case kRoughConductor:
        EvalRoughConductor(wi, wo, normal, texcoord, inside);
        break;
    case kPlastic:
        EvalPlastic(wi, wo, normal, texcoord, inside);
        break;
    case kRoughPlastic:
        EvalRoughPlastic(wi, wo, normal, texcoord, inside);
        break;
    }
    return vec3(0);
}

__device__ Float Material::Pdf(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, bool inside) const
{
    switch (type_)
    {
    case kDiffuse:
        return PdfDiffuse(wi, wo, normal, texcoord, inside);
        break;
    case kDielectric:
        return PdfDielectric(wi, wo, normal, texcoord, inside);
        break;
    case kRoughDielectric:
        return PdfRoughDielectric(wi, wo, normal, texcoord, inside);
        break;
    case kThinDielectric:
        return PdfThinDielectric(wi, wo, normal, texcoord, inside);
        break;
    case kConductor:
        return PdfConductor(wi, wo, normal, texcoord, inside);
        break;
    case kRoughConductor:
        return PdfRoughConductor(wi, wo, normal, texcoord, inside);
        break;
    case kPlastic:
        return PdfPlastic(wi, wo, normal, texcoord, inside);
        break;
    case kRoughPlastic:
        return PdfRoughPlastic(wi, wo, normal, texcoord, inside);
        break;
    }
    return 0;
}

__global__ void SetMaterialOtherInfo(uint m_idx,
                                     uint material_num,
                                     Material *material_list)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        Material *pre = nullptr;
        if (m_idx > 0)
            pre = material_list + m_idx - 1;
        Material *next = nullptr;
        if (m_idx + 1 < material_num)
            pre = material_list + m_idx + 1;

        material_list[m_idx].SetOtherInfo(m_idx, pre, next);
    }
}
