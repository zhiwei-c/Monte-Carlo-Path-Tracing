#pragma once

#include "materials/area_light.h"
#include "materials/diffuse.h"
#include "materials/dielectric.h"
#include "materials/rough_dielectric.h"
#include "materials/thin_dielectric.h"
#include "materials/conductor.h"
#include "materials/rough_conductor.h"
#include "materials/plastic.h"
#include "materials/rough_plastic.h"

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

__device__ vec3 Material::Eval(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const
{
    auto attenuation = vec3(0);
    switch (type_)
    {
    case kDiffuse:
        attenuation = EvalDiffuse(wi, wo, normal, texcoord, inside);
        break;
    case kDielectric:
        attenuation = EvalDielectric(wi, wo, normal, texcoord, inside);
        break;
    case kRoughDielectric:
        attenuation = EvalRoughDielectric(wi, wo, normal, texcoord, inside);
        break;
    case kThinDielectric:
        attenuation = EvalThinDielectric(wi, wo, normal, texcoord, inside);
        break;
    case kConductor:
        attenuation = EvalConductor(wi, wo, normal, texcoord, inside);
        break;
    case kRoughConductor:
        attenuation = EvalRoughConductor(wi, wo, normal, texcoord, inside);
        break;
    case kPlastic:
        attenuation = EvalPlastic(wi, wo, normal, texcoord, inside);
        break;
    case kRoughPlastic:
        attenuation = EvalRoughPlastic(wi, wo, normal, texcoord, inside);
        break;
    }
    return attenuation;
}

__device__ Float Material::Pdf(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const
{
    Float pdf = 0;
    switch (type_)
    {
    case kDiffuse:
        pdf = PdfDiffuse(wi, wo, normal, texcoord, inside);
        break;
    case kDielectric:
        pdf = PdfDielectric(wi, wo, normal, texcoord, inside);
        break;
    case kRoughDielectric:
        pdf = PdfRoughDielectric(wi, wo, normal, texcoord, inside);
        break;
    case kThinDielectric:
        pdf = PdfThinDielectric(wi, wo, normal, texcoord, inside);
        break;
    case kConductor:
        pdf = PdfConductor(wi, wo, normal, texcoord, inside);
        break;
    case kRoughConductor:
        pdf = PdfRoughConductor(wi, wo, normal, texcoord, inside);
        break;
    case kPlastic:
        pdf = PdfPlastic(wi, wo, normal, texcoord, inside);
        break;
    case kRoughPlastic:
        pdf = PdfRoughPlastic(wi, wo, normal, texcoord, inside);
        break;
    }
    return pdf;
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

__global__ void CreateMaterials(uint material_num,
                                MaterialInfo *material_info_list,
                                Texture *texture_list,
                                Material *material_list)
{
    if (threadIdx.x != 0 && blockIdx.x != 0)
        return;

    for (uint material_idx = 0; material_idx < material_num; material_idx++)
    {
        switch (material_info_list[material_idx].type)
        {
        case kAreaLight:
            InitAreaLight(material_idx, material_info_list, texture_list, material_list);
            break;
        case kDielectric:
            InitDielectric(material_idx, material_info_list, texture_list, material_list);
            break;
        case kRoughDielectric:
            InitRoughDielectric(material_idx, material_info_list, texture_list, material_list);
            break;
        case kThinDielectric:
            InitThinDielectric(material_idx, material_info_list, texture_list, material_list);
            break;
        case kConductor:
            InitConductor(material_idx, material_info_list, texture_list, material_list);
            break;
        case kRoughConductor:
            InitRoughConductor(material_idx, material_info_list, texture_list, material_list);
            break;
        case kPlastic:
            InitPlastic(material_idx, material_info_list, texture_list, material_list);
            break;
        case kRoughPlastic:
            InitRoughPlastic(material_idx, material_info_list, texture_list, material_list);
            break;
        default:
            InitDiffuse(material_idx, material_info_list, texture_list, material_list);
            break;
        }
    }
}
