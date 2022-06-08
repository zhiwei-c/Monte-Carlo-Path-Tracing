#pragma once

#include "../bsdfs/area_light.h"

#include "../bsdfs/diffuse.h"
#include "../bsdfs/dielectric.h"
#include "../bsdfs/rough_dielectric.h"
#include "../bsdfs/thin_dielectric.h"
#include "../bsdfs/conductor.h"
#include "../bsdfs/rough_conductor.h"
#include "../bsdfs/plastic.h"
#include "../bsdfs/rough_plastic.h"

__global__ void CreateBsdfs(uint bsdf_num, BsdfInfo *bsdf_info_list, Texture *texture_list, Bsdf **bsdf_list)
{
    if (threadIdx.x != 0 && blockIdx.x != 0)
        return;

    for (uint bsdf_idx = 0; bsdf_idx < bsdf_num; bsdf_idx++)
    {
        switch (bsdf_info_list[bsdf_idx].type)
        {
        case kAreaLight:
            InitAreaLight(bsdf_idx, bsdf_info_list, texture_list, bsdf_list);
            break;
        case kDielectric:
            InitDielectric(bsdf_idx, bsdf_info_list, texture_list, bsdf_list);
            break;
        case kRoughDielectric:
            InitRoughDielectric(bsdf_idx, bsdf_info_list, texture_list, bsdf_list);
            break;
        case kThinDielectric:
            InitThinDielectric(bsdf_idx, bsdf_info_list, texture_list, bsdf_list);
            break;
        case kConductor:
            InitConductor(bsdf_idx, bsdf_info_list, texture_list, bsdf_list);
            break;
        case kRoughConductor:
            InitRoughConductor(bsdf_idx, bsdf_info_list, texture_list, bsdf_list);
            break;
        case kPlastic:
            InitPlastic(bsdf_idx, bsdf_info_list, texture_list, bsdf_list);
            break;
        case kRoughPlastic:
            InitRoughPlastic(bsdf_idx, bsdf_info_list, texture_list, bsdf_list);
            break;
        default:
            InitDiffuse(bsdf_idx, bsdf_info_list, texture_list, bsdf_list);
            break;
        }
    }
}

__global__ void FreeBsdfs(uint bsdf_num,
                          Bsdf **bsdf_list)
{
    if (threadIdx.x != 0 && blockIdx.x != 0)
        return;

    for (uint i = 0; i < bsdf_num; i++)
    {
        delete bsdf_list[i];
        bsdf_list[i] = nullptr;
    }
}
