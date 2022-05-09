#pragma once

#include "../emitters/area_light.h"

#include "../bsdfs/diffuse.h"
#include "../bsdfs/dielectric.h"
#include "../bsdfs/rough_dielectric.h"
#include "../bsdfs/thin_dielectric.h"
#include "../bsdfs/conductor.h"
#include "../bsdfs/rough_conductor.h"
#include "../bsdfs/plastic.h"
#include "../bsdfs/rough_plastic.h"

__global__ void CreateMaterials(uint material_num,
                                MaterialInfo *material_info_list,
                                Texture *texture_list,
                                Material **material_list)
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


__global__ void FreeMaterials(uint material_num,
                                Material **material_list)
{
    if (threadIdx.x != 0 && blockIdx.x != 0)
        return;
        
    for(uint i = 0; i < material_num; i++)
    {
        delete material_list[i];
        material_list[i] = nullptr;
    }
}
