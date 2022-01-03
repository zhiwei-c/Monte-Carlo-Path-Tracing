#pragma once

#include <iostream>

#include "diffuse.h"
#include "glossy.h"

#include "dielectric.h"
#include "rough_dielectric.h"
#include "thin_dielectric.h"
#include "conductor.h"
#include "rough_conductor.h"
#include "plastic.h"
#include "rough_plastic.h"

#include "area_light.h"

NAMESPACE_BEGIN(simple_renderer)

inline void DeleteMaterialPointer(Material *&material)
{
    if (!material)
        return;
    switch (material->type())
    {
    case MaterialType::kDiffuse:
        delete ((Diffuse *)material);
        break;
    case MaterialType::kGlossy:
        delete ((Glossy *)material);
        break;
    case MaterialType::kDielectric:
        delete ((Dielectric *)material);
        break;
    case MaterialType::kRoughDielectric:
        delete ((RoughDielectric *)material);
        break;
    case MaterialType::kThinDielectric:
        delete ((ThinDielectric *)material);
        break;
    case MaterialType::kConductor:
        delete ((Conductor *)material);
        break;
    case MaterialType::kRoughConductor:
        delete ((RoughConductor *)material);
        break;
    case MaterialType::kPlastic:
        delete ((Plastic *)material);
        break;
    case MaterialType::kRoughPlastic:
        delete ((RoughPlastic *)material);
        break;
    case MaterialType::kAreaLight:
        delete ((AreaLight *)material);
        break;
    default:
        std::cerr << "unknown material type" << std::endl;
        exit(1);
    }
    material = nullptr;
}

NAMESPACE_END(simple_renderer)