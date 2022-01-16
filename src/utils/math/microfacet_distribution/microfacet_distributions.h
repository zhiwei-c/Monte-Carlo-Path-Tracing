#pragma once

#include "ggx.h"
#include "beckmann.h"

NAMESPACE_BEGIN(simple_renderer)

inline void DeleteDistribPointer(MicrofacetDistribution *&distrib)
{
    if (!distrib)
        return;
    switch (distrib->type())
    {
    case MicrofacetDistribType::kBeckmann:
        delete ((Beckmann *)distrib);
        break;
    case MicrofacetDistribType::kGgx:
        delete ((GGX *)distrib);
        break;
    default:
        std::cerr << "unknown microfacet distribution type" << std::endl;
        exit(1);
    }
    distrib = nullptr;
}

inline MicrofacetDistribution *InitDistrib(MicrofacetDistribType type, Float alpha_u, Float alpha_v)
{
    switch (type)
    {
    case MicrofacetDistribType::kBeckmann:
        return new Beckmann(alpha_u, alpha_v);
        break;
    case MicrofacetDistribType::kGgx:
        return new GGX(alpha_u, alpha_v);
        break;
    default:
        std::cerr << "unknown microfacet distribution type" << std::endl;
        exit(1);
    }
}

NAMESPACE_END(simple_renderer)