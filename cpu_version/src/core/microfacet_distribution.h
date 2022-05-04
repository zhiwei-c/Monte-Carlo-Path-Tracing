#pragma once

#include <iostream>
#include <memory>

#include "../microfacet_distribution/ggx.h"
#include "../microfacet_distribution/beckmann.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 创建给定参数的微表面分布函数
inline std::unique_ptr<MicrofacetDistribution> InitDistrib(MicrofacetDistribType type, Float alpha_u, Float alpha_v)
{
    switch (type)
    {
    case MicrofacetDistribType::kBeckmann:
        return std::make_unique<Beckmann>(alpha_u, alpha_v);
        break;
    case MicrofacetDistribType::kGgx:
        return std::make_unique<GGX>(alpha_u, alpha_v);
        break;
    default:
        std::cerr << "unknown microfacet distribution type" << std::endl;
        exit(1);
    }
}

NAMESPACE_END(simple_renderer)