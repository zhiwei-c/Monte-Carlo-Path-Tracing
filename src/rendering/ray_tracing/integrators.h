#pragma once

#include "path.h"
#include "bdpt.h"

NAMESPACE_BEGIN(simple_renderer)

inline void DeleteIntegrator(Integrator *&integrator)
{
    if (!integrator)
        return;
    switch (integrator->type())
    {
    case IntegratorType::kPath:
        delete ((PathIntegrator *)integrator);
        break;
    case IntegratorType::kBdpt:
        delete ((BdptIntegrator *)integrator);
        break;
    default:
        std::cerr << "unknown integrator type" << std::endl;
        exit(1);
    }
    integrator = nullptr;
}

NAMESPACE_END(simple_renderer)