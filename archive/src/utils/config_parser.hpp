#pragma once

#include "../global.hpp"
#include "../renderer.hpp"

NAMESPACE_BEGIN(raytracer)

void LoadMitsubaConfig(const std::string& filename, Renderer &renderer);

NAMESPACE_END(raytracer)