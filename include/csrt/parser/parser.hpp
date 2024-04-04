#ifndef CSRT__PARSER__PARSER_HPP
#define CSRT__PARSER__PARSER_HPP

#include "../renderer/renderer.hpp"

#include <string>

namespace csrt
{

struct Config
{
    BackendType backend_type;
    Camera::Info camera;
    Integrator::Info integrator;
    std::vector<TextureData> textures;
    std::vector<BsdfInfo> bsdfs;
    std::vector<Instance::Info> instances;
    std::vector<EmitterInfo> emitters;
};

Config LoadConfig(const std::string &filename);

} // namespace csrt

#endif