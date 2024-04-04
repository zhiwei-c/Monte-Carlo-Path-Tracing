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
    IntegratorInfo integrator;
    std::vector<TextureData> textures;
    std::vector<BsdfInfo> bsdfs;
    std::vector<InstanceInfo> instances;
    std::vector<EmitterInfo> emitters;
};

Config LoadConfig(const std::string &filename);

} // namespace csrt

#endif