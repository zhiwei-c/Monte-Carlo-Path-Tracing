#include "renderer.cuh"

#include <string>

namespace csrt
{

struct Config
{
    BackendType backend_type;
    Camera::Info camera;
    Integrator::Info integrator;
    std::vector<Texture::Info> textures;
    std::vector<BSDF::Info> bsdfs;
    std::vector<Instance::Info> instances;
    std::vector<Emitter::Info> emitters;
};

Config LoadConfig(const std::string &filename);

} // namespace csrt
