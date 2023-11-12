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
    std::vector<Bsdf::Info> bsdfs;
    std::vector<Instance::Info> instances;
};

Config LoadConfig(const std::string &filename);

} // namespace csrt
