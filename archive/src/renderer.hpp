#pragma once

#include <vector>

#include "accelerators/bvh.hpp"
#include "bsdfs/conductor.hpp"
#include "bsdfs/clear_coated_conductor.hpp"
#include "bsdfs/dielectric.hpp"
#include "bsdfs/diffuse.hpp"
#include "bsdfs/plastic.hpp"
#include "bsdfs/rough_conductor.hpp"
#include "bsdfs/rough_dielectric.hpp"
#include "bsdfs/rough_plastic.hpp"
#include "bsdfs/thin_dielectric.hpp"
#include "core/camera.hpp"
#include "emitters/area_light.hpp"
#include "emitters/directional_emitter.hpp"
#include "emitters/envmap.hpp"
#include "emitters/point_light.hpp"
#include "emitters/sky.hpp"
#include "emitters/spot_light.hpp"
#include "emitters/sun.hpp"
#include "global.hpp"
#include "integrators/bdpt.hpp"
#include "integrators/path.hpp"
#include "integrators/volpath.hpp"
#include "media/medium.hpp"
#include "media/phase_functions/phase_function.hpp"
#include "ndfs/ndf.hpp"
#include "shapes/cube.hpp"
#include "shapes/cylinder.hpp"
#include "shapes/disk.hpp"
#include "shapes/meshes.hpp"
#include "shapes/rectangle.hpp"
#include "shapes/sphere.hpp"
#include "shapes/triangle.hpp"
#include "textures/bitmap.hpp"
#include "textures/checkerboard.hpp"
#include "textures/constant_texture.hpp"

NAMESPACE_BEGIN(raytracer)

class Renderer
{
public:
    Renderer();
    ~Renderer();

    void AddMedium(Medium *medium);
    void AddBsdf(Bsdf *bsdf);
    void AddEmitter(Emitter *emitter);
    void AddShape(Shape *shape);
    void AddTexture(Texture *texture);

    void Render(const std::string &filename);

    void SetIntegratorInfo(IntegratorType integrator_type, int max_depth, int rr_depth, bool hide_emitters);
    void SetFilm(int width, int height, double fov_x);
    void SetCamera(const dmat4 &toworld, int spp);

private:
    bool hide_emitters_;
    IntegratorType integrator_type_;
    int max_depth_;
    int rr_depth_;
    Camera camera_;
    std::vector<Bsdf *> bsdfs_;
    std::vector<Emitter *> emitters_;
    std::vector<Medium *> media_;
    std::vector<Shape *> shapes_;
    std::vector<Texture *> textures_;
};

NAMESPACE_END(raytracer)