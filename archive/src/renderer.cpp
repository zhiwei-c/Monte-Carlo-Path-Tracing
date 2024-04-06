#include "renderer.hpp"

#include "math/coordinate.hpp"
#include "utils/image.hpp"

NAMESPACE_BEGIN(raytracer)

Renderer::Renderer()
    : integrator_type_(IntegratorType::kPath),
      max_depth_(-1),
      rr_depth_(5),
      camera_(Camera())
{
}

Renderer::~Renderer()
{
    for (Medium *&medium : media_)
    {
        delete medium;
        medium = nullptr;
    }
    for (Texture *&texture : textures_)
    {
        delete texture;
        texture = nullptr;
    }
    for (Bsdf *&bsdf : bsdfs_)
    {
        delete bsdf;
        bsdf = nullptr;
    }
    for (Shape *&shape : shapes_)
    {
        delete shape;
        shape = nullptr;
    }
    for (Emitter *&emitter : emitters_)
    {
        delete emitter;
        emitter = nullptr;
    }
}

void Renderer::AddMedium(Medium *medium)
{
    media_.push_back(medium);
}

void Renderer::AddTexture(Texture *texture)
{
    textures_.push_back(texture);
}

void Renderer::AddBsdf(Bsdf *bsdf)
{
    bsdfs_.push_back(bsdf);
}

void Renderer::AddShape(Shape *shape)
{
    shapes_.push_back(shape);
}

void Renderer::AddEmitter(Emitter *emitter)
{
    emitters_.push_back(emitter);
}

void Renderer::Render(const std::string &filename)
{
    Accelerator *accelerator = shapes_.empty() ? nullptr : new Bvh(shapes_);

    Integrator *integrator = nullptr;
    switch (integrator_type_)
    {
    case IntegratorType::kBdpt:
        integrator = new BdptIntegrator(max_depth_, rr_depth_, hide_emitters_, accelerator, emitters_, shapes_.size());
        break;
    case IntegratorType::kVolPath:
        integrator = new VolPathIntegrator(max_depth_, rr_depth_, hide_emitters_, accelerator, emitters_, shapes_.size());
        break;
    case IntegratorType::kPath:
    default:
        integrator = new PathIntegrator(max_depth_, rr_depth_, hide_emitters_, accelerator, emitters_, shapes_.size());
        break;
    }

    std::vector<float> frame_buffer = integrator->Shade(camera_);
    SaveImage(frame_buffer, camera_.width, camera_.height, filename);
    if (!shapes_.empty())
    {
        delete accelerator;
        accelerator = nullptr;
    }
}

void Renderer::SetIntegratorInfo(IntegratorType integrator_type, int max_depth, int rr_depth, bool hide_emitters)
{
    integrator_type_ = integrator_type;
    max_depth_ = max_depth;
    rr_depth_ = rr_depth;
    hide_emitters_ = hide_emitters;
}

void Renderer::SetFilm(int width, int height, double fov_x)
{
    camera_.width = width;
    camera_.height = height;
    camera_.fov_x = fov_x;
    camera_.fov_y = fov_x * height / width;
}

void Renderer::SetCamera(const dmat4 &toworld, int spp)
{
    const dvec3 look_at = TransfromPoint(toworld, {0, 0, 1}),
                up = TransfromVec(toworld, {0, 1, 0});
    camera_.eye = TransfromPoint(toworld, {0, 0, 0});
    camera_.front = glm::normalize(look_at - camera_.eye);
    camera_.right = glm::normalize(glm::cross(camera_.front, up));
    camera_.up = glm::normalize(glm::cross(camera_.right, camera_.front));
    camera_.spp = spp;
}

NAMESPACE_END(raytracer)