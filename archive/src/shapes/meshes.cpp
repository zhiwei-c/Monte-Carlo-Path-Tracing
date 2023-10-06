#include "meshes.hpp"

#include "../accelerators/accelerator.hpp"
#include "../accelerators/bvh.hpp"

NAMESPACE_BEGIN(raytracer)

Meshes::Meshes(const std::string &id, const std::vector<Shape *> &meshes, bool flip_normals)
    : Shape(id, ShapeType::kMeshes, flip_normals),
      accelerator_(new Bvh(meshes)),
      meshes_(meshes)
{
    aabb_ = accelerator_->aabb();
    area_ = accelerator_->area();
    pdf_area_ = 1.0 / area_;
    for (Shape *mesh : meshes_)
    {
        mesh->SetPdfArea(pdf_area_);
    }
}

Meshes::~Meshes()
{
    delete accelerator_;
    accelerator_ = nullptr;
}

void Meshes::Intersect(const Ray &ray, Sampler *sampler, Intersection *its) const
{
    accelerator_->Intersect(ray, sampler, its);
}

Intersection Meshes::SamplePoint(Sampler *sampler) const
{
    return accelerator_->SamplePoint(sampler);
}

void Meshes::SetBsdf(Bsdf *bsdf)
{
    Shape::SetBsdf(bsdf);
    for (Shape *&mesh : meshes_)
    {
        mesh->SetBsdf(bsdf);
    }
}

void Meshes::SetMedium(Medium *medium_int, Medium *medium_ext)
{
    Shape::SetMedium(medium_int, medium_ext);
    for (Shape *&mesh : meshes_)
    {
        mesh->SetMedium(medium_int, medium_ext);
    }
}

NAMESPACE_END(raytracer)