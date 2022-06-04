#pragma once

#include <memory>

#include "../core/shape_base.h"
#include "../accelerator/bvh_accel.h"

NAMESPACE_BEGIN(raytracer)
class Meshes : public Shape
{
public:
	Meshes(std::vector<Shape *> meshes, Bsdf *bsdf, bool flip_normals)
		: Shape(ShapeType::kMeshes, bsdf, flip_normals), meshes_(meshes), bvh_(std::make_unique<BvhAccel>(meshes))
	{
		aabb_ = bvh_->aabb();
		area_ = bvh_->area();
		pdf_area_ = 1 / area_;
		for (auto &mesh : meshes)
		{
			mesh->SetPdfArea(this->pdf_area_);
		}
	}

	~Meshes()
	{
		for (auto &mesh : meshes_)
		{
			if (mesh)
			{
				delete mesh;
				mesh = nullptr;
			}
		}
	}
	
	void Intersect(const Ray &ray, Intersection& its) const override
	{
		this->bvh_->Intersect(ray, its);
	}

	Intersection SampleP() const override
	{
		return this->bvh_->Sample();
	}

private:
	std::unique_ptr<BvhAccel> bvh_; //层次包围盒
	std::vector<Shape *> meshes_;	//包含的三角面片
};

NAMESPACE_END(raytracer)