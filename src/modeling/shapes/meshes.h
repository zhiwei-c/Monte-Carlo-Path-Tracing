#pragma once

#include <memory>

#include "../shape.h"
#include "../../utils/accelerator/bvh_accel.h"

NAMESPACE_BEGIN(simple_renderer)
class Meshes : public Shape
{
public:
	Meshes(std::vector<Shape *> meshes, Material *material, bool flip_normals)
		: Shape(ShapeType::kMeshes, material, flip_normals), meshes_(meshes), bvh_(std::make_unique<BvhAccel>(meshes))
	{
		area_ = bvh_->area();
		aabb_ = bvh_->aabb();
		for(auto& mesh: meshes){
			mesh->SetParent(this);
		}
	}

	~Meshes()
	{
		for (auto &mesh : meshes_)
		{
			if (mesh)
			{
				delete ((Triangle *)mesh);
				mesh = nullptr;
			}
		}
		meshes_.clear();
	}

	Intersection Intersect(const Ray &ray) const override
	{
		return this->bvh_->Intersect(ray);
	}

	std::pair<Intersection, Float> SampleP() const override
	{
		return this->bvh_->Sample();
	}

private:
	std::unique_ptr<BvhAccel> bvh_; //层次包围盒
	std::vector<Shape *> meshes_;	//包含的三角面片
};

NAMESPACE_END(simple_renderer)