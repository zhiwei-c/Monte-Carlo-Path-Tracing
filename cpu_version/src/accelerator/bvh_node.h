#pragma once

#include <memory>

#include "../core/shape_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 层次包围盒叶节点
class BvhNode
{
public:
	///\brief 层次包围盒叶节点
	///\param shape 叶节点包含的物体
	BvhNode(Shape *shape)
		: left_(nullptr), right_(nullptr), shape_(shape),
		  area_(shape->area()), aabb_(shape->aabb())
	{
	}

	///\brief 层次包围盒非叶节点
	///\param left 左子节点
	///\param right 右子节点
	BvhNode(std::unique_ptr<BvhNode> left, std::unique_ptr<BvhNode> right)
		: left_(std::move(left)), right_(std::move(right)), shape_(nullptr)
	{
		area_ = left_->area() + right_->area();
		aabb_ = left_->aabb() + right_->aabb();
	}

	///\brief 求取光线与层次包围盒的交点
	void Intersect(const Ray &ray, Intersection &its) const
	{
		if (!aabb_.Intersect(ray))
			return;
		else if (shape_)
			shape_->Intersect(ray, its);
		else
		{
			left_->Intersect(ray, its);
			right_->Intersect(ray, its);
		}
	}

	///\brief 按表面积从物体表面采样一点
	Intersection Sample(Float p) const
	{
		if (shape_)
			return shape_->SampleP();
		else if (p * area_ < left_->area())
			return left_->Sample(p);
		else
			return right_->Sample(p);
	}

	///\brief 物体表面积
	const Float &area() const { return area_; }

	///\brief 节点轴对齐包围盒
	const AABB &aabb() const { return aabb_; }

private:
	Float area_;							//节点包含物体的总表面积
	AABB aabb_;								//节点轴对齐包围盒
	Shape *shape_;							//节点包含的物体。仅在叶节点非空。
	std::unique_ptr<BvhNode> left_, right_; //子节点
};

NAMESPACE_END(raytracer)