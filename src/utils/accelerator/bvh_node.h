#pragma once

#include <memory>

#include "../../modeling/shape.h"

NAMESPACE_BEGIN(simple_renderer)

//层次包围盒节点
class BvhNode
{
public:
	BvhNode() : left_(nullptr), right_(nullptr), shape_(nullptr), area_(0) {}

	BvhNode(std::unique_ptr<BvhNode> left,
			std::unique_ptr<BvhNode> right,
			const Shape *const shape)
		: left_(std::move(left)),
		  right_(std::move(right)),
		  shape_(shape)
	{
		if (left_ != nullptr)
		{
			area_ = right_ ? left_->area() + right_->area() : left_->area();
			aabb_ = right_ ? left_->aabb() + right_->aabb() : left_->aabb();
		}
		else
		{
			area_ = shape ? shape->area() : 0;
			aabb_ = shape ? shape->aabb() : AABB();
		}
	}

	Intersection Intersect(const Ray &ray) const
	{
		if (!aabb_.Intersect(ray))
		{
			return Intersection();
		}

		if (left_ == nullptr)
		{
			if (shape_ != nullptr)
				return shape_->Intersect(ray);
			else
				return Intersection();
		}
		auto ret_left = left_->Intersect(ray);
		auto ret_right = right_->Intersect(ray);
		return ret_left.distance() < ret_right.distance() ? ret_left : ret_right;
	}

	std::pair<Intersection, Float> SampleP(Float p) const
	{
		if (shape_ != nullptr)
		{
			auto [point, pdf] = shape_->SampleP();
			return {point, pdf * area_};
		}

		if (p * area_ < left_->area())
			return left_->SampleP(p);
		else
			return right_->SampleP(p);
	}

	Float area() const { return area_; }

	AABB aabb() const { return aabb_; }

private:
	Float area_;							//节点包含物体的总表面积
	AABB aabb_;								//节点包围盒
	const Shape *shape_;					//节点包含的物体。仅在叶节点非空。
	std::unique_ptr<BvhNode> left_, right_; //子节点
};

NAMESPACE_END(simple_renderer)