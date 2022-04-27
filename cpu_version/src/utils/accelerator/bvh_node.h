#pragma once

#include <memory>

#include "../../modeling/shape.h"

NAMESPACE_BEGIN(simple_renderer)

//层次包围盒节点
class BvhNode
{
public:
	BvhNode() : left_(nullptr), right_(nullptr), shape_(nullptr), area_(0) {}

	BvhNode(BvhNode *left,
			BvhNode *right,
			Shape *shape)
		: left_(left),
		  right_(right),
		  shape_(shape)
	{
		if (left_)
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

	~BvhNode()
	{
		if (left_)
		{
			delete left_;
			left_ = nullptr;
		}
		if (right_)
		{
			delete right_;
			right_ = nullptr;
		}
	}

	void Intersect(const Ray &ray, Intersection &its) const
	{
		if (!aabb_.Intersect(ray))
			return;

		if (shape_)
			shape_->Intersect(ray, its);
		else
		{
			left_->Intersect(ray, its);
			right_->Intersect(ray, its);
		}
	}

	BvhNode *&left() { return left_; }

	BvhNode *&right() { return right_; }

	Shape *&shape() { return shape_; }

	const Float &area() const { return area_; }

	const AABB &aabb() const { return aabb_; }

private:
	Float area_;			 //节点包含物体的总表面积
	AABB aabb_;				 //节点包围盒
	Shape *shape_;			 //节点包含的物体。仅在叶节点非空。
	BvhNode *left_, *right_; //子节点
};

NAMESPACE_END(simple_renderer)