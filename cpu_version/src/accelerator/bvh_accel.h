#pragma once

#include "bvh_node.h"

NAMESPACE_BEGIN(simple_renderer)

//层次包围盒
class BvhAccel
{
public:
	///\brief 层次包围盒
	///\param shapes 构建层次包围盒的物体
	BvhAccel(std::vector<Shape *> &shapes)
	{
		root_ = BuildBvhRecursively(shapes);
	}

	///\brief 求取光线与层次包围盒包含物体的交点
	///\return 光线与层次包围盒包含的物体是否相交
	bool Intersect(const Ray &ray, Intersection &its) const
	{
		root_->Intersect(ray, its);
		return its.valid();
	}

	///\brief 按表面积从层次包围盒包含物体的表面采样一点
	Intersection Sample() const
	{
		return root_->Sample(UniformFloat());
	}

	///\return 层次包围盒的轴对齐包围盒
	const AABB &aabb() const { return root_->aabb(); }

	///\return 层次包围盒的表面积
	const Float &area() const { return root_->area(); }

private:
	std::unique_ptr<BvhNode> root_; //层次包围盒根节点

	///\brief 递归地建立层次包围盒
	std::unique_ptr<BvhNode> BuildBvhRecursively(std::vector<Shape *> shapes)
	{
		if (shapes.size() == 0)
			return nullptr;

		if (shapes.size() == 1)
			return std::make_unique<BvhNode>(shapes[0]);

		AABB bound_now;
		for (auto object : shapes)
		{
			bound_now += object->aabb();
		}

		auto length_x = bound_now.max().x - bound_now.min().x;
		auto length_y = bound_now.max().y - bound_now.min().y;
		auto length_z = bound_now.max().z - bound_now.min().z;
		if (length_x > length_y && length_x > length_z)
			std::sort(shapes.begin(), shapes.end(), [](auto obj1, auto obj2)
					  { return obj1->aabb().center().x < obj2->aabb().center().x; });
		else if (length_y > length_z)
			std::sort(shapes.begin(), shapes.end(), [](auto obj1, auto obj2)
					  { return obj1->aabb().center().y < obj2->aabb().center().y; });
		else
			std::sort(shapes.begin(), shapes.end(), [](auto obj1, auto obj2)
					  { return obj1->aabb().center().z < obj2->aabb().center().z; });
		auto beginning = shapes.begin();
		auto middling = shapes.begin() + (shapes.size() / 2);
		auto ending = shapes.end();
		auto objs_left = std::vector<Shape *>(beginning, middling);
		auto objs_right = std::vector<Shape *>(middling, ending);
		auto left = BuildBvhRecursively(objs_left);
		auto right = BuildBvhRecursively(objs_right);
		return std::make_unique<BvhNode>(std::move(left), std::move(right));
	}
};

NAMESPACE_END(simple_renderer)