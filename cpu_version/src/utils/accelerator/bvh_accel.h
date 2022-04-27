#pragma once

#include <algorithm>
#include <optional>
#include <stack>

#include "bvh_node.h"

NAMESPACE_BEGIN(simple_renderer)

//层次包围盒
class BvhAccel
{
public:
	///\brief 层次包围盒
	BvhAccel(std::vector<Shape *> shapes)
	{
		root_ = BuildBvhRecursively(shapes);
	}

	~BvhAccel()
	{
		if (root_)
		{
			delete root_;
			root_ = nullptr;
		}
	}

	bool Intersect(const Ray &ray, Intersection &its) const
	{
#ifdef NO_RECURSION
		std::stack<BvhNode *> st;
		st.push(root_);
		BvhNode *now = nullptr;
		while (!st.empty())
		{
			now = st.top();
			st.pop();
			while (now->aabb().Intersect(ray))
			{
				if (now->shape())
				{
					now->shape()->Intersect(ray, its);
					break;
				}
				else
				{
					st.push(now->right());
					now = now->left();
				}
			}
		}
#else
		root_->Intersect(ray, its);
#endif
		return its.valid();
	}

	Intersection Sample() const
	{
		auto now = root_;
		auto p = UniformFloat();
		while (!now->shape())
		{
			if (p * now->area() < now->left()->area())
				now = now->left();
			else
				now = now->right();
		}
		return now->shape()->SampleP();
	}

	const AABB &aabb() const { return root_->aabb(); }
	const Float &area() const { return root_->area(); }

private:
	BvhNode *root_;

	BvhNode *BuildBvhRecursively(std::vector<Shape *> shapes)
	{
		if (shapes.size() == 0)
		{
			return nullptr;
		}

		if (shapes.size() == 1)
		{
			return new BvhNode(nullptr, nullptr, shapes[0]);
		}

		AABB bound_now;
		for (auto object : shapes)
		{
			bound_now += object->aabb();
		}

		auto length_x = bound_now.max().x - bound_now.min().x;
		auto length_y = bound_now.max().y - bound_now.min().y;
		auto length_z = bound_now.max().z - bound_now.min().z;

		if (length_x > length_y && length_x > length_z)
		{
			std::sort(shapes.begin(), shapes.end(), [](auto obj1, auto obj2)
					  { return obj1->aabb().center().x < obj2->aabb().center().x; });
		}
		else if (length_y > length_z)
		{
			std::sort(shapes.begin(), shapes.end(), [](auto obj1, auto obj2)
					  { return obj1->aabb().center().y < obj2->aabb().center().y; });
		}
		else
		{
			std::sort(shapes.begin(), shapes.end(), [](auto obj1, auto obj2)
					  { return obj1->aabb().center().z < obj2->aabb().center().z; });
		}
		auto beginning = shapes.begin();
		auto middling = shapes.begin() + (shapes.size() / 2);
		auto ending = shapes.end();
		auto objs_left = std::vector<Shape *>(beginning, middling);
		auto objs_right = std::vector<Shape *>(middling, ending);
		auto left = BuildBvhRecursively(objs_left);
		auto right = BuildBvhRecursively(objs_right);
		return new BvhNode(left, right, nullptr);
	}
};

NAMESPACE_END(simple_renderer)