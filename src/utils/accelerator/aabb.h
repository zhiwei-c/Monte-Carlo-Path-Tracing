#pragma once

#include "../math/maths.h"
#include "../../rendering/ray.h"

NAMESPACE_BEGIN(simple_renderer)

//轴对齐包围盒类
class AABB
{
public:
	///\brief 轴对齐包围盒
	AABB()
	{
		max_ = Vector3(std::numeric_limits<Float>::lowest());
		min_ = Vector3(std::numeric_limits<Float>::max());
	}

	/**
	 * \brief 轴对齐包围盒
	 * \param min 底边界
	 * \param max 顶边界
	 */
	AABB(const Vector3 &min, const Vector3 &max) : min_(min), max_(max) {}

	/**
	 * \brief 判断光线与包围盒是否相交
	 * \param ray 待判断的光线
	 * \return 判断是否相交的结果
	 */
	bool Intersect(const Ray &ray) const
	{
		Float t_min = 0, t_max = 0;
		Float t_enter = -INFINITY, t_exit = INFINITY;
		for (int i = 0; i < 3; i += 1)
		{
			//检查光线是否与包围盒某一对平面平行
			if (ray.dir()[i] == 0)
			{
				if (ray.origin()[i] > this->max_[i] ||
					ray.origin()[i] < this->min_[i])
					return false;
			}
			else
			{
				t_min = (this->min_[i] - ray.origin()[i]) * ray.dir_inv()[i];
				t_max = (this->max_[i] - ray.origin()[i]) * ray.dir_inv()[i];
				if (ray.dir()[i] < 0)
					std::swap(t_min, t_max);

				t_enter = std::max(t_min, t_enter);
				t_exit = std::min(t_max, t_exit);
			}
		}
		t_exit *= 1 + 2 * GammaError(3);
		return t_exit > 0 && t_enter < t_exit;
	}

	///\return 底边界
	Vector3 min() const { return min_; }

	///\return 顶边界
	Vector3 max() const { return max_; }

	///\return 中心
	Vector3 center() const
	{
		return (this->min_ + this->max_) * static_cast<Float>(0.5);
	}

	AABB operator+(const AABB &b)
	{
		auto min = glm::min(this->min(), b.min());
		auto max = glm::max(this->max(), b.max());
		return AABB(min, max);
	}

	AABB &operator+=(const AABB &rhs)
	{
		this->min_ = glm::min(this->min_, rhs.min());
		this->max_ = glm::max(this->max_, rhs.max());
		return *this;
	}

	AABB &operator+=(const Vector3 &rhs)
	{
		this->min_ = glm::min(this->min_, rhs);
		this->max_ = glm::max(this->max_, rhs);
		return *this;
	}

private:
	Vector3 min_; //包围盒底边界
	Vector3 max_; //包围盒顶边界
};

NAMESPACE_END(simple_renderer)