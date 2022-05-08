#pragma once

#include "../core/ray.h"

#define kEpsilonMachine (1.192092896e-07f * 0.5f)

__host__ __device__ inline Float GammaError(int n)
{
    return (n * kEpsilonMachine) / (1 - n * kEpsilonMachine);
}

class AABB
{
public:
    ///\brief 轴对齐包围盒
    __host__ __device__ AABB() : max_(vec3(-INFINITY)), min_(vec3(INFINITY)) {}

    /**
     * \brief 轴对齐包围盒
     * \param min 底边界
     * \param max 顶边界
     */
    __host__ __device__ AABB(const vec3 &min, const vec3 &max) : min_(min), max_(max) {}

    /**
     * \brief 判断光线与包围盒是否相交
     * \param ray 待判断的光线
     * \return 判断是否相交的结果
     */
    __device__ bool Intersect(const Ray &ray) const
    {
		auto t_min = static_cast<Float>(0),
			 t_max = static_cast<Float>(0),
			 t_enter = static_cast<Float>(-INFINITY),
			 t_exit = static_cast<Float>(INFINITY);
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
                {
                    auto temp = t_min;
                    t_min = t_max;
                    t_max = temp;
                }

                t_enter = glm::max(t_min, t_enter);
                t_exit = glm::min(t_max, t_exit);
            }
        }
        t_exit *= 1.0 + 2.0 * GammaError(3);
        return t_exit > 0 && t_enter < t_exit;
    }

    ///\return 底边界
    __host__ __device__ vec3 min() const { return min_; }

    ///\return 顶边界
    __host__ __device__ vec3 max() const { return max_; }

    ///\return 中心
    __host__ __device__ vec3 center() const
    {
        return (this->min_ + this->max_) * static_cast<Float>(0.5);
    }

    __host__ __device__ AABB operator+(const AABB &b) const
    {
        auto min = myvec::min(this->min(), b.min());
        auto max = myvec::max(this->max(), b.max());
        return AABB(min, max);
    }

    __host__ __device__ AABB &operator+=(const AABB &rhs)
    {
        this->min_ = myvec::min(this->min_, rhs.min());
        this->max_ = myvec::max(this->max_, rhs.max());
        return *this;
    }

    __host__ __device__ AABB &operator+=(const vec3 &rhs)
    {
        this->min_ = myvec::min(this->min_, rhs);
        this->max_ = myvec::max(this->max_, rhs);
        return *this;
    }

private:
    vec3 min_; //包围盒底边界
    vec3 max_; //包围盒顶边界
};
