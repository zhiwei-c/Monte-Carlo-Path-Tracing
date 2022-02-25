#pragma once

#include "../integrator.h"

NAMESPACE_BEGIN(simple_renderer)

//路径追踪算法类
class PathIntegrator : public Integrator
{
public:
    /**
     * \brief 路径追踪算法类
     * \param max_depth 递归地追踪光线最大深度
     */
    PathIntegrator(int max_depth) : Integrator(IntegratorType::kPath, max_depth) {}

    Spectrum Shade(const Vector3 &eye_pos, const Vector3 &look_dir) const override;

private:
    /**
     * \brief 根据路径追踪算法递归地着色
     * \param obj 当前光线与物体的交点
     * \param wo 当前交点处光线的出射方向
     * \return 当前交点处出射光线的辐射亮度
     */
    Spectrum ShadeRecursively(const Intersection &obj, const Vector3 &wo, int depth) const;
};

NAMESPACE_END(simple_renderer)