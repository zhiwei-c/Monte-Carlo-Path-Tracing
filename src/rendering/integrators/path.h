#pragma once

#include "../integrator.h"

NAMESPACE_BEGIN(simple_renderer)

//路径追踪算法类
class PathIntegrator : public Integrator
{
public:
    /**
     * \brief 路径追踪算法类
     * \param max_depth 溯源光线的最大跟踪深度
     * \param rr_depth 最小的光线追踪深度，超过该深度后进行俄罗斯轮盘赌抽样控制光线追踪深度
     */
    PathIntegrator(int max_depth, int rr_depth)
        : Integrator(max_depth, rr_depth) {}

    /**
     * \brief 根据路径追踪算法迭代地着色
     * \param eye_pos 观察点的坐标
     * \param look_dir 观察方向
     * \return 观察点来源于给定观察方向的辐射亮度
     */
    Spectrum Shade(const Vector3 &eye_pos, const Vector3 &look_dir) const override;
};

NAMESPACE_END(simple_renderer)