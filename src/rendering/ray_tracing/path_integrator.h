#pragma once

#include "../integrator.h"

NAMESPACE_BEGIN(simple_renderer)

//路径追踪算法积分器类
class PathIntegrator : public Integrator
{
public:
    /**
     * \brief 路径追踪算法积分器类
     * \param scene 待着色的场景
     * \param pdf_rr 递归地追踪光线俄罗斯轮盘赌的概率
     */
    PathIntegrator(Scene *scene, Float pdf_rr = 0.95) : Integrator(scene), pdf_rr_(pdf_rr) {}

    Spectrum Shade(const Vector3 &eye_pos, const Vector3 &look_dir) const override;

private:
    Float pdf_rr_; //递归地追踪光线俄罗斯轮盘赌的概率

    /**
     * \brief 根据路径追踪算法递归地着色
     * \param obj 当前光线与物体的交点
     * \param wo 当前交点处光线的出射方向
     * \return 当前交点处出射光线的辐射亮度
     */
    Spectrum ShadeRecursively(const Intersection &obj, const Vector3 &wo) const;

    /**
     * \brief 对发光物体按表面积进行采样
     * \param pos_pre 作为采样时起点的，当前光线与物体的交点
     * \return 由 Intersection 类型和 Float 类型构成的 pair，分别代表采样到的发光物体上的点，和采样到该点的概率。
     */
    std::pair<Intersection, Float> SampleEmitter(const Vector3 &pos_pre) const;
};

NAMESPACE_END(simple_renderer)