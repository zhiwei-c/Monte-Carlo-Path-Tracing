#pragma once

#include "../core/integrator_base.h"

NAMESPACE_BEGIN(simple_renderer)

//路径点
struct PathVertex
{
    Intersection its;    //路径点对应的交点
    Vector3 wi;          //在路径中的当前点，光线入射方向
    Vector3 wo;          //在路径中的当前点，光线出射方向
    Float cos_theta_abs; //在路径中的当前点，光线入射方向和当前点法线夹角余弦的绝对值
    Float pdf;           //在路径中的当前点，光线入射并出射的概率
    Spectrum bsdf;       //在路径中的当前点，光线入射并出射对应的 BSDF 数值
    Spectrum L;          //在路径中的当前点，光线沿出射方向传递能量的数学期望

    PathVertex(Intersection its, Vector3 wi, Vector3 wo)
        : its(its), wi(wi), wo(wo), cos_theta_abs(2), pdf(-1), bsdf(Spectrum(0)), L(Spectrum(0)) {}
};

//双向路径追踪派生类
class BdptIntegrator : public Integrator
{
public:
    ///\brief 双向路径追踪
    ///\param max_depth 递归地追踪光线最大深度
    ///\param rr_depth 最小的光线追踪深度，超过该深度后进行俄罗斯轮盘赌抽样控制光线追踪深度
    BdptIntegrator(int max_depth, int rr_depth)
        : Integrator(max_depth, rr_depth) {}

    ///\brief 着色
    ///\param eye_pos 观察点的位置
    ///\param look_dir 观察方向
    ///\return 观察点来源于给定观察方向的辐射亮度
    Spectrum Shade(const Vector3 &eye_pos, const Vector3 &look_dir) const override;

private:
    ///\brief 处理双向路径追踪算法
    Spectrum ProcessBdpt(const Intersection &its, const Vector3 &wo) const;

    ///\brief 从光源出发，创建路径点
    std::vector<PathVertex> CreateEmitterPath() const;

    ///\brief 从相机出发，创建路径点
    std::vector<PathVertex> CreateCameraPath(const Intersection &its_first, const Vector3 &wo_first) const;

    ///\brief 光源与环境光 -> 某个路径点，计算直接光照的辐射亮度（光亮度）的数学期望
    Spectrum EmitterEnv2OneV(const PathVertex &v, const Intersection *its_emitter_ptr = nullptr) const;

    ///\brief 第二个及之后的某个光源路径点 -> 某个相机路径点，计算辐射亮度（光亮度）的数学期望及概率
    std::pair<Spectrum, Float> PrepareOtherEmitter2OneC(const std::vector<PathVertex> &emitter_path, const int e_index, PathVertex c) const;
};

NAMESPACE_END(simple_renderer)