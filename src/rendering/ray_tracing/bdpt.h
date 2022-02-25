#pragma once

#include "../integrator.h"

NAMESPACE_BEGIN(simple_renderer)

//路径点
struct PathVertex
{
    Intersection its;    //路径点对应的交点
    Vector3 wi;          //在路径中光线入射方向
    Vector3 wo;          //在路径中光线出射方向
    Float cos_theta_abs; //在路径中光线入射方向和交点法线夹角余弦的绝对值
    Float pdf;           //在路径中光线入射并出射的概率
    Spectrum bsdf;       //在路径中光线入射并出射对应的 BSDF 数值
    Spectrum L;          //在路径中光线沿出射方向传递能量的数学期望

    PathVertex(Intersection its, Vector3 wi, Vector3 wo)
        : its(its), wi(wi), wo(wo), cos_theta_abs(2), pdf(-1), bsdf(Spectrum(0)), L(Spectrum(0)) {}

    ///\return 交点的位置
    Vector3 pos() const { return its.pos(); }

    ///\return 交点处的物体表面的法线
    Vector3 normal() const { return its.normal(); }

    /**
     * \brief 交点处光线从给定方向射入后，从给定方向射出，
     *      计算 BSDF（bidirectional scattering distribution function，双向散射分别函数）系数
     * \param wi 给定的光线射入方向
     * \param wo 给定的光线射出方向
     * \return 算得的 BSDF 系数
     */
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo) const { return its.Eval(wi, wo); }

    /**
     * \brief 交点处光线从给定方向射入后，从给定方向射出的概率
     * \param wi 给定的光线射入方向
     * \param wo 给定的光线射出方向
     * \return 算得的概率
     */
    Float Pdf(const Vector3 &wi, const Vector3 &wo) const { return its.Pdf(wi, wo); }

    /**
     * \brief 交点处给定光线入射方向，采样出射方向
     * \param wi 给定的光线入射方向
     * \return 一个由 Vector3 类型和 Float 类型构成的 std::pair，分别表示采样到出射方向和光线逆向传播时的概率
     */
    std::pair<Vector3, Float> SampleWo(const Vector3 &wi) const
    {
        auto bs = its.Sample(-wi);
        return {-bs.wi, bs.pdf};
    }

    /**
     * \brief 交点处给定光线出射方向，采样入射方向
     * \param wi 给定的光线出射方向
     * \return 采样结果
     */
    BsdfSampling SampleWi(const Vector3 &wo) const { return its.Sample(wo); }
};

//双向路径追踪算法类
class BdptIntegrator : public Integrator
{
public:
    /**
     * \brief 路径追踪算法类
     * \param max_depth 递归地追踪光线最大深度
     */
    BdptIntegrator(int max_depth) : Integrator(IntegratorType::kBdpt, max_depth) {}

    ///\brief 着色
    Spectrum Shade(const Vector3 &eye_pos, const Vector3 &look_dir) const override;

private:
    ///\brief 处理双向路径追踪算法
    Spectrum ProcessBdpt(const Intersection &its, const Vector3 &wo) const;

    ///\brief 从光源出发，创建路径点
    std::vector<PathVertex> CreateEmitterPath() const;

    ///\brief 从相机出发，创建路径点
    std::vector<PathVertex> CreateCameraPath(const Intersection &its_first, const Vector3 &wo_first) const;

    ///\brief 第一个光源路径点（光源）与环境光 -> 某个路径点，计算直接光照的辐射亮度（光亮度）的数学期望
    Spectrum PrepareFisrtEmitter2OneV(const PathVertex &e, const PathVertex &v) const;

    ///\brief 第二个光源路径点 -> 某个相机路径点，计算辐射亮度（光亮度）的数学期望及概率
    std::pair<Spectrum, Float> PrepareOtherEmitter2OneC(const std::vector<PathVertex> &emitter_path, const int e_index, const PathVertex &c) const;
};

NAMESPACE_END(simple_renderer)