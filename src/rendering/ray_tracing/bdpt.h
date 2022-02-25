#pragma once

#include "../integrator.h"

NAMESPACE_BEGIN(simple_renderer)

struct PathVertex
{
    Intersection its;
    Vector3 wi;
    Vector3 wo;
    Float cos_theta_abs;
    Float pdf;
    Spectrum bsdf;
    Spectrum L;

    PathVertex(Intersection its, Vector3 wi, Vector3 wo)
        : its(its), wi(wi), wo(wo), cos_theta_abs(2), pdf(-1), bsdf(Spectrum(0)), L(Spectrum(0)) {}

    Vector3 pos() const { return its.pos(); }
    Vector3 normal() const { return its.normal(); }
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo) const { return its.Eval(wi, wo); }
    Float Pdf(const Vector3 &wi, const Vector3 &wo) const { return its.Pdf(wi, wo); }

    std::pair<Vector3, Float> SampleWo(const Vector3 &wi) const
    {
        auto bs = its.Sample(-wi);
        return {-bs.wi, bs.pdf};
    }

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

    Spectrum Shade(const Vector3 &eye_pos, const Vector3 &look_dir) const override;

private:
    Spectrum ProcessBdpt(const Intersection &start_obj, const Vector3 &start_wo) const;

    std::vector<PathVertex> CreateEmitterPath() const;

    void PrepareEmitterPath(std::vector<PathVertex> &emitter_path) const;

    std::vector<PathVertex> CreateCameraPath(const Intersection &start_obj, const Vector3 &start_wo) const;

    /**
     * \brief 第一个光源路径点（光源）或环境光 -> 某个路径点
     */
    Spectrum PrepareFisrtEmitter2OneV(const PathVertex &e, const PathVertex &v) const;

    /**
     * \brief  第二个光源路径点 -> 某个相机路径点
     */
    std::pair<Spectrum, Float> PrepareOtherEmitter2OneC(const std::vector<PathVertex> &emitter_path, const int e_index, const PathVertex &c) const;
};

NAMESPACE_END(simple_renderer)