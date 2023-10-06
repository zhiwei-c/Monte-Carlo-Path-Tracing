#pragma once

#include "../../core/sampling_record.hpp"
#include "../../global.hpp"

NAMESPACE_BEGIN(raytracer)

enum class PhaseFunctionType
{
    kIsotropic,        //各向同性的相函数
    kHenyeyGreenstein, //亨尼-格林斯坦相函数
};

//相函数
class PhaseFunction
{
public:
    virtual ~PhaseFunction() {}

    virtual void Sample(SamplingRecord *rec, Sampler *sampler) const = 0;
    virtual void Eval(SamplingRecord *rec) const = 0;

protected:
    PhaseFunction(PhaseFunctionType type) : type_(type) {}

private:
    PhaseFunctionType type_; //相函数类型
};

//各向同性的相函数
class IsotropicPhaseFunction : public PhaseFunction
{
public:
    IsotropicPhaseFunction() : PhaseFunction(PhaseFunctionType::kIsotropic) {}

    void Sample(SamplingRecord *rec, Sampler *sampler) const override;
    void Eval(SamplingRecord *rec) const override;
};

//亨尼-格林斯坦相函数
class HenyeyGreensteinPhaseFunction : public PhaseFunction
{
public:
    HenyeyGreensteinPhaseFunction(const dvec3 &g) : PhaseFunction(PhaseFunctionType::kHenyeyGreenstein), g_(g) {}

    void Sample(SamplingRecord *rec, Sampler *sampler) const override;
    void Eval(SamplingRecord *rec) const override;

private:
    dvec3 g_; //代表散射光线平均余弦的参数
};
NAMESPACE_END(raytracer)