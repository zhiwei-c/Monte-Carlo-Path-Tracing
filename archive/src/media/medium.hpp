#pragma once

#include <utility>

#include "../core/sampling_record.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

enum class MediumType
{
    kHomogeneous, //均匀介质
};

//介质
class Medium
{
public:
    virtual ~Medium() {}

    virtual bool SampleDistance(double max_distance, double *distance, double *pdf, dvec3 *attenuation,
                                Sampler *sampler) const = 0;
    virtual std::pair<dvec3, double> EvalDistance(bool scattered, double distance) const = 0;

    virtual void SamplePhaseFunction(SamplingRecord *rec, Sampler *sampler) const = 0;
    virtual void EvalPhaseFunction(SamplingRecord *rec) const = 0;

protected:
    Medium(MediumType type, const std::string &id) : type_(type), id_(id) {}

private:
    MediumType type_; //介质类型
    std::string id_;  //介质ID
};

//均匀介质
class HomogeneousMedium : public Medium
{
public:
    HomogeneousMedium(const std::string &id, const dvec3 &sigma_a, const dvec3 &sigma_s, PhaseFunction *phase_function);
    ~HomogeneousMedium();

    bool SampleDistance(double max_distance, double *distance, double *pdf, dvec3 *attenuation,
                        Sampler *sampler) const override;
    std::pair<dvec3, double> EvalDistance(bool scattered, double distance) const override;

    void SamplePhaseFunction(SamplingRecord *rec, Sampler *sampler) const override;
    void EvalPhaseFunction(SamplingRecord *rec) const override;

private:
    PhaseFunction *phase_function_; //相函数
    double medium_sampling_weight_; //抽样光线在介质内部发生散射的权重
    dvec3 sigma_s_;                 //散射系数
    dvec3 sigma_t_;                 //衰减系数
};

NAMESPACE_END(raytracer)