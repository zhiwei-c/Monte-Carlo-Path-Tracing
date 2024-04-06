#include "medium.hpp"

#include "phase_functions/phase_function.hpp"
#include "../math/sampler.hpp"

NAMESPACE_BEGIN(raytracer)

HomogeneousMedium::HomogeneousMedium(const std::string &id, const dvec3 &sigma_a, const dvec3 &sigma_s, PhaseFunction *phase_function)
    : Medium(MediumType::kHomogeneous, id),
      sigma_s_(sigma_s),
      sigma_t_(sigma_a + sigma_s),
      medium_sampling_weight_(0),
      phase_function_(phase_function)
{
    const dvec3 albedo = sigma_s / (sigma_a + sigma_s);
    for (int dim = 0; dim < 3; ++dim)
    {
        if (albedo[dim] > medium_sampling_weight_ && sigma_t_[dim] != 0.0)
        {
            medium_sampling_weight_ = albedo[dim];
        }
    }
    if (medium_sampling_weight_ > 0)
    {
        medium_sampling_weight_ = std::max(medium_sampling_weight_, 0.5);
    }
}

bool HomogeneousMedium::SampleDistance(double max_distance, double *distance, double *pdf, dvec3 *attenuation,
                                       Sampler *sampler) const
{
    bool scattered = false;
    double xi_1 = sampler->Next1D();
    if (xi_1 < medium_sampling_weight_)
    { //抽样光线在介质内部是否发生散射
        xi_1 /= medium_sampling_weight_;
        const int channel = std::min(static_cast<int>(sampler->Next1D() * 3), 2);
        *distance = -std::log(1.0 - xi_1) / sigma_t_[channel];
        if (*distance < max_distance)
        { //光线在介质内部发生了散射
            *pdf = 0;
            for (int dim = 0; dim < 3; ++dim)
            {
                *pdf += sigma_t_[dim] * std::exp(-sigma_t_[dim] * *distance);
            }
            *pdf *= medium_sampling_weight_ * (1.0 / 3.0);
            scattered = true;
        }
    }
    if (!scattered)
    { //光线在介质内部没有发生散射
        *distance = max_distance;
        *pdf = 0;
        for (int dim = 0; dim < 3; ++dim)
        {
            *pdf += std::exp(-sigma_t_[dim] * *distance);
        }
        *pdf = medium_sampling_weight_ * (1.0 / 3.0) * *pdf + (1.0 - medium_sampling_weight_);
    }

    bool valid = false;
    for (int dim = 0; dim < 3; ++dim)
    {
        (*attenuation)[dim] = std::exp(-sigma_t_[dim] * *distance);
        if ((*attenuation)[dim] > 0.0)
        {
            valid = true;
        }
    }
    if (scattered)
    {
        *attenuation *= sigma_s_;
    }
    if (!valid)
    {
        *attenuation = dvec3(0);
    }
    return scattered;
}

HomogeneousMedium::~HomogeneousMedium()
{
    if (phase_function_)
    {
        delete phase_function_;
        phase_function_ = nullptr;
    }
}

std::pair<dvec3, double> HomogeneousMedium::EvalDistance(bool scattered, double distance) const
{
    auto attenuation = dvec3(0);
    double pdf = 0;
    bool valid = false;
    for (int dim = 0; dim < 3; ++dim)
    {
        attenuation[dim] = std::exp(-sigma_t_[dim] * distance);
        if (attenuation[dim] > 0.0)
        {
            valid = true;
        }
    }

    if (scattered)
    {
        for (int dim = 0; dim < 3; ++dim)
        {
            pdf += sigma_t_[dim] * attenuation[dim];
        }
        pdf *= medium_sampling_weight_ * (1.0 / 3.0);
        attenuation *= sigma_s_;
    }
    else
    {
        for (int dim = 0; dim < 3; ++dim)
        {
            pdf += attenuation[dim];
        }
        pdf = medium_sampling_weight_ * (1.0 / 3.0) * pdf + (1.0 - medium_sampling_weight_);
    }

    if (!valid)
    {
        attenuation = dvec3(0);
    }

    return {attenuation, pdf};
}

void HomogeneousMedium::SamplePhaseFunction(SamplingRecord *rec, Sampler *sampler) const
{
    phase_function_->Sample(rec, sampler);
}

void HomogeneousMedium::EvalPhaseFunction(SamplingRecord *rec) const
{
    phase_function_->Eval(rec);
}

NAMESPACE_END(raytracer)