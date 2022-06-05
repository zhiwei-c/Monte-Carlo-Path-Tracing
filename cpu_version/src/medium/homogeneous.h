#pragma once

#include "../core/medium_base.h"

NAMESPACE_BEGIN(raytracer)

class HomogeneousMedium : public Medium
{
public:
    HomogeneousMedium(const Spectrum &sigma_a, const Spectrum &sigma_s, std::unique_ptr<PhaseFunction> phase_function)
        : Medium(std::move(phase_function)), sigma_s_(sigma_s), sigma_t_(sigma_a + sigma_s),
          medium_sampling_weight_(0)
    {
        Spectrum albedo = sigma_s / (sigma_a + sigma_s);
        for (int i = 0; i < 3; i++)
        {
            if (albedo[i] > medium_sampling_weight_ && sigma_t_[i] != 0)
                medium_sampling_weight_ = albedo[i];
        }
        if (medium_sampling_weight_ > 0)
            medium_sampling_weight_ = std::max(medium_sampling_weight_, static_cast<Float>(0.5));
    }

    bool SampleDistance(Float max_distance, Float &distance, Float &pdf, Spectrum &attenuation) const override
    {
        bool scattered = false;
        Float x1 = UniformFloat();
        if (x1 < medium_sampling_weight_)
        { //抽样光线在介质内部是否发生散射
            x1 /= medium_sampling_weight_;
            int channel = std::min(static_cast<int>(UniformFloat() * 3), 2);
            distance = -std::log(1 - x1) / sigma_t_[channel];
            if (distance < max_distance)
            { //光线在介质内部发生了散射
                pdf = 0;
                for (int i = 0; i < 3; i++)
                    pdf += sigma_t_[i] * std::exp(-sigma_t_[i] * distance);
                pdf /= 3.0;
                pdf *= medium_sampling_weight_;
                scattered = true;
            }
        }
        if (!scattered)
        { //光线在介质内部没有发生散射
            distance = max_distance;
            pdf = 0;
            for (int i = 0; i < 3; i++)
                pdf += std::exp(-sigma_t_[i] * distance);
            pdf /= 3.0;
            pdf = medium_sampling_weight_ * pdf + (1.0 - medium_sampling_weight_);
        }

        bool valid = false;
        for (int i = 0; i < 3; i++)
        {
            attenuation[i] = std::exp(-sigma_t_[i] * distance);
            if (attenuation[i] > kEpsilon)
                valid = true;
        }
        if (scattered)
            attenuation *= sigma_s_;
        if (!valid)
            attenuation = Spectrum(0);

        return scattered;
    }

    std::pair<Spectrum, Float> EvalDistance(bool scattered, Float distance) const override
    {
        auto attenuation = Spectrum(0);
        Float pdf = 0;
        bool valid = false;
        for (int i = 0; i < 3; i++)
        {
            attenuation[i] = std::exp(-sigma_t_[i] * distance);
            if (attenuation[i] > kEpsilon)
                valid = true;
        }

        if (scattered)
        {
            for (int i = 0; i < 3; i++)
                pdf += sigma_t_[i] * attenuation[i];
            pdf /= 3.0;
            pdf *= medium_sampling_weight_;
            attenuation *= sigma_s_;
        }
        else
        {
            for (int i = 0; i < 3; i++)
                pdf += attenuation[i];
            pdf /= 3.0;
            pdf = medium_sampling_weight_ * pdf + (1.0 - medium_sampling_weight_);
        }

        if (!valid)
            attenuation = Spectrum(0);

        return {attenuation, pdf};
    }

    void SamplePhaseFunction(SamplingRecord &rec) const override
    {
        phase_function_->Sample(rec);
    }

    void EvalPhaseFunction(SamplingRecord &rec) const override
    {
        phase_function_->Eval(rec);
    }

private:
    Spectrum sigma_s_;
    Spectrum sigma_t_;
    Float medium_sampling_weight_;
};

NAMESPACE_END(raytracer)