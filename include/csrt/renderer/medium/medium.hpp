#ifndef CSRT__RENDERER__MEDIUM__MEDIUM_HPP
#define CSRT__RENDERER__MEDIUM__MEDIUM_HPP

#include "../../tensor.hpp"

#include "henyey_greenstein.hpp"
#include "isotropic.hpp"

#include "homogeneous.hpp"

namespace csrt
{

enum class PhaseFunctionType
{
    kIsotropic,        //各向同性的相函数
    kHenyeyGreenstein, //亨尼-格林斯坦相函数
};

struct PhaseFunctionData
{
    PhaseFunctionType type = PhaseFunctionType::kIsotropic;
    Vec3 g = {}; // Henyey Greenstein 散射光线方向余弦的平均
};

struct PhaseSampleRec
{
    bool valid = false;
    float pdf = 0;
    Vec3 wi = {};
    Vec3 wo = {};
    Vec3 attenuation = {};
};

enum class MediumType
{
    kHomogeneous, //均匀介质
};

struct MediumInfo
{
    MediumType type = MediumType::kHomogeneous;
    HomogeneousMediumInfo homogeneous = {};
    PhaseFunctionData phase_func = {};
};

struct MediumData
{
    MediumType type = MediumType::kHomogeneous;
    HomogeneousMediumData homogeneous = {};
    PhaseFunctionData phase_func = {};
};

struct MediumSampleRec
{
    bool valid = false;
    bool scattered = false;
    float pdf = 1.0f;
    float distance = 0;
    Vec3 attenuation = {1.0f};
};

class Medium
{
public:
    QUALIFIER_D_H Medium() : id_(kInvalidId), data_() {}
    QUALIFIER_D_H Medium(const uint32_t id, const MediumInfo &info);

    QUALIFIER_D_H void Sample(const float max_distance, uint32_t *seed,
                              MediumSampleRec *rec) const;
    QUALIFIER_D_H void Evaluate(MediumSampleRec *rec) const;

    QUALIFIER_D_H void SamplePhase(uint32_t *seed, PhaseSampleRec *rec) const;
    QUALIFIER_D_H void EvaluatePhase(PhaseSampleRec *rec) const;

private:
    uint32_t id_;
    MediumData data_;
};

} // namespace csrt

#endif