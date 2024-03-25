#pragma once

#include "../../rtcore/scene.cuh"
#include "../../tensor.cuh"
#include "../../utils.cuh"
#include "../textures/texture.cuh"

#include "constant_light.cuh"
#include "directional_light.cuh"
#include "envmap.cuh"
#include "point_light.cuh"
#include "spot_light.cuh"
#include "sun.cuh"

namespace csrt
{

enum class EmitterType
{
    kNone,
    kPoint,
    kSpot,
    kDirectional,
    kSun,
    kEnvMap,
    kConstant,
};

struct EmitterInfo
{

    EmitterType type;
    union
    {
        PointLightData point;
        SpotLightInfo spot;
        DirectionalLightData directional;
        SunInfo sun;
        EnvMapInfo envmap;
        ConstantLightData constant;
    };

    QUALIFIER_D_H EmitterInfo();
    QUALIFIER_D_H EmitterInfo(const EmitterInfo &info);
    QUALIFIER_D_H ~EmitterInfo() {}
};

struct EmitterSampleRec
{
    bool valid = false;
    bool harsh = true;
    float distance = kMaxFloat;
    Vec3 wi = {};
};

struct EmitterData
{
    EmitterType type;
    union
    {
        PointLightData point;
        SpotLightData spot;
        DirectionalLightData directional;
        SunData sun;
        EnvMapData envmap;
        ConstantLightData constant;
    };

    QUALIFIER_D_H EmitterData();
    QUALIFIER_D_H ~EmitterData() {}
    QUALIFIER_D_H void operator=(const EmitterData &data);
};

class Emitter
{
public:
    QUALIFIER_D_H Emitter();
    QUALIFIER_D_H Emitter(const uint32_t id, const EmitterInfo &info,
                          Texture *texture_buffer);

    QUALIFIER_D_H void InitEnvMap(const int width, const int height,
                                  const float normalization, float *pixels);

    QUALIFIER_D_H EmitterSampleRec Sample(const Vec3 &origin, const float xi_0,
                                          const float xi_1) const;
    QUALIFIER_D_H Vec3 Evaluate(const EmitterSampleRec &rec) const;
    QUALIFIER_D_H float Pdf(const Vec3 &look_dir) const;
    QUALIFIER_D_H Vec3 Evaluate(const Vec3 &look_dir) const;

private:
    uint32_t id_;
    EmitterData data_;
};

} // namespace csrt