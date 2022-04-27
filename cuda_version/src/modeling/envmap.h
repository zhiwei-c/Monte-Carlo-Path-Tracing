#pragma once

#include "texture.h"
#include "../utils/math/coordinate.h"

struct EnvMapInfo
{
    gmat4 *to_local;
    uint radiance_idx;

    EnvMapInfo() : to_local(nullptr), radiance_idx(kUintMax) {}

    ~EnvMapInfo()
    {
        if (to_local)
        {
            delete to_local;
            to_local = nullptr;
        }
    }
};

class EnvMap
{
public:
    __device__ EnvMap() : pdf_area_(0), to_local_(nullptr), radiance_(nullptr) {}

    __device__ ~EnvMap()
    {
        if (to_local_)
        {
            delete to_local_;
            to_local_ = nullptr;
        }
    }

    __device__ vec3 radiance(const vec3 &wo) const
    {
        if (radiance_->type() == kConstant)
            return radiance_->Color(vec2(0));

        auto look_dir = gvec3(-wo.x, -wo.y, -wo.z);

        if (to_local_)
            look_dir = TransfromDir(*to_local_, look_dir);

        Float phi = 0, theta = 0;
        CartesianToSpherical(look_dir, theta, phi);

        phi = 2 * kPi - phi;
        while (phi > 2 * kPi)
            phi -= 2 * kPi;

        auto texcoord = vec2(0);
        texcoord.x = static_cast<Float>(phi * 0.5 * kPiInv); // width
        texcoord.y = static_cast<Float>(theta * kPiInv);     // height
        return radiance_->Color(texcoord);
    }

    __device__ void InitEnvMap(Float pdf_area, gmat4 *to_local, Texture *radiance)
    {
        pdf_area_ = pdf_area;

        if (to_local)
            to_local_ = new gmat4(*to_local);

        radiance_ = radiance;
    }

private:
    Float pdf_area_;
    gmat4 *to_local_;
    Texture *radiance_;
};

__global__ void CreateEnvMap(Float pdf_area,
                             gmat4 *to_local,
                             uint radiance_idx,
                             Texture *texture_list,
                             EnvMap *envmap)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        envmap->InitEnvMap(pdf_area,
                           to_local,
                           texture_list + radiance_idx);
    }
}