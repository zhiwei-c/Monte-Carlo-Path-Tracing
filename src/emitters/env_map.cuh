#pragma once

#include "emitter.cuh"

class EnvMap
{
public:
    QUALIFIER_DEVICE EnvMap(const unsigned int id_radiance, const float scale, const Mat4 &to_local);

    QUALIFIER_DEVICE Vec3 GetRadiance(Vec3 look_dir, const float *pixel_buffer,
                                      Texture **texture_buffer) const;

private:
    unsigned int id_radiance_;
    float scale_;
    Mat4 to_local_;
};