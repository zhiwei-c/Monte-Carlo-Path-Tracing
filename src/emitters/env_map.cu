#include "env_map.cuh"

#include "../utils/math.cuh"

QUALIFIER_DEVICE EnvMap::EnvMap(const unsigned int id_radiance, const float scale,
                                const Mat4 &to_local)
    : id_radiance_(id_radiance), scale_(scale), to_local_(to_local)
{
}

QUALIFIER_DEVICE Vec3 EnvMap::GetRadiance(Vec3 look_dir, const float *pixel_buffer,
                                          Texture **texture_buffer) const
{
    look_dir = TransfromVector(to_local_, look_dir);
    float phi = 0, theta = 0;
    CartesianToSpherical(look_dir, &theta, &phi, nullptr);
    const Vec2 texcoord = {phi * kOneDivTwoPi, theta * kPiInv};
    return scale_ * texture_buffer[id_radiance_]->GetColor(texcoord, pixel_buffer);
}
