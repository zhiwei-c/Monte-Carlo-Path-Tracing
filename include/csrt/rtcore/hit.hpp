#ifndef CSRT__RTCORE__HIT_HPP
#define CSRT__RTCORE__HIT_HPP

#include "../tensor.hpp"

namespace csrt
{

struct Hit
{
    bool valid;
    bool inside;
    uint32_t id_instance;
    uint32_t id_primitve;
    Vec2 texcoord;
    Vec3 position;
    Vec3 normal;
    Vec3 tangent;
    Vec3 bitangent;

    QUALIFIER_D_H Hit();
    QUALIFIER_D_H Hit(const uint32_t _id_primitve, const Vec2 &_texcoord,
                      const Vec3 &_position, const Vec3 &_normal);
    QUALIFIER_D_H Hit(const uint32_t _id_primitve, const bool _inside,
                      const Vec2 &_texcoord, const Vec3 &_position,
                      const Vec3 &_normal, const Vec3 &_tangent,
                      const Vec3 &_bitangent);
};

} // namespace csrt

#endif