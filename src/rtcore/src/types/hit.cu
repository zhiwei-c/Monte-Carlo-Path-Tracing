#include "types.cuh"

#include "utils.cuh"

namespace csrt
{

QUALIFIER_D_H Hit::Hit()
    : valid(false), inside(false), id_instance(kInvalidId),
      id_primitve(kInvalidId), texcoord{}, position{}, normal{}, tangent{},
      bitangent{}
{
}

QUALIFIER_D_H Hit::Hit(const bool _id_primitve, const Vec2 &_texcoord,
                       const Vec3 &_position, const Vec3 &_normal)
    : valid(true), inside(false), id_instance(kInvalidId),
      id_primitve(_id_primitve), texcoord(_texcoord), position(_position),
      normal(_normal), tangent{}, bitangent{}
{
}

QUALIFIER_D_H Hit::Hit(const bool _id_primitve, const bool _inside,
                       const Vec2 &_texcoord, const Vec3 &_position,
                       const Vec3 &_normal, const Vec3 &_tangent,
                       const Vec3 &_bitangent)
    : valid(true), inside(_inside), id_instance(kInvalidId),
      id_primitve(_id_primitve), texcoord(_texcoord), position(_position),
      normal(_normal), tangent(_tangent), bitangent(_bitangent)
{
}

} // namespace csrt