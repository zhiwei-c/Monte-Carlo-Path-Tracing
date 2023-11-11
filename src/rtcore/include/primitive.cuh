#pragma once

#include "types.cuh"

namespace rt
{

class Primitive
{
public:
    enum class Type
    {
        kNone,
        kTriangle,
        kSphere,
    };

    struct Data
    {
        struct Triangle
        {
            Vec3 v0v1 = {};
            Vec3 v0v2 = {};
            Vec2 texcoords[3] = {};
            Vec3 positions[3] = {};
            Vec3 normals[3] = {};
            Vec3 tangents[3] = {};
            Vec3 bitangents[3] = {};
        };

        struct Sphere
        {
            float radius = 0;
            Vec3 center = {};
            Mat4 to_world = {};
            Mat4 normal_to_world = {};
            Mat4 to_local = {};
        };

        Primitive::Type type;

        union
        {
            Triangle triangle;
            Sphere sphere;
        };

        QUALIFIER_D_H Data();
        QUALIFIER_D_H Data(const Primitive::Data &info);
        QUALIFIER_D_H void operator=(const Primitive::Data &info);
    };

    QUALIFIER_D_H Primitive();
    QUALIFIER_D_H Primitive(const uint32_t id, const Primitive::Data &info);

    QUALIFIER_D_H AABB aabb() const;

    QUALIFIER_D_H void Intersect(Ray *ray, Hit *hit) const;
    QUALIFIER_D_H Hit Sample(const float xi_0, const float xi_1) const;

private:
    QUALIFIER_D_H void IntersectTriangle(Ray *ray, Hit *hit) const;
    QUALIFIER_D_H void IntersectSphere(Ray *ray, Hit *hit) const;

    QUALIFIER_D_H Hit SampleTriangle(const float xi_0, const float xi_1) const;
    QUALIFIER_D_H Hit SampleSphere(const float xi_0, const float xi_1) const;

    QUALIFIER_D_H AABB GetAabbTriangle() const;
    QUALIFIER_D_H AABB GetAabbSphere() const;

    uint32_t id_;
    Primitive::Data data_;
};

} // namespace rt