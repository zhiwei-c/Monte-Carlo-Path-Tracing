#pragma once

#include "vec3.cuh"
#include "vec4.cuh"

namespace rt
{

struct Mat4
{
    Vec4 rows[4];

    QUALIFIER_D_H Mat4();
    QUALIFIER_D_H Mat4(const Mat4 &m);
    QUALIFIER_D_H Mat4(const Vec4 &row0, const Vec4 &row1, const Vec4 &row2,
                       const Vec4 &row3);
    QUALIFIER_D_H Mat4(const float x0, const float y0, const float z0,
                       const float w0, const float x1, const float y1,
                       const float z1, const float w1, const float x2,
                       const float y2, const float z2, const float w2,
                       const float x3, const float y3, const float z3,
                       const float w3);

    QUALIFIER_D_H Vec4 &operator[](const int i);
    QUALIFIER_D_H Vec4 operator[](const int i) const;

    QUALIFIER_D_H void operator=(const Mat4 &m);

    QUALIFIER_D_H const Mat4 &operator+() const { return *this; }
    QUALIFIER_D_H Mat4 operator-() const
    {
        return {-rows[0], -rows[1], -rows[2], -rows[3]};
    }

    QUALIFIER_D_H Mat4 &operator+=(const Mat4 &m);
    QUALIFIER_D_H Mat4 &operator-=(const Mat4 &m);

    QUALIFIER_D_H Mat4 &operator*=(const float t);
    QUALIFIER_D_H Mat4 &operator/=(const float t);

    QUALIFIER_D_H Mat4 Transpose() const;
    QUALIFIER_D_H Mat4 Inverse() const;
};

QUALIFIER_D_H Mat4 operator+(const Mat4 &m0, const Mat4 &m1);
QUALIFIER_D_H Mat4 operator-(const Mat4 &m0, const Mat4 &m1);
QUALIFIER_D_H Mat4 Mul(const Mat4 &m0, const Mat4 &m1);

QUALIFIER_D_H Vec4 Mul(const Mat4 &m, const Vec4 &vec);
QUALIFIER_D_H Vec4 Mul(const Vec4 vec, const Mat4 &m);

QUALIFIER_D_H Mat4 operator*(const Mat4 &m, const float t);
QUALIFIER_D_H Mat4 operator/(const Mat4 &m, const float t);

QUALIFIER_D_H Mat4 operator*(const float t, const Mat4 &m);

QUALIFIER_D_H Mat4 Translate(const Vec3 &vec);
QUALIFIER_D_H Mat4 Scale(const Vec3 &vec);
QUALIFIER_D_H Mat4 Rotate(const float angle, Vec3 axis);
QUALIFIER_D_H Mat4 LookAtLH(const Vec3 &eye, const Vec3 &target, Vec3 up);

QUALIFIER_D_H Vec3 TransformPoint(const Mat4& m, const Vec3& p);
QUALIFIER_D_H Vec3 TransformVector(const Mat4& m, const Vec3& v);

} // namespace rt
