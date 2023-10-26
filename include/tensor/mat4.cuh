#pragma once

#include "tensor/vec3.cuh"
#include "tensor/vec4.cuh"

NAMESPACE_BEGIN(rt)

struct Mat4
{
    Vec4 rows[4];

    QUALIFIER_DEVICE Mat4();
    QUALIFIER_DEVICE Mat4(const Mat4 &m);
    QUALIFIER_DEVICE Mat4(const Vec4 &row0, const Vec4 &row1, const Vec4 &row2, const Vec4 &row3);
    QUALIFIER_DEVICE Mat4(const float x0, const float y0, const float z0, const float w0,
                          const float x1, const float y1, const float z1, const float w1,
                          const float x2, const float y2, const float z2, const float w2,
                          const float x3, const float y3, const float z3, const float w3);

    QUALIFIER_DEVICE Vec4 &operator[](const int i);
    QUALIFIER_DEVICE Vec4 operator[](const int i) const;

    QUALIFIER_DEVICE void operator=(const Mat4 &m);

    QUALIFIER_DEVICE const Mat4 &operator+() const { return *this; }
    QUALIFIER_DEVICE Mat4 operator-() const { return {-rows[0], -rows[1], -rows[2], -rows[3]}; }

    QUALIFIER_DEVICE Mat4 &operator+=(const Mat4 &m);
    QUALIFIER_DEVICE Mat4 &operator-=(const Mat4 &m);

    QUALIFIER_DEVICE Mat4 &operator*=(const float t);
    QUALIFIER_DEVICE Mat4 &operator/=(const float t);

    QUALIFIER_DEVICE Mat4 Transpose() const;
    QUALIFIER_DEVICE Mat4 Inverse() const;
};

QUALIFIER_DEVICE Mat4 operator+(const Mat4 &m0, const Mat4 &m1);
QUALIFIER_DEVICE Mat4 operator-(const Mat4 &m0, const Mat4 &m1);
QUALIFIER_DEVICE Mat4 Mul(const Mat4 &m0, const Mat4 &m1);

QUALIFIER_DEVICE Vec4 Mul(const Mat4 &m, const Vec4 &vec);
QUALIFIER_DEVICE Vec4 Mul(const Vec4 vec, const Mat4 &m);

QUALIFIER_DEVICE Mat4 operator*(const Mat4 &m, float t);
QUALIFIER_DEVICE Mat4 operator/(const Mat4 &m, float t);

QUALIFIER_DEVICE Mat4 operator*(float t, const Mat4 &m);

QUALIFIER_DEVICE Mat4 Translate(const Vec3 &vec);
QUALIFIER_DEVICE Mat4 Scale(const Vec3 &vec);
QUALIFIER_DEVICE Mat4 Rotate(const float angle, Vec3 axis);
QUALIFIER_DEVICE Mat4 LookAtLH(const Vec3 &eye, const Vec3 &target, Vec3 up);

NAMESPACE_END(rt)