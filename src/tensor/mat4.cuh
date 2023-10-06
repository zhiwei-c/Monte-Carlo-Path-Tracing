#pragma once

#include <cassert>

#include "vec4.cuh"

struct Mat4
{

    Vec4 value[4];

    QUALIFIER_DEVICE Mat4()
        : value{{1.0f, 0.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 0.0f, 1.0f}}
    {
    }

    QUALIFIER_DEVICE Mat4(const Mat4 &m)
        : value{m.value[0], m.value[1], m.value[2], m.value[3]}
    {
    }

    QUALIFIER_DEVICE Mat4(const Vec4 &v0, const Vec4 &v1, const Vec4 &v2, const Vec4 &v3)
        : value{v0, v1, v2, v3}
    {
    }

    QUALIFIER_DEVICE Mat4(const float x0, const float y0, const float z0, const float w0,
                          const float x1, const float y1, const float z1, const float w1,
                          const float x2, const float y2, const float z2, const float w2,
                          const float x3, const float y3, const float z3, const float w3)
        : value{{x0, y0, z0, w0},
                {x1, y1, z1, w1},
                {x2, y2, z2, w2},
                {x3, y3, z3, w3}}
    {
    }

    QUALIFIER_DEVICE Vec4 &operator[](const int i)
    {
        assert(i < 4);
        return value[i];
    }

    QUALIFIER_DEVICE Vec4 operator[](const int i) const
    {
        assert(i < 4);
        return value[i];
    }

    QUALIFIER_DEVICE void operator=(const Mat4 &m)
    {
        value[0] = m.value[0];
        value[1] = m.value[1];
        value[2] = m.value[2];
        value[3] = m.value[3];
    }

    QUALIFIER_DEVICE const Mat4 &operator+() const { return *this; }

    QUALIFIER_DEVICE Mat4 operator-() const
    {
        return Mat4(-value[0], -value[1], -value[2], -value[3]);
    }

    QUALIFIER_DEVICE Mat4 &operator+=(const Mat4 &m)
    {
        value[0] += m.value[0];
        value[1] += m.value[1];
        value[2] += m.value[2];
        value[3] += m.value[3];
        return *this;
    }

    QUALIFIER_DEVICE Mat4 &operator-=(const Mat4 &m)
    {
        value[0] -= m.value[0];
        value[1] -= m.value[1];
        value[2] -= m.value[2];
        value[3] -= m.value[3];
        return *this;
    }

    QUALIFIER_DEVICE Mat4 &operator*=(const float t)
    {
        value[0] *= t;
        value[1] *= t;
        value[2] *= t;
        value[3] *= t;
        return *this;
    }

    QUALIFIER_DEVICE Mat4 &operator/=(const float t)
    {
        const float k = 1.0f / t;
        value[0] *= k;
        value[1] *= k;
        value[2] *= k;
        value[3] *= k;
        return *this;
    }

    QUALIFIER_DEVICE Mat4 Transpose() const
    {
        return Mat4(value[0][0], value[1][0], value[2][0], value[3][0],
                    value[0][1], value[1][1], value[2][1], value[3][1],
                    value[0][2], value[1][2], value[2][2], value[3][2],
                    value[0][3], value[1][3], value[2][3], value[3][3]);
    }

    QUALIFIER_DEVICE Mat4 Inverse() const;
};

inline QUALIFIER_DEVICE Mat4 operator*(float t, const Mat4 &m)
{
    return Mat4(t * m.value[0], t * m.value[1], t * m.value[2], t * m.value[3]);
}

inline QUALIFIER_DEVICE Mat4 operator*(const Mat4 &m, float t)
{
    return Mat4(t * m.value[0], t * m.value[1], t * m.value[2], t * m.value[3]);
}

inline QUALIFIER_DEVICE Mat4 operator/(const Mat4 &m, float t)
{
    const float k = 1.0f / t;
    return Mat4(k * m.value[0], k * m.value[1], k * m.value[2], k * m.value[3]);
}

inline QUALIFIER_DEVICE Mat4 operator+(const Mat4 &m0, const Mat4 &m1)
{
    return Mat4(m0.value[0] + m1.value[0], m0.value[1] + m1.value[1], m0.value[2] + m1.value[2], m0.value[3] + m1.value[3]);
}

inline QUALIFIER_DEVICE Mat4 operator-(const Mat4 &m0, const Mat4 &m1)
{
    return Mat4(m0.value[0] - m1.value[0], m0.value[1] - m1.value[1], m0.value[2] - m1.value[2], m0.value[3] - m1.value[3]);
}

inline QUALIFIER_DEVICE Vec4 Mul(const Mat4 &m, const Vec4 &v)
{
    return Vec4(Dot(m[0], v), Dot(m[1], v), Dot(m[2], v), Dot(m[3], v));
}

inline QUALIFIER_DEVICE Vec4 Mul(const Vec4 v, const Mat4 &m)
{
    const Mat4 m_trans = m.Transpose();
    return Vec4(Dot(v, m_trans[0]), Dot(v, m_trans[1]), Dot(v, m_trans[2]), Dot(v, m_trans[3]));
}

inline QUALIFIER_DEVICE Mat4 Mul(const Mat4 &m0, const Mat4 &m1)
{
    return Mat4(Mul(m0[0], m1), Mul(m0[1], m1), Mul(m0[2], m1), Mul(m0[3], m1));
}

inline QUALIFIER_DEVICE Mat4 Translate(const Vec3 &v)
{
    Mat4 translate;
    translate[0][3] += v.x;
    translate[1][3] += v.y;
    translate[2][3] += v.z;
    return translate;
}

inline QUALIFIER_DEVICE Mat4 Scale(const Vec3 &v)
{
    Mat4 scale;
    scale[0][0] *= v.x;
    scale[1][1] *= v.y;
    scale[2][2] *= v.z;
    return scale;
}

QUALIFIER_DEVICE Mat4 Rotate(float angle, Vec3 axis);

QUALIFIER_DEVICE Mat4 LookAtLH(const Vec3 &eye, const Vec3 &look_at, Vec3 up);