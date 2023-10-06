#pragma once

#include <cmath>

#include "../global.cuh"

struct Vec2
{

    float u, v;

    QUALIFIER_DEVICE Vec2() : u(0), v(0) {}

    QUALIFIER_DEVICE Vec2(float u) : u(u), v(u) {}

    QUALIFIER_DEVICE Vec2(float u, float v) : u(u), v(v) {}

    QUALIFIER_DEVICE float &operator[](int i)
    {
        switch (i)
        {
        case 0:
            return u;
            break;
        default:
            return v;
            break;
        }
    }

    QUALIFIER_DEVICE float operator[](int i) const
    {
        switch (i)
        {
        case 0:
            return u;
            break;
        default:
            return v;
            break;
        }
    }

    QUALIFIER_DEVICE void operator=(const Vec2 &v1)
    {
        u = v1.u;
        v = v1.v;
    }

    QUALIFIER_DEVICE const Vec2 &operator+() const { return *this; }
    QUALIFIER_DEVICE Vec2 operator-() const { return Vec2(-u, -v); }

    QUALIFIER_DEVICE Vec2 &operator+=(const Vec2 &vec)
    {
        u += vec.u;
        v += vec.v;
        return *this;
    }

    QUALIFIER_DEVICE Vec2 &operator-=(const Vec2 &vec)
    {
        u -= vec.u;
        v -= vec.v;
        return *this;
    }

    QUALIFIER_DEVICE Vec2 &operator*=(const Vec2 &vec)
    {
        u *= vec.u;
        v *= vec.v;
        return *this;
    }

    QUALIFIER_DEVICE Vec2 &operator/=(const Vec2 &vec)
    {
        u /= vec.u;
        v /= vec.v;
        return *this;
    }

    QUALIFIER_DEVICE Vec2 &operator*=(const float t)
    {
        u *= t;
        v *= t;
        return *this;
    }

    QUALIFIER_DEVICE Vec2 &operator/=(const float t)
    {
        const float k = 1.0f / t;
        u *= k;
        v *= k;
        return *this;
    }

    QUALIFIER_DEVICE float Length() const
    {
        return fmaxf(sqrt(u * u + v * v), kEpsilonFloat);
    }

    QUALIFIER_DEVICE float SquaredLength() const
    {
        return fmaxf(u * u + v * v, kEpsilonFloat);
    }
};

inline QUALIFIER_DEVICE Vec2 operator+(const Vec2 &v1, const Vec2 &v2)
{
    return Vec2(v1.u + v2.u, v1.v + v2.v);
}

inline QUALIFIER_DEVICE Vec2 operator-(const Vec2 &v1, const Vec2 &v2)
{
    return Vec2(v1.u - v2.u, v1.v - v2.v);
}

inline QUALIFIER_DEVICE Vec2 operator*(const Vec2 &v1, const Vec2 &v2)
{
    return Vec2(v1.u * v2.u, v1.v * v2.v);
}

inline QUALIFIER_DEVICE Vec2 operator/(const Vec2 &v1, const Vec2 &v2)
{
    return Vec2(v1.u / v2.u, v1.v / v2.v);
}

inline QUALIFIER_DEVICE Vec2 operator+(const Vec2 &vec, float t)
{
    return Vec2(t + vec.u, t + vec.v);
}

inline QUALIFIER_DEVICE Vec2 operator+(float t, const Vec2 &vec)
{
    return Vec2(t + vec.u, t + vec.v);
}

inline QUALIFIER_DEVICE Vec2 operator-(float t, const Vec2 &vec)
{
    return Vec2(t - vec.u, t - vec.v);
}

inline QUALIFIER_DEVICE Vec2 operator-(const Vec2 &vec, float t)
{
    return Vec2(vec.u - t, vec.v - t);
}

inline QUALIFIER_DEVICE Vec2 operator*(float t, const Vec2 &vec)
{
    return Vec2(t * vec.u, t * vec.v);
}

inline QUALIFIER_DEVICE Vec2 operator/(const Vec2 &vec, float t)
{
    const float k = 1.0f / t;
    return Vec2(vec.u * k, vec.v * k);
}

inline QUALIFIER_DEVICE Vec2 operator*(const Vec2 &vec, float t)
{
    return Vec2(t * vec.u, t * vec.v);
}

inline QUALIFIER_DEVICE float Dot(const Vec2 &v1, const Vec2 &v2)
{
    return v1.u * v2.u + v1.v * v2.v;
}

inline QUALIFIER_DEVICE Vec2 Min(const Vec2 &v1, const Vec2 &v2)
{
    return Vec2(fminf(v1.u, v2.u), fminf(v1.v, v2.v));
}

inline QUALIFIER_DEVICE Vec2 Max(const Vec2 &v1, const Vec2 &v2)
{
    return Vec2(fmaxf(v1.u, v2.u), fmaxf(v1.v, v2.v));
}

inline QUALIFIER_DEVICE Vec2 Normalize(const Vec2 &vec)
{
    return vec / vec.Length();
}

inline QUALIFIER_DEVICE float Length(const Vec2 &vec)
{
    return vec.Length();
}

inline QUALIFIER_DEVICE Vec2 Sqr(const Vec2 &vec)
{
    return vec * vec;
}
