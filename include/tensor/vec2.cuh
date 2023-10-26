#pragma once

#include "utils/defs.cuh"

NAMESPACE_BEGIN(rt)

struct Vec2
{
    union { float u; float x; };
    union { float v; float y; };

    QUALIFIER_DEVICE Vec2() : u(0), v(0) {}
    QUALIFIER_DEVICE Vec2(float u) : u(u), v(u) {}
    QUALIFIER_DEVICE Vec2(float u, float v) : u(u), v(v) {}

    QUALIFIER_DEVICE float &operator[](int i) { return i == 0 ? u : v; }
    QUALIFIER_DEVICE float operator[](int i) const { return i == 0 ? u : v; }

    QUALIFIER_DEVICE void operator=(const Vec2 &vec) { u = vec.u, v = vec.v; }

    QUALIFIER_DEVICE const Vec2 &operator+() const { return *this; }
    QUALIFIER_DEVICE Vec2 operator-() const { return {-u, -v}; }

    QUALIFIER_DEVICE Vec2 &operator+=(const Vec2 &vec);
    QUALIFIER_DEVICE Vec2 &operator-=(const Vec2 &vec);
    QUALIFIER_DEVICE Vec2 &operator*=(const Vec2 &vec);
    QUALIFIER_DEVICE Vec2 &operator/=(const Vec2 &vec);

    QUALIFIER_DEVICE Vec2 &operator*=(const float t);
    QUALIFIER_DEVICE Vec2 &operator/=(const float t);

    QUALIFIER_DEVICE float Length();
    QUALIFIER_DEVICE Vec2 Normalize();
};

QUALIFIER_DEVICE Vec2 operator+(const Vec2 &v1, const Vec2 &v2);
QUALIFIER_DEVICE Vec2 operator-(const Vec2 &v1, const Vec2 &v2);
QUALIFIER_DEVICE Vec2 operator*(const Vec2 &v1, const Vec2 &v2);
QUALIFIER_DEVICE Vec2 operator/(const Vec2 &v1, const Vec2 &v2);

QUALIFIER_DEVICE Vec2 operator+(const Vec2 &vec, float t);
QUALIFIER_DEVICE Vec2 operator-(const Vec2 &vec, float t);
QUALIFIER_DEVICE Vec2 operator*(const Vec2 &vec, float t);
QUALIFIER_DEVICE Vec2 operator/(const Vec2 &vec, float t);

QUALIFIER_DEVICE Vec2 operator+(float t, const Vec2 &vec);
QUALIFIER_DEVICE Vec2 operator-(float t, const Vec2 &vec);
QUALIFIER_DEVICE Vec2 operator*(float t, const Vec2 &vec);

QUALIFIER_DEVICE float Dot(const Vec2 &v1, const Vec2 &v2);
QUALIFIER_DEVICE Vec2 Min(const Vec2 &v1, const Vec2 &v2);
QUALIFIER_DEVICE Vec2 Max(const Vec2 &v1, const Vec2 &v2);

NAMESPACE_END(rt)