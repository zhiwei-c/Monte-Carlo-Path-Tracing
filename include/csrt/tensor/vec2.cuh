#pragma once

#include "../defs.cuh"

namespace csrt
{

struct Vec2
{
    union
    {
        float u;
        float x;
    };
    union
    {
        float v;
        float y;
    };

    QUALIFIER_D_H Vec2() : u(0), v(0) {}
    QUALIFIER_D_H Vec2(const float u) : u(u), v(u) {}
    QUALIFIER_D_H Vec2(const float u, const float v) : u(u), v(v) {}

    QUALIFIER_D_H float &operator[](const int i) { return i == 0 ? u : v; }
    QUALIFIER_D_H float operator[](const int i) const { return i == 0 ? u : v; }

    QUALIFIER_D_H void operator=(const Vec2 &vec) { u = vec.u, v = vec.v; }

    QUALIFIER_D_H const Vec2 &operator+() const { return *this; }
    QUALIFIER_D_H Vec2 operator-() const { return {-u, -v}; }

    QUALIFIER_D_H Vec2 &operator+=(const Vec2 &vec);
    QUALIFIER_D_H Vec2 &operator-=(const Vec2 &vec);
    QUALIFIER_D_H Vec2 &operator*=(const Vec2 &vec);
    QUALIFIER_D_H Vec2 &operator/=(const Vec2 &vec);

    QUALIFIER_D_H Vec2 &operator*=(const float t);
    QUALIFIER_D_H Vec2 &operator/=(const float t);

    QUALIFIER_D_H float Length();
    QUALIFIER_D_H Vec2 Normalize();
};

QUALIFIER_D_H Vec2 operator+(const Vec2 &v1, const Vec2 &v2);
QUALIFIER_D_H Vec2 operator-(const Vec2 &v1, const Vec2 &v2);
QUALIFIER_D_H Vec2 operator*(const Vec2 &v1, const Vec2 &v2);
QUALIFIER_D_H Vec2 operator/(const Vec2 &v1, const Vec2 &v2);

QUALIFIER_D_H Vec2 operator+(const Vec2 &vec, const float t);
QUALIFIER_D_H Vec2 operator-(const Vec2 &vec, const float t);
QUALIFIER_D_H Vec2 operator*(const Vec2 &vec, const float t);
QUALIFIER_D_H Vec2 operator/(const Vec2 &vec, const float t);

QUALIFIER_D_H Vec2 operator+(const float t, const Vec2 &vec);
QUALIFIER_D_H Vec2 operator-(const float t, const Vec2 &vec);
QUALIFIER_D_H Vec2 operator*(const float t, const Vec2 &vec);

QUALIFIER_D_H float Dot(const Vec2 &v1, const Vec2 &v2);
QUALIFIER_D_H Vec2 Min(const Vec2 &v1, const Vec2 &v2);
QUALIFIER_D_H Vec2 Max(const Vec2 &v1, const Vec2 &v2);

} // namespace csrt