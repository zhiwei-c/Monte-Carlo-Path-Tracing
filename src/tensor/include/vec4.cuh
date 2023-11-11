#pragma once

#include "defs.cuh"
#include "vec3.cuh"

namespace rt
{

struct Vec4
{
    union
    {
        float x;
        float r;
    };
    union
    {
        float y;
        float g;
    };
    union
    {
        float z;
        float b;
    };
    union
    {
        float w;
        float a;
    };

    QUALIFIER_D_H Vec4() : x(0), y(0), z(0), w(0) {}
    QUALIFIER_D_H Vec4(const float x) : x(x), y(x), z(x), w(x) {}
    QUALIFIER_D_H Vec4(const Vec3 &vec, float w);
    QUALIFIER_D_H Vec4(const float x, const float y, const float z,
                       const float w);

    QUALIFIER_D_H float &operator[](const int i);
    QUALIFIER_D_H float operator[](const int i) const;

    QUALIFIER_D_H void operator=(const Vec4 &vec);

    QUALIFIER_D_H const Vec4 &operator+() const { return *this; }
    QUALIFIER_D_H Vec4 operator-() const { return {-x, -y, -z, -w}; }

    QUALIFIER_D_H Vec4 &operator+=(const Vec4 &vec);
    QUALIFIER_D_H Vec4 &operator-=(const Vec4 &vec);
    QUALIFIER_D_H Vec4 &operator*=(const Vec4 &vec);
    QUALIFIER_D_H Vec4 &operator/=(const Vec4 &vec);

    QUALIFIER_D_H Vec4 &operator*=(const float t);
    QUALIFIER_D_H Vec4 &operator/=(const float t);

    QUALIFIER_D_H Vec3 direction() const { return Normalize({x, y, z}); }
    QUALIFIER_D_H Vec3 position() const;
};

QUALIFIER_D_H Vec4 operator+(const Vec4 &v1, const Vec4 &v2);
QUALIFIER_D_H Vec4 operator-(const Vec4 &v1, const Vec4 &v2);
QUALIFIER_D_H Vec4 operator*(const Vec4 &v1, const Vec4 &v2);
QUALIFIER_D_H Vec4 operator/(const Vec4 &v1, const Vec4 &v2);

QUALIFIER_D_H Vec4 operator+(const Vec4 &vec, const float t);
QUALIFIER_D_H Vec4 operator-(const Vec4 &vec, const float t);
QUALIFIER_D_H Vec4 operator*(const Vec4 &vec, const float t);
QUALIFIER_D_H Vec4 operator/(const Vec4 &vec, const float t);

QUALIFIER_D_H Vec4 operator+(const float t, const Vec4 &vec);
QUALIFIER_D_H Vec4 operator-(const float t, const Vec4 &vec);
QUALIFIER_D_H Vec4 operator*(const float t, const Vec4 &vec);

QUALIFIER_D_H float Dot(const Vec4 &v1, const Vec4 &v2);
QUALIFIER_D_H Vec4 Min(const Vec4 &v1, const Vec4 &v2);
QUALIFIER_D_H Vec4 Max(const Vec4 &v1, const Vec4 &v2);

} // namespace rt