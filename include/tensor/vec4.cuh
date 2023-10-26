#pragma once

#include "utils/defs.cuh"

NAMESPACE_BEGIN(rt)

struct Vec4
{
    union { float x; float r; };
    union { float y; float g; };
    union { float z; float b; };
    union { float w; float a; };

    QUALIFIER_DEVICE Vec4() : x(0), y(0), z(0), w(0) {}
    QUALIFIER_DEVICE Vec4(float x) : x(x), y(x), z(x), w(x) {}
    QUALIFIER_DEVICE Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    QUALIFIER_DEVICE float &operator[](int i) { return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w)); }
    QUALIFIER_DEVICE float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w)); }

    QUALIFIER_DEVICE void operator=(const Vec4 &vec) { x = vec.x, y = vec.y, z = vec.z, w = vec.w; }

    QUALIFIER_DEVICE const Vec4 &operator+() const { return *this; }
    QUALIFIER_DEVICE Vec4 operator-() const { return {-x, -y, -z, -w}; }

    QUALIFIER_DEVICE Vec4 &operator+=(const Vec4 &vec);
    QUALIFIER_DEVICE Vec4 &operator-=(const Vec4 &vec);
    QUALIFIER_DEVICE Vec4 &operator*=(const Vec4 &vec);
    QUALIFIER_DEVICE Vec4 &operator/=(const Vec4 &vec);

    QUALIFIER_DEVICE Vec4 &operator*=(const float t);
    QUALIFIER_DEVICE Vec4 &operator/=(const float t);
};

QUALIFIER_DEVICE Vec4 operator+(const Vec4 &v1, const Vec4 &v2);
QUALIFIER_DEVICE Vec4 operator-(const Vec4 &v1, const Vec4 &v2);
QUALIFIER_DEVICE Vec4 operator*(const Vec4 &v1, const Vec4 &v2);
QUALIFIER_DEVICE Vec4 operator/(const Vec4 &v1, const Vec4 &v2);

QUALIFIER_DEVICE Vec4 operator+(const Vec4 &vec, float t);
QUALIFIER_DEVICE Vec4 operator-(const Vec4 &vec, float t);
QUALIFIER_DEVICE Vec4 operator*(const Vec4 &vec, float t);
QUALIFIER_DEVICE Vec4 operator/(const Vec4 &vec, float t);

QUALIFIER_DEVICE Vec4 operator+(float t, const Vec4 &vec);
QUALIFIER_DEVICE Vec4 operator-(float t, const Vec4 &vec);
QUALIFIER_DEVICE Vec4 operator*(float t, const Vec4 &vec);

QUALIFIER_DEVICE float Dot(const Vec4 &v1, const Vec4 &v2);
QUALIFIER_DEVICE Vec4 Min(const Vec4 &v1, const Vec4 &v2);
QUALIFIER_DEVICE Vec4 Max(const Vec4 &v1, const Vec4 &v2);

NAMESPACE_END(rt)