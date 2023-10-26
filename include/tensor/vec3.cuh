#pragma once

#include "utils/defs.cuh"

NAMESPACE_BEGIN(rt)

struct Vec3
{
    union { float x; float r; };
    union { float y; float g; };
    union { float z; float b; };

    QUALIFIER_DEVICE Vec3() : x(0), y(0), z(0) {}
    QUALIFIER_DEVICE Vec3(float x) : x(x), y(x), z(x) {}
    QUALIFIER_DEVICE Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    QUALIFIER_DEVICE float &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
    QUALIFIER_DEVICE float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }

    QUALIFIER_DEVICE void operator=(const Vec3 &vec) { x = vec.x, y = vec.y, z = vec.z; }

    QUALIFIER_DEVICE const Vec3 &operator+() const { return *this; }
    QUALIFIER_DEVICE Vec3 operator-() const { return {-x, -y, -z}; }

    QUALIFIER_DEVICE Vec3 &operator+=(const Vec3 &vec);
    QUALIFIER_DEVICE Vec3 &operator-=(const Vec3 &vec);
    QUALIFIER_DEVICE Vec3 &operator*=(const Vec3 &vec);
    QUALIFIER_DEVICE Vec3 &operator/=(const Vec3 &vec);

    QUALIFIER_DEVICE Vec3 &operator*=(const float t);
    QUALIFIER_DEVICE Vec3 &operator/=(const float t);
};

QUALIFIER_DEVICE Vec3 operator+(const Vec3 &v1, const Vec3 &v2);
QUALIFIER_DEVICE Vec3 operator-(const Vec3 &v1, const Vec3 &v2);
QUALIFIER_DEVICE Vec3 operator*(const Vec3 &v1, const Vec3 &v2);
QUALIFIER_DEVICE Vec3 operator/(const Vec3 &v1, const Vec3 &v2);

QUALIFIER_DEVICE Vec3 operator+(const Vec3 &vec, float t);
QUALIFIER_DEVICE Vec3 operator-(const Vec3 &vec, float t);
QUALIFIER_DEVICE Vec3 operator*(const Vec3 &vec, float t);
QUALIFIER_DEVICE Vec3 operator/(const Vec3 &vec, float t);

QUALIFIER_DEVICE Vec3 operator+(float t, const Vec3 &vec);
QUALIFIER_DEVICE Vec3 operator-(float t, const Vec3 &vec);
QUALIFIER_DEVICE Vec3 operator*(float t, const Vec3 &vec);

QUALIFIER_DEVICE float Length(const Vec3 &vec);
QUALIFIER_DEVICE Vec3 Normalize(const Vec3 &vec);

QUALIFIER_DEVICE float Dot(const Vec3 &v1, const Vec3 &v2);
QUALIFIER_DEVICE Vec3 Cross(const Vec3 &v1, const Vec3 &v2);
QUALIFIER_DEVICE Vec3 Min(const Vec3 &v1, const Vec3 &v2);
QUALIFIER_DEVICE Vec3 Max(const Vec3 &v1, const Vec3 &v2);

NAMESPACE_END(rt)