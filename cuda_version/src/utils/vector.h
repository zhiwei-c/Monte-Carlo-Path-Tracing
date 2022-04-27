#pragma once

#include <glm/glm.hpp>

using Float = double;
using uint = unsigned int;

namespace myvec
{
    struct uvec2
    {
        uint x, y;

        __host__ __device__ uvec2() : x(0), y(0) {}

        __host__ __device__ uvec2(uint x) : x(x), y(x) {}

        __host__ __device__ uvec2(uint x, uint y) : x(x), y(y) {}

        __host__ __device__ uint &operator[](uint i)
        {
            switch (i)
            {
            case 0:
                return x;
                break;
            default:
                return y;
                break;
            }
        }
    };

    struct uvec3
    {
        uint x, y, z;

        __host__ __device__ uint &operator[](uint i)
        {
            switch (i)
            {
            case 0:
                return x;
                break;
            case 1:
                return y;
                break;
            default:
                return z;
                break;
            }
        }
    };

    struct vec2
    {
        Float x, y;

        __host__ __device__ vec2() : x(0), y(0) {}

        __host__ __device__ vec2(Float x) : x(x), y(x) {}

        __host__ __device__ vec2(Float x, Float y) : x(x), y(y) {}

        __host__ __device__ vec2(const vec2& v) : x(v.x), y(v.y) {}

        __host__ __device__ Float &operator[](size_t i)
        {
            switch (i)
            {
            case 0:
                return x;
                break;
            default:
                return y;
                break;
            }
        }

        __host__ __device__ void operator=(const glm::vec2 &v1)
        {
            x = v1.x;
            y = v1.y;
        }
    };

    __host__ __device__ inline vec2 operator+(const vec2 &v1, const vec2 &v2)
    {
        return vec2(v1.x + v2.x, v1.y + v2.y);
    }

    __host__ __device__ inline vec2 operator-(const vec2 &v1, const vec2 &v2)
    {
        return vec2(v1.x - v2.x, v1.y - v2.y);
    }

    __host__ __device__ inline vec2 operator*(Float t, const vec2 &v)
    {
        return vec2(t * v.x, t * v.y);
    }

    __host__ __device__ inline vec2 operator*(const vec2 &v, Float t)
    {
        return vec2(t * v.x, t * v.y);
    }

    struct vec3
    {

        Float x, y, z;

        __host__ __device__ vec3() : x(0), y(0), z(0) {}

        __host__ __device__ vec3(Float x) : x(x), y(x), z(x) {}

        __host__ __device__ vec3(Float x, Float y, Float z) : x(x), y(y), z(z) {}

        __host__ __device__ Float &operator[](int i)
        {
            switch (i)
            {
            case 0:
                return x;
                break;
            case 1:
                return y;
                break;
            default:
                return z;
                break;
            }
        }

        __host__ __device__ Float operator[](int i) const
        {
            switch (i)
            {
            case 0:
                return x;
                break;
            case 1:
                return y;
                break;
            default:
                return z;
                break;
            }
        }

        __host__ __device__ void operator=(const glm::vec3 &v1)
        {
            x = v1.x;
            y = v1.y;
            z = v1.z;
        }

        __host__ __device__ inline const vec3 &operator+() const { return *this; }
        __host__ __device__ inline vec3 operator-() const { return vec3(-x, -y, -z); }

        __host__ __device__ inline vec3 &operator+=(const vec3 &v2);
        __host__ __device__ inline vec3 &operator-=(const vec3 &v2);
        __host__ __device__ inline vec3 &operator*=(const vec3 &v2);
        __host__ __device__ inline vec3 &operator/=(const vec3 &v2);
        __host__ __device__ inline vec3 &operator*=(const Float t);
        __host__ __device__ inline vec3 &operator/=(const Float t);

        __host__ __device__ inline Float length() const { return sqrt(x * x + y * y + z * z); }
        __host__ __device__ inline Float squared_length() const { return x * x + y * y + z * z; }

        __host__ __device__ inline void normalize()
        {
            Float k = 1.0 / sqrt(x * x + y * y + z * z);
            x *= k;
            y *= k;
            z *= k;
        }
    };

    __host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2)
    {
        return vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
    }

    __host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2)
    {
        return vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
    }

    __host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2)
    {
        return vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
    }

    __host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2)
    {
        return vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
    }

    __host__ __device__ inline vec3 operator*(Float t, const vec3 &v)
    {
        return vec3(t * v.x, t * v.y, t * v.z);
    }

    __host__ __device__ inline vec3 operator/(vec3 v, Float t)
    {
        return vec3(v.x / t, v.y / t, v.z / t);
    }

    __host__ __device__ inline vec3 operator*(const vec3 &v, Float t)
    {
        return vec3(t * v.x, t * v.y, t * v.z);
    }

    __host__ __device__ inline Float dot(const vec3 &v1, const vec3 &v2)
    {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }

    __host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2)
    {
        return vec3((v1.y * v2.z - v1.z * v2.y),
                    (-(v1.x * v2.z - v1.z * v2.x)),
                    (v1.x * v2.y - v1.y * v2.x));
    }

    __host__ __device__ inline vec3 min(const vec3 &v1, const vec3 &v2)
    {
        return vec3(glm::min(v1.x, v2.x),
                    glm::min(v1.y, v2.y),
                    glm::min(v1.z, v2.z));
    }

    __host__ __device__ inline vec3 max(const vec3 &v1, const vec3 &v2)
    {
        return vec3(glm::max(v1.x, v2.x),
                    glm::max(v1.y, v2.y),
                    glm::max(v1.z, v2.z));
    }

    __host__ __device__ inline vec3 normalize(const vec3 &v1)
    {
        return v1 / v1.length();
    }

    __host__ __device__ inline Float length(const vec3 &v1)
    {
        return v1.length();
    }

    __host__ __device__ inline vec3 &vec3::operator+=(const vec3 &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    __host__ __device__ inline vec3 &vec3::operator-=(const vec3 &v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    __host__ __device__ inline vec3 &vec3::operator*=(const vec3 &v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }

    __host__ __device__ inline vec3 &vec3::operator/=(const vec3 &v)
    {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        return *this;
    }

    __host__ __device__ inline vec3 &vec3::operator*=(const Float t)
    {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    __host__ __device__ inline vec3 &vec3::operator/=(const Float t)
    {
        Float k = 1.0 / t;

        x *= k;
        y *= k;
        z *= k;
        return *this;
    }

} // namespace myvec
