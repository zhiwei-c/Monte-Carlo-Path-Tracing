#include "math.cuh"

#include <cassert>

QUALIFIER_DEVICE Vec3 ToLocal(const Vec3 &dir, const Vec3 &up)
{
    Vec3 B = Vec3(0), C = Vec3(0);
    if (sqrt(up.x * up.x + up.z * up.z) > kEpsilon)
    {
        float len_inv = 1.0f / sqrt(up.x * up.x + up.z * up.z);
        C = Vec3(up.z * len_inv, 0, -up.x * len_inv);
    }
    else
    {
        float len_inv = 1.0 / sqrt(up.y * up.y + up.z * up.z);
        C = Vec3(0, up.z * len_inv, -up.y * len_inv);
    }
    B = Cross(C, up);
    return Vec3(Dot(dir, B), Dot(dir, C), Dot(dir, up));
}

QUALIFIER_DEVICE Vec3 ToWorld(const Vec3 &dir, const Vec3 &normal)
{
    Vec3 B = Vec3(0), C = Vec3(0);
    if (sqrt(normal.x * normal.x + normal.z * normal.z) > kEpsilon)
    {
        float len_inv = 1.0f / sqrt(normal.x * normal.x + normal.z * normal.z);
        C = Vec3(normal.z * len_inv, 0, -normal.x * len_inv);
    }
    else
    {
        float len_inv = 1.0f / sqrt(normal.y * normal.y + normal.z * normal.z);
        C = Vec3(0, normal.z * len_inv, -normal.y * len_inv);
    }
    B = Cross(C, normal);
    return Normalize(dir.x * B + dir.y * C + dir.z * normal);
}

///\brief 将向量从笛卡尔坐标系转换到球坐标系
///\param dir - 待转换的单位向量
///\param theta - 向量与 Up 方向的夹角（天顶角）
///\param phi - 向量与 Front 方向的夹角（方位角）
///\param r - 向量的长度
QUALIFIER_DEVICE void CartesianToSpherical(Vec3 vec, float *theta, float *phi, float *r)
{
    if (r != nullptr)
        *r = Length(vec);
    vec = Normalize(vec);
    *theta = acosf(fminf(1.0f, fmaxf(-1.0f, vec[UP_DIM_WORLD])));
    *phi = atan2f(vec[RIGHT_DIM_WORLD], -vec[FRONT_DIM_WORLD]);
    if (*phi < 0.0f)
        *phi += 2.0f * kPi;
}

QUALIFIER_DEVICE bool SolveQuadratic(float a, float b, float c, float &x0, float &x1)
{
    /* Linear case */
    if (a == 0.0f)
    {
        if (b != 0.0f)
        {
            x0 = x1 = -c / b;
            return true;
        }
        return false;
    }
    float discrim = b * b - 4.0f * a * c;
    /* Leave if there is no solution */
    if (discrim < 0.0f)
    {
        return false;
    }
    float temp, sqrtDiscrim = sqrt(discrim);
    /* Numerically stable version of (-b (+/-) sqrtDiscrim) / (2 * a)
     *
     * Based on the observation that one solution is always
     * accurate while the other is not. Finds the solution of
     * greater magnitude which does not suffer from loss of
     * precision and then uses the identity x1 * x2 = c / a
     */
    if (b < 0.0f)
    {
        temp = -0.5f * (b - sqrtDiscrim);
    }
    else
    {
        temp = -0.5f * (b + sqrtDiscrim);
    }
    x0 = temp / a;
    x1 = c / temp;
    /* Return the results so that x0 < x1 */
    if (x0 > x1)
    {
        float temp = x0;
        x0 = x1;
        x1 = temp;
    }
    return true;
}