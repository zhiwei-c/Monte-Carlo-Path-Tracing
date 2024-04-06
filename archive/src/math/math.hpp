#pragma once

#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

constexpr double kPi = glm::pi<double>();
constexpr double kPiRcp = 1.0 / kPi;
constexpr double kEpsilonJitter = 0.01 * kPiRcp;
constexpr double kMaxDouble = std::numeric_limits<double>::max();
constexpr double kLowestDouble = std::numeric_limits<double>::lowest();
constexpr double kEpsilonCompare = 1e-10;
constexpr double kEpsilonDistance = 1e-6;
constexpr double kEpsilonPdf = 0.01;
constexpr double kEpsilonMachine = std::numeric_limits<double>::epsilon() * 0.5;
constexpr double kAabbErrorBound = 1.0 + 6.0 * kEpsilonMachine / (1.0 - 3.0 * kEpsilonMachine);

///\brief 乘方
inline double Sqr(double n)
{
    return n * n;
}

///\brief 乘方
inline dvec3 Sqr(dvec3 n)
{
    return n * n;
}

///\brief 模运算
inline int Modulo(int a, int b)
{
    auto c = a % b;
    if (c < 0)
    {
        c += b;
    }
    return c;
}

///\brief 插值
inline double Lerp(double t, double v1, double v2)
{
    return (1.0 - t) * v1 + t * v2;
}

///\brief 插值
inline dvec3 Lerp(double t, dvec3 v1, dvec3 v2)
{
    return (1.0 - t) * v1 + t * v2;
}

/// @brief 根据行号和列号转换莫顿数
/// @param column 行
/// @param row 列
/// @return 十进制莫顿数
inline uint64_t GetMortonCode(uint32_t column, uint32_t row)
{
    uint64_t morton = 0;
    for (int i = 0; i < sizeof(row) * 8; ++i)
    {
        morton |= (row & (uint64_t)1 << i) << i | (column & (uint64_t)1 << i) << (i + 1);
    }
    return morton;
}

/// @brief 求解一元二次方程
/// @param a 二次项系数
/// @param b 一次项系数
/// @param c 常数项
/// @param x0 数值较小的那个实数解
/// @param x1 数值较大的那个实数解
/// @return 方程是否有实数解
inline bool SolveQuadratic(double a, double b, double c, double *x0, double *x1)
{
    /* Linear case */
    if (a == 0.0)
    {
        if (b != 0.0)
        {
            *x0 = *x1 = -c / b;
            return true;
        }
        return false;
    }
    double discrim = b * b - 4.0 * a * c;
    /* Leave if there is no solution */
    if (discrim < 0.0)
    {
        return false;
    }
    double temp, sqrtDiscrim = std::sqrt(discrim);
    /* Numerically stable version of (-b (+/-) sqrtDiscrim) / (2 * a)
     *
     * Based on the observation that one solution is always
     * accurate while the other is not. Finds the solution of
     * greater magnitude which does not suffer from loss of
     * precision and then uses the identity x1 * x2 = c / a
     */
    if (b < 0.0)
    {
        temp = -0.5 * (b - sqrtDiscrim);
    }
    else
    {
        temp = -0.5 * (b + sqrtDiscrim);
    }
    *x0 = temp / a;
    *x1 = c / temp;
    /* Return the results so that x0 < x1 */
    if (*x0 > *x1)
    {
        std::swap(*x0, *x1);
    }
    return true;
}

NAMESPACE_END(raytracer)