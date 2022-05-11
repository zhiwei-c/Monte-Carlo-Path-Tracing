#pragma once

#include <vector>

#include "../global.h"

NAMESPACE_BEGIN(raytracer)


inline bool FloatEqual(Float a, Float b, Float epsilon = kEpsilon)
{
    return std::fabs(a - b) < epsilon;
}

///\brief 限制输入的数在某个闭区间内
///\param bottom 闭区间下限
///\param top 闭区间上限
///\param num 待限制的数
///\return 限制后的数
template <typename T>
inline T Clamp(T bottom, T top, T num)
{
    return std::max(bottom, std::min(top, num));
}

inline Float CyclicClamp(Float num)
{
    while (num > 1)
        num -= 1;
    while (num < 0)
        num += 1;
    return num;
}

///\brief 模运算
inline int Modulo(int a, int b)
{
    auto c = a % b;
    if (c < 0)
        c += b;
    return c;
}

template <typename T>
inline T ClampTop(T top, T num)
{
    return std::min(top, num);
}

template <typename T>
inline T ClampBottom(T bottom, T num)
{
    return std::max(bottom, num);
}

///\brief 平方
template <typename T>
inline T Sqr(T num)
{
    return num * num;
}

///\brief 插值
inline Float Lerp(Float t, Float v1, Float v2)
{
    return (1 - t) * v1 + t * v2;
}

///\brief 求解一元二次方程
template <typename T>
bool SolveQuadratic(T a, T b, T c, Float &x0, Float &x1)
{
    /* Linear case */
    if (a == 0)
    {
        if (b != 0)
        {
            x0 = x1 = -c / b;
            return true;
        }
        return false;
    }
    Float discrim = b * b - 4 * a * c;
    /* Leave if there is no solution */
    if (discrim < 0)
        return false;
    Float temp, sqrtDiscrim = std::sqrt(discrim);
    /* Numerically stable version of (-b (+/-) sqrtDiscrim) / (2 * a)
     *
     * Based on the observation that one solution is always
     * accurate while the other is not. Finds the solution of
     * greater magnitude which does not suffer from loss of
     * precision and then uses the identity x1 * x2 = c / a
     */
    if (b < 0)
        temp = -0.5 * (b - sqrtDiscrim);
    else
        temp = -0.5 * (b + sqrtDiscrim);
    x0 = temp / a;
    x1 = c / temp;
    /* Return the results so that x0 < x1 */
    if (x0 > x1)
        std::swap(x0, x1);
    return true;
}
inline size_t BinarySearch(const std::vector<Float> &data, Float target)
{
    size_t begin = 0, end = data.size();
    while (begin + 1 < end)
    {
        auto mid = static_cast<size_t>((begin + end) * 0.5);
        if (data[mid] < target)
            begin = mid;
        else if (data[mid] > target)
            end = mid;
        else
            return mid;
    }
    return begin;
}

NAMESPACE_END(raytracer)
