#include "mat4.cuh"

#include <cassert>
#include <cmath>

namespace rt
{

QUALIFIER_D_H Mat4::Mat4()
    : rows{{1.0f, 0.0f, 0.0f, 0.0f},
           {0.0f, 1.0f, 0.0f, 0.0f},
           {0.0f, 0.0f, 1.0f, 0.0f},
           {0.0f, 0.0f, 0.0f, 1.0f}}
{
}

QUALIFIER_D_H Mat4::Mat4(const Mat4 &m)
    : rows{m.rows[0], m.rows[1], m.rows[2], m.rows[3]}
{
}

QUALIFIER_D_H Mat4::Mat4(const Vec4 &row0, const Vec4 &row1, const Vec4 &row2,
                         const Vec4 &row3)
    : rows{row0, row1, row2, row3}
{
}

QUALIFIER_D_H
Mat4::Mat4(const float x0, const float y0, const float z0, const float w0,
           const float x1, const float y1, const float z1, const float w1,
           const float x2, const float y2, const float z2, const float w2,
           const float x3, const float y3, const float z3, const float w3)
    : rows{{x0, y0, z0, w0},
           {x1, y1, z1, w1},
           {x2, y2, z2, w2},
           {x3, y3, z3, w3}}
{
}

QUALIFIER_D_H Vec4 &Mat4::operator[](const int i)
{
    assert(i < 4);
    return rows[i];
}

QUALIFIER_D_H Vec4 Mat4::operator[](const int i) const
{
    assert(i < 4);
    return rows[i];
}

QUALIFIER_D_H void Mat4::operator=(const Mat4 &m)
{
    rows[0] = m.rows[0];
    rows[1] = m.rows[1];
    rows[2] = m.rows[2];
    rows[3] = m.rows[3];
}

QUALIFIER_D_H Mat4 &Mat4::operator+=(const Mat4 &m)
{
    rows[0] += m.rows[0];
    rows[1] += m.rows[1];
    rows[2] += m.rows[2];
    rows[3] += m.rows[3];
    return *this;
}

QUALIFIER_D_H Mat4 &Mat4::operator-=(const Mat4 &m)
{
    rows[0] -= m.rows[0];
    rows[1] -= m.rows[1];
    rows[2] -= m.rows[2];
    rows[3] -= m.rows[3];
    return *this;
}

QUALIFIER_D_H Mat4 &Mat4::operator*=(const float t)
{
    rows[0] *= t;
    rows[1] *= t;
    rows[2] *= t;
    rows[3] *= t;
    return *this;
}

QUALIFIER_D_H Mat4 &Mat4::operator/=(const float t)
{
    const float k = 1.0f / t;
    rows[0] *= k;
    rows[1] *= k;
    rows[2] *= k;
    rows[3] *= k;
    return *this;
}

QUALIFIER_D_H Mat4 Mat4::Transpose() const
{
    return {{rows[0].x, rows[1].x, rows[2].x, rows[3].x},
            {rows[0].y, rows[1].y, rows[2].y, rows[3].y},
            {rows[0].z, rows[1].z, rows[2].z, rows[3].z},
            {rows[0].w, rows[1].w, rows[2].w, rows[3].w}};
}

QUALIFIER_D_H Mat4 Mat4::Inverse() const
{
    const float coef00 = rows[2].z * rows[3].w - rows[3].z * rows[2].w,
                coef02 = rows[1].z * rows[3].w - rows[3].z * rows[1].w,
                coef03 = rows[1].z * rows[2].w - rows[2].z * rows[1].w;

    const float coef04 = rows[2].y * rows[3].w - rows[3].y * rows[2].w,
                coef06 = rows[1].y * rows[3].w - rows[3].y * rows[1].w,
                coef07 = rows[1].y * rows[2].w - rows[2].y * rows[1].w;

    const float coef08 = rows[2].y * rows[3].z - rows[3].y * rows[2].z,
                coef10 = rows[1].y * rows[3].z - rows[3].y * rows[1].z,
                coef11 = rows[1].y * rows[2].z - rows[2].y * rows[1].z;

    const float coef12 = rows[2].x * rows[3].w - rows[3].x * rows[2].w,
                coef14 = rows[1].x * rows[3].w - rows[3].x * rows[1].w,
                coef15 = rows[1].x * rows[2].w - rows[2].x * rows[1].w;

    const float coef16 = rows[2].x * rows[3].z - rows[3].x * rows[2].z,
                coef18 = rows[1].x * rows[3].z - rows[3].x * rows[1].z,
                coef19 = rows[1].x * rows[2].z - rows[2].x * rows[1].z;

    const float coef20 = rows[2].x * rows[3].y - rows[3].x * rows[2].y,
                coef22 = rows[1].x * rows[3].y - rows[3].x * rows[1].y,
                coef23 = rows[1].x * rows[2].y - rows[2].x * rows[1].y;

    const Vec4 fac0 = {coef00, coef00, coef02, coef03},
               fac1 = {coef04, coef04, coef06, coef07},
               fac2 = {coef08, coef08, coef10, coef11},
               fac3 = {coef12, coef12, coef14, coef15},
               fac4 = {coef16, coef16, coef18, coef19},
               fac5 = {coef20, coef20, coef22, coef23};

    const Vec4 vec0 = {rows[1].x, rows[0].x, rows[0].x, rows[0].x},
               vec1 = {rows[1].y, rows[0].y, rows[0].y, rows[0].y},
               vec2 = {rows[1].z, rows[0].z, rows[0].z, rows[0].z},
               vec3 = {rows[1].w, rows[0].w, rows[0].w, rows[0].w};

    const Vec4 inv0 = {vec1 * fac0 - vec2 * fac1 + vec3 * fac2},
               inv1 = {vec0 * fac0 - vec2 * fac3 + vec3 * fac4},
               inv2 = {vec0 * fac1 - vec1 * fac3 + vec3 * fac5},
               inv3 = {vec0 * fac2 - vec1 * fac4 + vec2 * fac5};

    const Vec4 sign_a = {+1.0f, -1.0f, +1.0f, -1.0f},
               sign_b = {-1.0f, +1.0f, -1.0f, +1.0f};
    const Mat4 inverse = {inv0 * sign_a, inv1 * sign_b, inv2 * sign_a,
                          inv3 * sign_b};

    const Vec4 row0 = {inverse[0].x, inverse[1].x, inverse[2].x, inverse[3].x};

    const Vec4 dot0 = {rows[0] * row0};
    const float dot1 = (dot0.x + dot0.y) + (dot0.z + dot0.w);

    const float one_over_determinant = 1.0f / dot1;

    return {
        one_over_determinant * inverse[0], one_over_determinant * inverse[1],
        one_over_determinant * inverse[2], one_over_determinant * inverse[3]};
}

QUALIFIER_D_H Mat4 operator+(const Mat4 &m0, const Mat4 &m1)
{
    return {m0.rows[0] + m1.rows[0], m0.rows[1] + m1.rows[1],
            m0.rows[2] + m1.rows[2], m0.rows[3] + m1.rows[3]};
}

QUALIFIER_D_H Mat4 operator-(const Mat4 &m0, const Mat4 &m1)
{
    return {m0.rows[0] - m1.rows[0], m0.rows[1] - m1.rows[1],
            m0.rows[2] - m1.rows[2], m0.rows[3] - m1.rows[3]};
}

QUALIFIER_D_H Mat4 Mul(const Mat4 &m0, const Mat4 &m1)
{
    return {Mul(m0[0], m1), Mul(m0[1], m1), Mul(m0[2], m1), Mul(m0[3], m1)};
}

QUALIFIER_D_H Vec4 Mul(const Mat4 &m, const Vec4 &vec)
{
    return {Dot(m[0], vec), Dot(m[1], vec), Dot(m[2], vec), Dot(m[3], vec)};
}

QUALIFIER_D_H Vec4 Mul(const Vec4 vec, const Mat4 &m)
{
    const Mat4 m_trans = m.Transpose();
    return {Dot(vec, m_trans[0]), Dot(vec, m_trans[1]), Dot(vec, m_trans[2]),
            Dot(vec, m_trans[3])};
}

QUALIFIER_D_H Mat4 operator*(const Mat4 &m, const float t)
{
    return {m.rows[0] * t, m.rows[1] * t, m.rows[2] * t, m.rows[3] * t};
}

QUALIFIER_D_H Mat4 operator/(const Mat4 &m, const float t)
{
    const float k = 1.0f / t;
    return {m.rows[0] * k, m.rows[1] * k, m.rows[2] * k, m.rows[3] * k};
}

QUALIFIER_D_H Mat4 operator*(const float t, const Mat4 &m)
{
    return {t * m.rows[0], t * m.rows[1], t * m.rows[2], t * m.rows[3]};
}

QUALIFIER_D_H Mat4 Translate(const Vec3 &vec)
{
    return {{1.0f, 0.0f, 0.0f, vec.x},
            {0.0f, 1.0f, 0.0f, vec.y},
            {0.0f, 0.0f, 1.0f, vec.z},
            {0.0f, 0.0f, 0.0f, 1.0f}};
}

QUALIFIER_D_H Mat4 Scale(const Vec3 &vec)
{
    return {{vec.x, 0.0f, 0.0f, 0.0f},
            {0.0f, vec.y, 0.0f, 0.0f},
            {0.0f, 0.0f, vec.z, 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f}};
}

QUALIFIER_D_H Mat4 Rotate(const float angle, Vec3 axis)
{
    const float cos_theta = cosf(angle), sin_theta = sinf(angle);

    axis = Normalize(axis);
    Vec3 temp = (1.0f - cos_theta) * axis;

    Mat4 rotate;
    rotate[0].x = cos_theta + temp.x * axis.x;
    rotate[0].y = temp.y * axis.x - sin_theta * axis.z;
    rotate[0].z = temp.z * axis.x + sin_theta * axis.y;

    rotate[1].x = temp.x * axis.y + sin_theta * axis.z;
    rotate[1].y = cos_theta + temp.y * axis.y;
    rotate[1].z = temp.z * axis.y - sin_theta * axis.x;

    rotate[2].x = temp.x * axis.z - sin_theta * axis.y;
    rotate[2].y = temp.y * axis.z + sin_theta * axis.x;
    rotate[2].z = cos_theta + temp.z * axis.z;
    return rotate;
}

QUALIFIER_D_H Mat4 LookAtLH(const Vec3 &eye, const Vec3 &target, Vec3 up)
{
    const Vec3 front = Normalize(target - eye),
               right = Normalize(Cross(up, front));
    up = Normalize(Cross(front, right));

    return {{right.x, right.y, right.z, -Dot(right, eye)},
            {up.x, up.y, up.z, -Dot(up, eye)},
            {front.x, front.y, front.z, -Dot(front, eye)},
            {0.0f, 0.0f, 0.0f, 1.0f}};
}

QUALIFIER_D_H Vec3 TransformPoint(const Mat4 &m, const Vec3 &p)
{
    return Mul(m, Vec4{p, 1.0f}).position();
}

QUALIFIER_D_H Vec3 TransformVector(const Mat4 &m, const Vec3 &v)
{
    return Mul(m, Vec4{v, 0.0f}).direction();
}

} // namespace rt