#include "mat4.cuh"

QUALIFIER_DEVICE Mat4 Mat4::Inverse() const
{
    const float coef00 = value[2][2] * value[3][3] - value[3][2] * value[2][3],
                coef02 = value[1][2] * value[3][3] - value[3][2] * value[1][3],
                coef03 = value[1][2] * value[2][3] - value[2][2] * value[1][3];

    const float coef04 = value[2][1] * value[3][3] - value[3][1] * value[2][3],
                coef06 = value[1][1] * value[3][3] - value[3][1] * value[1][3],
                coef07 = value[1][1] * value[2][3] - value[2][1] * value[1][3];

    const float coef08 = value[2][1] * value[3][2] - value[3][1] * value[2][2],
                coef10 = value[1][1] * value[3][2] - value[3][1] * value[1][2],
                coef11 = value[1][1] * value[2][2] - value[2][1] * value[1][2];

    const float coef12 = value[2][0] * value[3][3] - value[3][0] * value[2][3],
                coef14 = value[1][0] * value[3][3] - value[3][0] * value[1][3],
                coef15 = value[1][0] * value[2][3] - value[2][0] * value[1][3];

    const float coef16 = value[2][0] * value[3][2] - value[3][0] * value[2][2],
                coef18 = value[1][0] * value[3][2] - value[3][0] * value[1][2],
                coef19 = value[1][0] * value[2][2] - value[2][0] * value[1][2];

    const float coef20 = value[2][0] * value[3][1] - value[3][0] * value[2][1],
                coef22 = value[1][0] * value[3][1] - value[3][0] * value[1][1],
                coef23 = value[1][0] * value[2][1] - value[2][0] * value[1][1];

    const Vec4 fac0 = {coef00, coef00, coef02, coef03},
               fac1 = {coef04, coef04, coef06, coef07},
               fac2 = {coef08, coef08, coef10, coef11},
               fac3 = {coef12, coef12, coef14, coef15},
               fac4 = {coef16, coef16, coef18, coef19},
               fac5 = {coef20, coef20, coef22, coef23};

    const Vec4 vec0 = {value[1][0], value[0][0], value[0][0], value[0][0]},
               vec1 = {value[1][1], value[0][1], value[0][1], value[0][1]},
               vec2 = {value[1][2], value[0][2], value[0][2], value[0][2]},
               vec3 = {value[1][3], value[0][3], value[0][3], value[0][3]};

    const Vec4 inv0 = {vec1 * fac0 - vec2 * fac1 + vec3 * fac2},
               inv1 = {vec0 * fac0 - vec2 * fac3 + vec3 * fac4},
               inv2 = {vec0 * fac1 - vec1 * fac3 + vec3 * fac5},
               inv3 = {vec0 * fac2 - vec1 * fac4 + vec2 * fac5};

    const Vec4 sign_a = {+1.0f, -1.0f, +1.0f, -1.0f},
               sign_b = {-1.0f, +1.0f, -1.0f, +1.0f};
    const Mat4 inverse = {inv0 * sign_a, inv1 * sign_b, inv2 * sign_a, inv3 * sign_b};

    const Vec4 row0 = {inverse[0][0], inverse[1][0], inverse[2][0], inverse[3][0]};

    const Vec4 dot0 = {value[0] * row0};
    const float dot1 = (dot0.x + dot0.y) + (dot0.z + dot0.w);

    const float one_over_determinant = 1.0f / dot1;

    return Mat4(one_over_determinant * inverse[0],
                one_over_determinant * inverse[1],
                one_over_determinant * inverse[2],
                one_over_determinant * inverse[3]);
}

QUALIFIER_DEVICE Mat4 Rotate(float angle, Vec3 axis)
{
    const float a = angle,
                c = cosf(a),
                s = sinf(a);

    axis = Normalize(axis);
    Vec3 temp = (1.0f - c) * axis;

    Mat4 rotate;
    rotate[0][0] = c + temp[0] * axis[0];
    rotate[0][1] = temp[1] * axis[0] - s * axis[2];
    rotate[0][2] = temp[2] * axis[0] + s * axis[1];

    rotate[1][0] = temp[0] * axis[1] + s * axis[2];
    rotate[1][1] = c + temp[1] * axis[1];
    rotate[1][2] = temp[2] * axis[1] - s * axis[0];

    rotate[2][0] = temp[0] * axis[2] - s * axis[1];
    rotate[2][1] = temp[1] * axis[2] + s * axis[0];
    rotate[2][2] = c + temp[2] * axis[2];
    return rotate;
}

QUALIFIER_DEVICE Mat4 LookAtLH(const Vec3 &eye, const Vec3 &look_at, Vec3 up)
{
    const Vec3 front = Normalize(look_at - eye),
               right = Normalize(Cross(up, front));
    up = Normalize(Cross(front, right));

    Mat4 result;
    result[0][0] = right.x,
    result[0][1] = right.y,
    result[0][2] = right.z;
    result[0][3] = -Dot(right, eye);

    result[1][0] = up.x,
    result[1][1] = up.y,
    result[1][2] = up.z;
    result[1][3] = -Dot(up, eye);

    result[2][0] = front.x,
    result[2][1] = front.y,
    result[2][2] = front.z;
    result[2][3] = -Dot(front, eye);

    return result;
}
