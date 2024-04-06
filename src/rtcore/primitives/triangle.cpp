#include "csrt/rtcore/primitives/triangle.hpp"

#include "csrt/renderer/bsdfs/bsdf.hpp"
#include "csrt/utils.hpp"

namespace csrt
{

QUALIFIER_D_H AABB GetAabbTriangle(const TriangleData &data)
{
    AABB aabb;
    for (int i = 0; i < 3; ++i)
        aabb += data.positions[i];
    return aabb;
}

/// \brief Woop's watertight intersection algorithm or Möller–Trumbore
/// intersection algorithm
QUALIFIER_D_H bool IntersectTriangle(const uint32_t id_primitive,
                                     const TriangleData &data, Bsdf *bsdf,
                                     uint32_t *seed, Ray *ray, Hit *hit)
{
#ifdef WATERTIGHT_TRIANGLES
    //
    // Woop's watertight intersection algorithm
    //

    // 计算三角形顶点坐标相对于光线起点的位置
    const Vec3 A = data.positions[0] - ray->origin;
    const Vec3 B = data.positions[1] - ray->origin;
    const Vec3 C = data.positions[2] - ray->origin;

    // 对三角形顶点施加剪切变换和放缩变换，
    // 变换后光线起点位于原点，方向朝z轴正向
    const float Ax = A[ray->k[0]] - ray->shear.x * A[ray->k[2]];
    const float Ay = A[ray->k[1]] - ray->shear.y * A[ray->k[2]];
    const float Bx = B[ray->k[0]] - ray->shear.x * B[ray->k[2]];
    const float By = B[ray->k[1]] - ray->shear.y * B[ray->k[2]];
    const float Cx = C[ray->k[0]] - ray->shear.x * C[ray->k[2]];
    const float Cy = C[ray->k[1]] - ray->shear.y * C[ray->k[2]];

    // 计算未归一化的重心坐标
    float U = Cx * By - Cy * Bx;
    float V = Ax * Cy - Ay * Cx;
    float W = Bx * Ay - By * Ax;

    // 对于三角形边界上的特殊情况，使用双精度浮点数重新计算，
    // 因为 IEEE 754 浮点数标准可以保证两个浮点数舍入前后的大小次序不变，
    // 但是对于两组浮点数的乘积是否相等没有精确的保证
    if (U == 0.0f || V == 0.0f || W == 0.0f)
    {
        double CxBy = (double)Cx * (double)By;
        double CyBx = (double)Cy * (double)Bx;
        U = (float)(CxBy - CyBx);

        double AxCy = (double)Ax * (double)Cy;
        double AyCx = (double)Ay * (double)Cx;
        V = (float)(AxCy - AyCx);

        double BxAy = (double)Bx * (double)Ay;
        double ByAx = (double)By * (double)Ax;
        W = (float)(BxAy - ByAx);
    }

    // 进行边界测试
    if ((U < 0.0f || V < 0.0f || W < 0.0f) &&
        (U > 0.0f || V > 0.0f || W > 0.0f))
        return false;

    // 计算行列式
    const float det = U + V + W;
    if (det == 0.0f)
        return false;

    // 计算未归一化的z坐标，并以此计算交点距离
    const float Az = ray->shear.z * A[ray->k[2]];
    const float Bz = ray->shear.z * B[ray->k[2]];
    const float Cz = ray->shear.z * C[ray->k[2]];
    const float T = U * Az + V * Bz + W * Cz;

    const float det_inv = 1.0f / det;
    const float t = T * det_inv;
    if (t > ray->t_max || t < ray->t_min)
        return false;

    // 计算归一化的重心坐标 U，V，W 和距离 T
    const float u = U * det_inv, v = V * det_inv, w = W * det_inv;
#else
    //
    // Möller–Trumbore intersection algorithm
    //

    const Vec3 v0v1 = data.positions[1] - data.positions[0],
               v0v2 = data.positions[2] - data.positions[0];

    const Vec3 P = Cross(ray->dir, v0v2);
    const float det_inv = 1.0f / Dot(v0v1, P);

    const Vec3 T = ray->origin - data.positions[0];
    const float v = Dot(T, P) * det_inv;
    if (v < 0.0f || v > 1.0f)
        return false;

    const Vec3 Q = Cross(T, v0v1);
    const float w = Dot(ray->dir, Q) * det_inv;
    if (w < 0.0f || (v + w) > 1.0f)
        return false;

    const float t = Dot(v0v2, Q) * det_inv;
    if (t > ray->t_max || t < ray->t_min)
        return false;

    const float u = 1.0f - v - w;
#endif

    const Vec2 texcoord = Lerp(data.texcoords, u, v, w);
    if (bsdf != nullptr && bsdf->IsTransparent(texcoord, seed))
        return false;

    ray->t_max = t;

    if (hit != nullptr)
    {
        const bool inside = det_inv < 0;
        const Vec3 position = Lerp(data.positions, u, v, w);
        Vec3 normal = Normalize(Lerp(data.normals, u, v, w)),
             tangent = Normalize(Lerp(data.tangents, u, v, w)),
             bitangent = Normalize(Lerp(data.bitangents, u, v, w));
        if (bsdf != nullptr)
        {
            normal =
                bsdf->ApplyBumpMapping(normal, tangent, bitangent, texcoord);
            bitangent = Normalize(Cross(normal, tangent));
            tangent = Normalize(Cross(bitangent, normal));
        }

        if (inside)
        {
            normal = -normal;
            bitangent = -bitangent;
        }

        *hit = Hit(id_primitive, inside, texcoord, position, normal, tangent,
                   bitangent);
    }

    return true;
}

QUALIFIER_D_H Hit SampleTriangle(const uint32_t id_primitive,
                                 const TriangleData &data, const float xi_0,
                                 const float xi_1)
{
    const float temp = sqrtf(1.0f - xi_0);
    const float u = 1.0f - temp, v = temp * xi_1, w = 1.0f - u - v;
    const Vec2 texcoord = Lerp(data.texcoords, w, u, v);
    const Vec3 position = Lerp(data.positions, w, u, v),
               normal = Normalize(Lerp(data.normals, w, u, v));
    return Hit(id_primitive, texcoord, position, normal);
}

} // namespace csrt