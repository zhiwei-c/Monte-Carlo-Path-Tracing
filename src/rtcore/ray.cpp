#include "csrt/rtcore/ray.hpp"

#include "csrt/utils.hpp"

namespace csrt
{

QUALIFIER_D_H Ray::Ray()
    : origin{}, dir{0, 1, 0}, t_min(kEpsilonDistance), t_max(kMaxFloat),
      dir_rcp{kMaxFloat, 1, kMaxFloat}
{
#ifdef WATERTIGHT_TRIANGLES
    k[2] = 1, k[0] = 2, k[1] = 0;
    shear = {0, 0, 1};
#endif
}

QUALIFIER_D_H Ray::Ray(const Vec3 &_origin, const Vec3 &_dir)
    : origin(_origin), dir(_dir), t_min(kEpsilonDistance), t_max(kMaxFloat)
{
    for (int i = 0; i < 3; ++i)
        dir_rcp[i] = 1.0f / (dir[i] != 0 ? dir[i] : kEpsilonDistance);

#ifdef WATERTIGHT_TRIANGLES
    // 计算光线方向的坐标绝对值最大的维度
    k[2] = (fabs(dir.x) > fabs(dir.y) && fabs(dir.x) > fabs(dir.z))
               ? 0
               : (fabs(dir.y) > fabs(dir.z) ? 1 : 2);
    k[0] = k[2] + 1;
    if (k[0] == 3)
        k[0] = 0;
    k[1] = k[0] + 1;
    if (k[1] == 3)
        k[1] = 0;

    // 保证三角形顶点的环绕顺序一致
    if (dir[k[2]] < 0.0f)
    {
        int temp = k[0];
        k[0] = k[1];
        k[1] = temp;
    }

    // 计算剪切变换
    shear = {dir[k[0]] / dir[k[2]], dir[k[1]] / dir[k[2]], 1.0f / dir[k[2]]};
#endif
}

QUALIFIER_D_H Vec3 Ray::Reflect(const Vec3 &wi, const Vec3 &normal)
{
    return Normalize(wi - 2.0f * Dot(wi, normal) * normal);
}

QUALIFIER_D_H bool Ray::Refract(const Vec3 &wi, const Vec3 &normal,
                                const float eta_inv, Vec3 *wt)
{
    const float cos_theta = fabs(Dot(wi, normal));
    const float k = 1.0f - Sqr(eta_inv) * (1.0f - Sqr(cos_theta));
    if (k < 0)
    {
        return false;
    }
    else
    {
        *wt = Normalize(
            (eta_inv * wi + (eta_inv * cos_theta - sqrtf(k)) * normal));
        return true;
    }
}

} // namespace csrt