#include "csrt/renderer/integrators/path.hpp"

#include "csrt/renderer/integrators/integrator.hpp"

namespace csrt
{

QUALIFIER_D_H Vec3 ShadePath(const IntegratorData *data, const Vec3 &eye,
                             const Vec3 &look_dir, uint32_t *seed)
{
    Vec3 L(0);

    //
    // 求取原初光线与场景的交点
    //
    Ray ray = {eye, look_dir};
    Hit hit;
    if (data->tlas)
    {
        hit = data->tlas->Intersect(data->bsdfs, data->map_instance_bsdf, seed,
                                    &ray);
    }

    if (!hit.valid)
    { // 原初光线逃逸出场景
        if (data->id_envmap != kInvalidId)
        {
            L += data->emitters[data->id_envmap].Evaluate(look_dir);
        }
        if (data->id_sun != kInvalidId)
        {
            L += data->emitters[data->id_sun].Evaluate(look_dir);
        }
        return L;
    }

    Bsdf *bsdf = nullptr;
    if (data->map_instance_bsdf[hit.id_instance] != kInvalidId)
        bsdf = data->bsdfs + data->map_instance_bsdf[hit.id_instance];

    if (bsdf != nullptr)
    {
        if (hit.inside && !bsdf->IsTwosided())
        { // 原初光线溯源至景物的背面，且景物的背面吸收一切光照
            return {0};
        }
        else if (bsdf->IsEmitter())
        { // 原初光线溯源至光源，不存在由 shadow ray 贡献的直接光照，返回直接光照
            if (data->info.hide_emitters)
                return {0};
            else
                return bsdf->GetRadiance(hit.texcoord);
        }
    }

    Vec3 attenuation(1), wo = -look_dir;
    for (uint32_t depth = 1;
         depth < data->info.depth_rr || (depth < data->info.depth_max &&
                                         RandomFloat(seed) < data->info.pdf_rr);
         ++depth)
    {
        // 按表面积进行抽样得到阴影光线，合并阴影光线贡献的直接光照
        L += attenuation * EvaluateDirectLightPath(data, hit, wo, seed);

        // 抽样次生光线光线
        BsdfSampleRec rec = SampleRayPath(wo, hit, bsdf, seed);
        if (!rec.valid)
            break;

        // 累积场景的反射率
        attenuation *= rec.attenuation / rec.pdf;
        if (fmaxf(fmaxf(attenuation.x, attenuation.y), attenuation.z) <
            kEpsilon)
            break;

        // 溯源光线
        ray = Ray(rec.position, -rec.wi);
        hit = data->tlas->Intersect(data->bsdfs, data->map_instance_bsdf, seed,
                                    &ray);

        if (!hit.valid)
        { // 次生光线逃逸出场景
            if (data->id_envmap != kInvalidId)
            {
                const Vec3 radiance =
                    data->emitters[data->id_envmap].Evaluate(-rec.wi);
                const float pdf_direct =
                                data->emitters[data->id_envmap].Pdf(-rec.wi),
                            weight_bsdf = MisWeight(rec.pdf, pdf_direct);
                L += weight_bsdf * attenuation * radiance;
            }
            break;
        }

        bsdf = nullptr;
        if (data->map_instance_bsdf[hit.id_instance] != kInvalidId)
            bsdf = data->bsdfs + data->map_instance_bsdf[hit.id_instance];

        if (bsdf != nullptr)
        {
            if (hit.inside && !bsdf->IsTwosided())
            { // 次生光线溯源至景物的背面，且景物的背面吸收一切光照
                break;
            }
            else if (bsdf->IsEmitter())
            { // 原初光线溯源至光源，累积直接光照，并停止溯源
                const float cos_theta_prime = Dot(rec.wi, hit.normal);
                if (cos_theta_prime < kEpsilonFloat)
                    break;
                const uint32_t id_instance_area_light =
                    data->map_id_instance_area_light[hit.id_instance];
                const float
                    pdf_area =
                        (data->cdf_area_light[id_instance_area_light + 1] -
                         data->cdf_area_light[id_instance_area_light]) *
                        data->list_pdf_area_instance[hit.id_instance],
                    pdf_direct = pdf_area * Sqr(ray.t_max) / cos_theta_prime,
                    weight_bsdf = MisWeight(rec.pdf, pdf_direct);
                // 场景中的面光源以类似漫反射的形式向各个方向均匀地发光
                const Vec3 radiance = bsdf->GetRadiance(hit.texcoord),
                           L_dir = weight_bsdf * attenuation * radiance;
                L += L_dir;
                break;
            }
        }

        // 原初光线溯源至不发光的景物表面，被散射
        wo = rec.wi;
        if (depth >= data->info.depth_rr)
        { // 根据俄罗斯轮盘赌算法的概率处理场景的反射率，使之符合应有的数学期望
            attenuation *= data->pdf_rr_rcp;
        }
    }

    return L;
}

QUALIFIER_D_H Vec3 EvaluateDirectLightPath(const IntegratorData *data,
                                           const Hit &hit, const Vec3 &wo,
                                           uint32_t *seed)
{
    Vec3 L(0);

    for (uint32_t i = 0; i < data->num_emitter; ++i)
    {
        Emitter *emitter = data->emitters + i;

        EmitterSampleRec rec =
            emitter->Sample(hit.position, RandomFloat(seed), RandomFloat(seed));

        // 光源与当前着色点之间不能被其它物体遮挡
        Ray ray_test = {hit.position, -rec.wi};
        ray_test.t_max = rec.distance - kEpsilonDistance;
        if (data->tlas->IntersectAny(data->bsdfs, data->map_instance_bsdf, seed,
                                     &ray_test))
            continue;

        if (Dot(-rec.wi, hit.normal) < kEpsilonFloat)
            continue;

        Bsdf *bsdf = nullptr;
        if (data->map_instance_bsdf[hit.id_instance] != kInvalidId)
            bsdf = data->bsdfs + data->map_instance_bsdf[hit.id_instance];

        const BsdfSampleRec rec1 = EvaluateRayPath(rec.wi, wo, hit, bsdf);
        if (!rec1.valid)
            continue;

        const Vec3 radiance = emitter->Evaluate(rec);
        if (rec.harsh)
        {
            L += radiance * rec1.attenuation;
        }
        else
        {
            const float pdf_direct = emitter->Pdf(-rec.wi);
            if (pdf_direct > kEpsilonFloat)
            {
                const float weight_direct = MisWeight(pdf_direct, rec1.pdf);
                L += weight_direct * radiance * (rec1.attenuation / pdf_direct);
            }
        }
    }

    if (data->num_area_light != 0)
    {
        // 抽样得到的面光源上一点
        const uint32_t index_area_light =
                           BinarySearch(data->size_cdf_area_light,
                                        data->cdf_area_light,
                                        RandomFloat(seed)) -
                           1,
                       id_area_light_instance =
                           data->map_id_area_light_instance[index_area_light];
        const Hit hit_pre = data->instances[id_area_light_instance].Sample(
            RandomFloat(seed), RandomFloat(seed), RandomFloat(seed));

        // 抽样点与当前着色点之间不能被其它物体遮挡
        const Vec3 d_vec = hit.position - hit_pre.position;
        const float distance = Length(d_vec);
        Ray ray_test = {hit_pre.position, Normalize(d_vec)};
        ray_test.t_max = distance - kEpsilonDistance;
        if (data->tlas->IntersectAny(data->bsdfs, data->map_instance_bsdf, seed,
                                     &ray_test))
            return L;

        const Vec3 wi = Normalize(d_vec);
        const float cos_theta_prime = Dot(wi, hit_pre.normal);
        if (cos_theta_prime < kEpsilonFloat)
            return L;
        if (Dot(-wi, hit.normal) < kEpsilonFloat)
            return L;

        Bsdf *bsdf = nullptr;
        if (data->map_instance_bsdf[hit.id_instance] != kInvalidId)
            bsdf = data->bsdfs + data->map_instance_bsdf[hit.id_instance];

        const BsdfSampleRec rec = EvaluateRayPath(wi, wo, hit, bsdf);
        if (!rec.valid)
            return L;

        // 根据多重重要抽样（MIS，multiple importance sampling）合并按表面积进行抽样得到的阴影光线贡献的直接光照
        const float pdf_area =
                        (data->cdf_area_light[index_area_light + 1] -
                         data->cdf_area_light[index_area_light]) *
                        data->list_pdf_area_instance[id_area_light_instance],
                    pdf_direct = pdf_area * Sqr(distance) / cos_theta_prime,
                    weight_direct = MisWeight(pdf_direct, rec.pdf);
        Bsdf *bsdf_pre =
            data->bsdfs + data->map_instance_bsdf[id_area_light_instance];
        const Vec3 radiance = bsdf_pre->GetRadiance(hit_pre.texcoord);
        L += weight_direct * radiance * (rec.attenuation / pdf_direct);
    }

    return L;
}

QUALIFIER_D_H BsdfSampleRec EvaluateRayPath(const Vec3 &wi, const Vec3 &wo,
                                            const Hit &hit, Bsdf *bsdf)
{
    BsdfSampleRec rec;
    rec.wi = wi;
    rec.wo = wo;
    rec.texcoord = hit.texcoord;
    rec.position = hit.position;
    if (bsdf)
    {
        rec.inside = hit.inside;
        rec.normal = hit.normal;
        rec.tangent = hit.tangent;
        rec.bitangent = hit.bitangent;
        if (Dot(-wi, hit.normal) < 0.0f)
        {
            rec.inside = !rec.inside;
            rec.normal = -rec.normal;
        }
        bsdf->Evaluate(&rec);
    }
    else
    {
        rec.pdf = 1;
        rec.attenuation = Vec3(1);
        rec.valid = true;
    }
    return rec;
}

QUALIFIER_D_H BsdfSampleRec SampleRayPath(const Vec3 &wo, const Hit &hit,
                                          Bsdf *bsdf, uint32_t *seed)
{
    BsdfSampleRec rec;
    rec.wo = wo;
    rec.texcoord = hit.texcoord;
    rec.position = hit.position;
    if (bsdf != nullptr)
    {
        rec.inside = hit.inside;
        rec.normal = hit.normal;
        rec.tangent = hit.tangent;
        rec.bitangent = hit.bitangent;
        if (Dot(wo, hit.normal) < 0.0f)
        {
            rec.inside = !rec.inside;
            rec.normal = -rec.normal;
        }
        bsdf->Sample(seed, &rec);
    }
    else
    {
        rec.wi = wo;
        rec.pdf = 1.0f;
        rec.attenuation = Vec3(1.0f);
        rec.valid = true;
    }
    return rec;
}

} // namespace csrt
