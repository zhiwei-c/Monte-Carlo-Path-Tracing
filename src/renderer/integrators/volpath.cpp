#include "csrt/renderer/integrators/volpath.hpp"

#include "csrt/renderer/integrators/integrator.hpp"

namespace csrt
{

QUALIFIER_D_H Vec3 ShadeVolPath(const IntegratorData *data, const Vec3 &eye,
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

    Vec3 attenuation(1), wo = -look_dir;
    bool scattering = false;
    MediumHit medium_hit;

    //
    // 处理参与介质的影响
    //
    const bool inside = Dot(wo, hit.normal) > 0 ? hit.inside : !hit.inside;
    const uint32_t id_medium = inside ? hit.id_medium_int : hit.id_medium_ext;
    if (id_medium != kInvalidId)
    {
        Medium *medium = data->media + id_medium;
        MediumSampleRec medium_rec;
        medium->Sample(ray.t_max, seed, &medium_rec);
        if (medium_rec.valid)
        { //光线在参与介质中传播，存在明显衰减
            attenuation *= medium_rec.attenuation / medium_rec.pdf;
            if (medium_rec.scattered)
            { //光线在传播时发生了散射，实际上来源于更近的地方
                scattering = true;
                medium_hit.position =
                    ray.origin + ray.dir * medium_rec.distance;
                medium_hit.medium = medium;
            }
        }
    }

    Bsdf *bsdf = nullptr;
    if (!scattering)
    {
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
    }

    Vec3 wi = {};
    float pdf_sample = 0;
    for (uint32_t depth = 1;
         depth < data->info.depth_rr || (depth < data->info.depth_max &&
                                         RandomFloat(seed) < data->info.pdf_rr);
         ++depth)
    {
        if (scattering)
        { //当前散射点在参与介质之中
            // 按表面积进行抽样得到阴影光线，合并阴影光线贡献的直接光照
            L += attenuation * EvaluateDirectLightVolPath(data, medium_hit, wo, seed);

            // 抽样次生光线光线
            PhaseSampleRec phase_rec;
            phase_rec.wo = wo;
            medium_hit.medium->SamplePhase(seed, &phase_rec);
            if (!phase_rec.valid)
                break;
            wi = phase_rec.wi;

            // 累积场景的反射率
            attenuation *= phase_rec.attenuation / phase_rec.pdf;
            pdf_sample = phase_rec.pdf;
            if (fmaxf(fmaxf(attenuation.x, attenuation.y), attenuation.z) <
                kEpsilon)
                break;

            // 继续溯源光线
            ray = Ray(medium_hit.position, -wi);
            hit = data->tlas->Intersect(data->bsdfs, data->map_instance_bsdf,
                                        seed, &ray);

            // 处理参与介质的影响
            MediumSampleRec medium_rec;
            medium_hit.medium->Sample(ray.t_max, seed, &medium_rec);
            if (medium_rec.valid)
            { //光线在参与介质中传播，存在明显衰减
                attenuation *= medium_rec.attenuation / medium_rec.pdf;
                if (medium_rec.scattered)
                { //光线在传播时发生了散射，实际上来源于更近的地方
                    scattering = true;
                    medium_hit.position =
                        ray.origin + ray.dir * medium_rec.distance;
                }
                else
                { //光线在传播时，没有发生散射
                    scattering = false;
                }
            }
            else
            { //光线在传播时，没有和参与介质相互作用
                scattering = false;
            }
        }
        else
        { //当前散射点在景物表面
            // 按表面积进行抽样得到阴影光线，合并阴影光线贡献的直接光照
            L += attenuation * EvaluateDirectLightVolPath(data, hit, wo, seed);

            // 抽样次生光线光线
            BsdfSampleRec rec = SampleRayPath(wo, hit, bsdf, seed);
            if (!rec.valid)
                break;
            wi = rec.wi;
            pdf_sample = rec.pdf;

            // 累积场景的反射率
            attenuation *= rec.attenuation / pdf_sample;
            if (fmaxf(fmaxf(attenuation.x, attenuation.y), attenuation.z) <
                kEpsilon)
                break;

            // 继续溯源光线
            ray = Ray(rec.position, -wi);
            hit = data->tlas->Intersect(data->bsdfs, data->map_instance_bsdf,
                                        seed, &ray);

            // 处理参与介质的影响
            const bool inside =
                Dot(wi, hit.normal) > 0 ? hit.inside : !hit.inside;
            const uint32_t id_medium =
                inside ? hit.id_medium_int : hit.id_medium_ext;
            if (id_medium != kInvalidId)
            {
                Medium *medium = data->media + id_medium;
                MediumSampleRec medium_rec;
                medium->Sample(ray.t_max, seed, &medium_rec);
                if (medium_rec.valid)
                { //光线在参与介质中传播，存在明显衰减
                    attenuation *= medium_rec.attenuation / medium_rec.pdf;
                    if (medium_rec.scattered)
                    { //光线在传播时发生了散射，实际上来源于更近的地方
                        scattering = true;
                        medium_hit.position =
                            ray.origin + ray.dir * medium_rec.distance;
                        medium_hit.medium = medium;
                    }
                }
            }
        }

        if (!scattering)
        {
            if (!hit.valid)
            { // 次生光线逃逸出场景
                if (data->id_envmap != kInvalidId)
                {
                    const Vec3 radiance =
                        data->emitters[data->id_envmap].Evaluate(-wi);
                    const float pdf_direct =
                                    data->emitters[data->id_envmap].Pdf(-wi),
                                weight_bsdf = MisWeight(pdf_sample, pdf_direct);
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
                { // 次生光线溯源至光源，累积直接光照，并停止溯源
                    const float cos_theta_prime = Dot(wi, hit.normal);
                    if (cos_theta_prime < kEpsilonFloat)
                        break;
                    const uint32_t id_instance_area_light =
                        data->map_id_instance_area_light[hit.id_instance];
                    const float
                        pdf_area =
                            (data->cdf_area_light[id_instance_area_light + 1] -
                             data->cdf_area_light[id_instance_area_light]) *
                            data->list_pdf_area_instance[hit.id_instance],
                        pdf_direct =
                            pdf_area * Sqr(ray.t_max) / cos_theta_prime,
                        weight_bsdf = MisWeight(pdf_sample, pdf_direct);
                    // 场景中的面光源以类似漫反射的形式向各个方向均匀地发光
                    const Vec3 radiance = bsdf->GetRadiance(hit.texcoord),
                               L_dir = weight_bsdf * attenuation * radiance;
                    L += L_dir;
                    break;
                }
            }

            // 原初光线溯源至不发光的景物表面，被散射
            wo = wi;
            if (depth >= data->info.depth_rr)
            { // 根据俄罗斯轮盘赌算法的概率处理场景的反射率，使之符合应有的数学期望
                attenuation *= data->pdf_rr_rcp;
            }
        }
    }

    return L;
}

QUALIFIER_D_H Vec3 EvaluateDirectLightVolPath(const IntegratorData *data,
                                              const Hit &hit, const Vec3 &wo,
                                              uint32_t *seed)
{
    Vec3 L(0);

    const bool inside = Dot(wo, hit.normal) > 0 ? hit.inside : !hit.inside;
    const uint32_t id_medium = inside ? hit.id_medium_int : hit.id_medium_ext;
    Medium *medium = nullptr;
    if (id_medium != kInvalidId)
        medium = data->media + id_medium;

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

        Vec3 medium_attenuation = {1.0f};
        if (medium != nullptr)
        {
            MediumSampleRec medium_rec;
            medium_rec.distance = rec.distance;
            medium->Evaluate(&medium_rec);
            if (!medium_rec.valid)
                continue;
            medium_attenuation = medium_rec.attenuation / medium_rec.pdf;
        }

        Bsdf *bsdf = nullptr;
        if (data->map_instance_bsdf[hit.id_instance] != kInvalidId)
            bsdf = data->bsdfs + data->map_instance_bsdf[hit.id_instance];

        const BsdfSampleRec rec1 = EvaluateRayPath(rec.wi, wo, hit, bsdf);
        if (!rec1.valid)
            continue;

        const Vec3 radiance = emitter->Evaluate(rec);
        if (rec.harsh)
        {
            L += radiance * medium_attenuation * rec1.attenuation;
        }
        else
        {
            const float pdf_direct = emitter->Pdf(-rec.wi);
            if (pdf_direct > kEpsilonFloat)
            {
                const float weight_direct = MisWeight(pdf_direct, rec1.pdf);
                L += weight_direct * radiance * medium_attenuation *
                     rec1.attenuation / pdf_direct;
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

        Vec3 medium_attenuation = {1.0f};
        if (medium != nullptr)
        {
            MediumSampleRec medium_rec;
            medium_rec.distance = distance;
            medium->Evaluate(&medium_rec);
            if (!medium_rec.valid)
                return L;
            medium_attenuation = medium_rec.attenuation / medium_rec.pdf;
        }

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
        L += weight_direct *
             (radiance * medium_attenuation * rec.attenuation / pdf_direct);
    }

    return L;
}

QUALIFIER_D_H Vec3 EvaluateDirectLightVolPath(const IntegratorData *data,
                                              const MediumHit &hit,
                                              const Vec3 &wo, uint32_t *seed)
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

        MediumSampleRec medium_rec;
        medium_rec.distance = rec.distance;
        hit.medium->Evaluate(&medium_rec);
        if (!medium_rec.valid)
            continue;
        const Vec3 medium_attenuation = medium_rec.attenuation / medium_rec.pdf;

        PhaseSampleRec phase_rec;
        phase_rec.wi = rec.wi;
        phase_rec.wo = wo;
        hit.medium->EvaluatePhase(&phase_rec);
        if (!phase_rec.valid)
            continue;

        const Vec3 radiance = emitter->Evaluate(rec);
        if (rec.harsh)
        {
            L += radiance * medium_attenuation * phase_rec.attenuation;
        }
        else
        {
            const float pdf_direct = emitter->Pdf(-rec.wi);
            if (pdf_direct > kEpsilonFloat)
            {
                const float weight_direct =
                    MisWeight(pdf_direct, phase_rec.pdf);
                L += weight_direct * radiance * medium_attenuation *
                     phase_rec.attenuation / pdf_direct;
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

        MediumSampleRec medium_rec;
        medium_rec.distance = distance;
        hit.medium->Evaluate(&medium_rec);
        if (!medium_rec.valid)
            return L;
        const Vec3 medium_attenuation = medium_rec.attenuation / medium_rec.pdf;

        PhaseSampleRec phase_rec;
        phase_rec.wi = wi;
        phase_rec.wo = wo;
        hit.medium->EvaluatePhase(&phase_rec);
        if (!phase_rec.valid)
            return L;

        // 根据多重重要抽样（MIS，multiple importance sampling）合并按表面积进行抽样得到的阴影光线贡献的直接光照
        const float pdf_area =
                        (data->cdf_area_light[index_area_light + 1] -
                         data->cdf_area_light[index_area_light]) *
                        data->list_pdf_area_instance[id_area_light_instance],
                    pdf_direct = pdf_area * Sqr(distance) / cos_theta_prime,
                    weight_direct = MisWeight(pdf_direct, phase_rec.pdf);
        Bsdf *bsdf_pre =
            data->bsdfs + data->map_instance_bsdf[id_area_light_instance];
        const Vec3 radiance = bsdf_pre->GetRadiance(hit_pre.texcoord);
        L += weight_direct * (radiance * medium_attenuation *
                              phase_rec.attenuation / pdf_direct);
    }

    return L;
}

} // namespace csrt
