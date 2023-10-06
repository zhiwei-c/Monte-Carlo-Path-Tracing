#include "integrator.cuh"

#include "../bsdfs/bsdfs.cuh"
#include "../geometry/primitive.cuh"
#include "../geometry/instance.cuh"
#include "../utils/math.cuh"

QUALIFIER_DEVICE Integrator::Integrator(float *pixel_buffer, Texture **texture_buffer,
                                        Bsdf **bsdf_buffer, Primitive *primitive_buffer,
                                        Instance *instance_buffer, Accel *accel,
                                        uint64_t num_emitter, Emitter **emitter_buffer,
                                        uint64_t num_area_light, uint64_t *area_light_id_buffer,
                                        EnvMap *env_map, Sun *sun)
    : pixel_buffer_(pixel_buffer), texture_buffer_(texture_buffer), bsdf_buffer_(bsdf_buffer),
      primitive_buffer_(primitive_buffer), instance_buffer_(instance_buffer), accel_(accel),
      num_emitter_(num_emitter), emitter_buffer_(emitter_buffer), num_area_light_(num_area_light),
      area_light_id_buffer_(area_light_id_buffer), env_map_(env_map), sun_(sun)
{
}

QUALIFIER_DEVICE Vec3 Integrator::TraceRay(const Vec3 &eye, const Vec3 &look_dir,
                                           uint64_t *seed) const
{
    Vec3 L(0), attenuation(1), wo = -look_dir;

    // 求取原初光线与场景的交点
    Intersection its; // 光线与场景的交点
    if (!accel_->Empty())
        accel_->Intersect(Ray(eye, look_dir), bsdf_buffer_, texture_buffer_, pixel_buffer_, seed, &its);
    if (!its.valid())
    { // 溯源光线逃逸出场景, 计算来自外部环境的光照
        if (env_map_ != nullptr)
            L += env_map_->GetRadiance(look_dir, pixel_buffer_, texture_buffer_);
        if (sun_ != nullptr)
            L += sun_->GetRadianceDirect(look_dir, pixel_buffer_, texture_buffer_);
        return L;
    }
    else if (its.absorb())
    { // 溯源至景物的背面，而景物的背面吸收一切光照
        return L;
    }

    Bsdf **bsdf = bsdf_buffer_ + its.id_bsdf();
    if ((*bsdf)->HasEmission())
    { // 如果是原初光线溯源至光源，那么不存在由 shadow ray 贡献的直接光照，因此直接累积
        return (*bsdf)->GetRadiance(its.texcoord(), pixel_buffer_, texture_buffer_);
    }

    // 光线追踪主循环，根据俄罗斯轮盘算法判断是否继续溯源光线
    constexpr float pdf_rr = 0.8f; // 俄罗斯轮盘（russian roulette）算法的概率
    for (uint64_t depth = 0; RandomFloat(seed) < pdf_rr; ++depth)
    {
        // 按表面积进行抽样得到阴影光线，根据多重重要抽样（MIS，multiple importance sampling）合并
        const Vec3 L_dir = attenuation * EvaluateDirectLight(its, wo, seed); // 阴影光线贡献的直接光照
        L += L_dir;

        // 抽样次生光线光线
        SamplingRecord rec = its.Sample(wo, bsdf, pixel_buffer_, texture_buffer_, seed);
        if (!rec.valid)
            break;

        // 累积场景的反射率
        attenuation *= rec.attenuation / rec.pdf;
        if (fmaxf(fmaxf(attenuation.x, attenuation.y), attenuation.z) < kEpsilon)
            break;

        // 溯源光线
        its = Intersection();
        accel_->Intersect(Ray(rec.pos, -rec.wi), bsdf_buffer_, texture_buffer_, pixel_buffer_, seed, &its);
        if (!its.valid())
        { // 光线逃逸出场景
            if (env_map_ != nullptr)
            { // 计算来自外部环境的光照
                const Vec3 radiance = env_map_->GetRadiance(-rec.wi, pixel_buffer_, texture_buffer_);
                L += attenuation * radiance;
            }
            break;
        }
        else if (its.absorb())
        { // 溯源至景物的背面，而景物的背面吸收一切光照
            break;
        }

        bsdf = bsdf_buffer_ + its.id_bsdf();
        if ((*bsdf)->HasEmission())
        { // 如果溯源至光源，累积直接光照，并停止溯源
          // 因为次生光线是按 BSDF 进行重要抽样得到的，所以根据多重重要抽样（MIS，multiple importance sampling）合并
            const float cos_theta_prime = Dot(rec.wi, its.normal());
            if (cos_theta_prime < kEpsilon)
                break;
            const float pdf_area = PdfDirectLight(its, rec.wi, cos_theta_prime),
                        weight_bsdf = MisWeight(rec.pdf, pdf_area);
            // 场景中的面光源以类似漫反射的形式向各个方向均匀地发光
            const Vec3 radiance = (*bsdf)->GetRadiance(rec.texcoord, pixel_buffer_, texture_buffer_),
                       L_dir = weight_bsdf * attenuation * radiance;
            L += L_dir;
            break;
        }

        wo = rec.wi;
        // 根据俄罗斯轮盘赌算法的概率处理场景的反射率，使之符合应有的数学期望
        attenuation *= (1.0f / pdf_rr);
    }
    return L;
}

QUALIFIER_DEVICE Vec3 Integrator::EvaluateDirectLight(const Intersection &its, const Vec3 &wo,
                                                      uint64_t *seed) const
{
    Vec3 L(0.0f);
    if (num_emitter_ != 0)
    {
        // 随机抽样一个点光源或平行光源
        const uint64_t id_emitter = static_cast<uint64_t>(RandomFloat(seed) * num_emitter_);

        Vec3 wi, radiance;
        if (emitter_buffer_[id_emitter]->GetRadiance(its.pos(), accel_, bsdf_buffer_, texture_buffer_,
                                                     pixel_buffer_, seed, &radiance, &wi))
        {
            Bsdf **bsdf = bsdf_buffer_ + its.id_bsdf();
            const SamplingRecord rec = its.Evaluate(wi, wo, bsdf, pixel_buffer_, texture_buffer_,
                                                    seed);
            if (rec.pdf > kEpsilon)
                L += radiance * rec.attenuation;
        }
    }

    // 随机抽样一个面光源，按面积抽样面光源上一点
    if (num_area_light_ == 0)
        return L;
    const uint64_t id_area_light = static_cast<uint64_t>(RandomFloat(seed) * num_area_light_),
                   target = area_light_id_buffer_[id_area_light];
    Intersection its_pre;
    instance_buffer_[target].SamplePoint(primitive_buffer_, seed, &its_pre);

    // 抽样得到的面光源上一点与当前着色点之间不能被其它物体遮挡
    const Vec3 d_vec = its.pos() - its_pre.pos();
    Intersection its_test;
    accel_->Intersect(Ray(its_pre.pos(), Normalize(d_vec)), bsdf_buffer_, texture_buffer_, pixel_buffer_,
                      seed, &its_test);
    const float distance = Length(d_vec);
    if (its_test.distance() + kEpsilonDistance < distance)
        return L;
    const Vec3 wi = Normalize(its.pos() - its_pre.pos());
    const float cos_theta_prime = Dot(wi, its_pre.normal());
    if (cos_theta_prime < kEpsilon)
        return L;
    if (Dot(-wi, its.normal()) < kEpsilon)
        return L;

    // 根据多重重要抽样（MIS，multiple importance sampling）合并按表面积进行抽样得到的阴影光线贡献的直接光照
    Bsdf **bsdf = bsdf_buffer_ + its.id_bsdf();
    const SamplingRecord rec = its.Evaluate(wi, wo, bsdf, pixel_buffer_, texture_buffer_, seed);
    if (rec.pdf < kEpsilon)
        return L;

    const float pdf_area = its_pre.pdf_area() / num_area_light_,
                pdf_direct = pdf_area * (distance * distance) / cos_theta_prime,
                weight_direct = MisWeight(pdf_direct, rec.pdf);
    Bsdf **bsdf_pre = bsdf_buffer_ + its_pre.id_bsdf();
    // 场景中的面光源以类似漫反射的形式向各个方向均匀地发光
    const Vec3 radiance = (*bsdf_pre)->GetRadiance(rec.texcoord, pixel_buffer_, texture_buffer_);
    L += weight_direct * (radiance * rec.attenuation / pdf_direct);

    return L;
}

QUALIFIER_DEVICE float Integrator::PdfDirectLight(const Intersection &its_pre, const Vec3 &wi,
                                                  const float cos_theta_prime) const
{
    const float pdf_area = its_pre.pdf_area() / num_area_light_,
                distance_sqr_ = its_pre.distance() * its_pre.distance();
    return pdf_area * distance_sqr_ / cos_theta_prime;
}