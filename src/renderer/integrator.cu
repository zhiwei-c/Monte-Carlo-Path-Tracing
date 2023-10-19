#include "integrator.cuh"

#include "../bsdfs/bsdfs.cuh"
#include "../geometry/primitive.cuh"
#include "../geometry/instance.cuh"
#include "../utils/math.cuh"

QUALIFIER_DEVICE Integrator::Integrator(float *pixel_buffer, Texture **texture_buffer,
                                        Bsdf **bsdf_buffer, Primitive *primitive_buffer,
                                        Instance *instance_buffer, Accel *accel,
                                        uint32_t num_emitter, Emitter **emitter_buffer,
                                        uint32_t num_area_light, uint32_t *area_light_id_buffer,
                                        EnvMap *env_map, Sun *sun)
    : pixel_buffer_(pixel_buffer), texture_buffer_(texture_buffer), bsdf_buffer_(bsdf_buffer),
      primitive_buffer_(primitive_buffer), instance_buffer_(instance_buffer), accel_(accel),
      num_emitter_(num_emitter), emitter_buffer_(emitter_buffer), num_area_light_(num_area_light),
      area_light_id_buffer_(area_light_id_buffer), env_map_(env_map), sun_(sun)
{
}

QUALIFIER_DEVICE Vec3 Integrator::GenerateRay(const Vec3 &eye, const Vec3 &look_dir,
                                              uint32_t *seed) const
{
    Vec3 L(0), attenuation(1), wo = -look_dir;

    // 求取原初光线与场景的交点
    Intersection its; // 光线与场景的交点
    if (!accel_->Empty())
        its = accel_->TraceRay(Ray(eye, look_dir), bsdf_buffer_, texture_buffer_, pixel_buffer_,
                               seed);
    if (!its.valid)
    { // 溯源光线逃逸出场景, 计算来自外部环境的光照
        {
            if (env_map_ != nullptr)
                L += env_map_->GetRadiance(look_dir, pixel_buffer_, texture_buffer_);
            if (sun_ != nullptr)
                L += sun_->GetRadianceDirect(look_dir, pixel_buffer_, texture_buffer_);
        }
        return L;
    }
    else if (its.absorb)
    { // 溯源至景物的背面，而景物的背面吸收一切光照
        return L;
    }

    Bsdf **bsdf = bsdf_buffer_ + its.id_bsdf;
    if ((*bsdf)->HasEmission())
    { // 如果是原初光线溯源至光源，那么不存在由 shadow ray 贡献的直接光照，因此直接累积
        return (*bsdf)->GetRadiance(its.texcoord, texture_buffer_, pixel_buffer_);
    }

    // 光线追踪主循环，根据俄罗斯轮盘算法判断是否继续溯源光线
    constexpr float pdf_rr = 0.9f; // 俄罗斯轮盘（russian roulette）算法的概率
    for (uint32_t depth = 0; RandomFloat(seed) < pdf_rr; ++depth)
    {
        // 按表面积进行抽样得到阴影光线，合并阴影光线贡献的直接光照
        L += attenuation * EvaluateDirectAreaLight(its, wo, seed);
        L += attenuation * EvaluateDirectOtherLight(its, wo, seed);

        // 抽样次生光线光线
        SamplingRecord rec = SampleRay(wo, its, bsdf, seed);
        if (!rec.valid)
            break;

        // 累积场景的反射率
        attenuation *= rec.attenuation / rec.pdf;
        if (fmaxf(fmaxf(attenuation.x, attenuation.y), attenuation.z) < kEpsilon)
            break;

        // 溯源光线
        its = accel_->TraceRay(Ray(rec.position, -rec.wi), bsdf_buffer_, texture_buffer_,
                               pixel_buffer_, seed);
        if (!its.valid)
        { // 光线逃逸出场景
            if (env_map_ != nullptr)
            { // 计算来自外部环境的光照
                const Vec3 radiance = env_map_->GetRadiance(-rec.wi, pixel_buffer_, texture_buffer_);
                L += attenuation * radiance;
            }
            break;
        }
        else if (its.absorb)
        { // 溯源至景物的背面，而景物的背面吸收一切光照
            break;
        }

        bsdf = bsdf_buffer_ + its.id_bsdf;
        if ((*bsdf)->HasEmission())
        { // 如果溯源至光源，累积直接光照，并停止溯源。
          // 因为次生光线是按 BSDF 进行重要抽样得到的，
          // 所以根据多重重要抽样（MIS，multiple importance sampling）合并。
            const float cos_theta_prime = Dot(rec.wi, its.normal);
            if (cos_theta_prime < kEpsilon)
                break;
            const float pdf_area = PdfDirectLight(its, rec.wi, cos_theta_prime),
                        weight_bsdf = MisWeight(rec.pdf, pdf_area);
            // 场景中的面光源以类似漫反射的形式向各个方向均匀地发光
            const Vec3 radiance = (*bsdf)->GetRadiance(rec.texcoord, texture_buffer_, pixel_buffer_),
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

QUALIFIER_DEVICE SamplingRecord Integrator::SampleRay(const Vec3 &wo, const Intersection &its,
                                                      Bsdf **bsdf, uint32_t *seed) const
{
    SamplingRecord rec;
    rec.wo = wo;
    rec.texcoord = its.texcoord;
    rec.position = its.position;
    if (bsdf != nullptr)
    {
        rec.inside = its.inside;
        rec.normal = its.normal;
        rec.tangent = its.tangent;
        rec.bitangent = its.bitangent;
        if (Dot(wo, its.normal) < 0.0f)
        {
            rec.inside = !its.inside;
            rec.normal = -rec.normal;
            rec.tangent = -rec.tangent;
            rec.bitangent = -rec.bitangent;
        }
        (*bsdf)->Sample(texture_buffer_, pixel_buffer_, seed, &rec);
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

QUALIFIER_DEVICE SamplingRecord Integrator::EvaluateRay(const Vec3 &wi, const Vec3 &wo,
                                                        const Intersection &its, Bsdf **bsdf,
                                                        uint32_t *seed) const
{
    SamplingRecord rec;
    rec.wi = wi;
    rec.wo = wo;
    rec.texcoord = its.texcoord;
    rec.position = its.position;
    if (bsdf)
    {
        rec.inside = its.inside;
        rec.normal = its.normal;
        rec.tangent = its.tangent;
        rec.bitangent = its.bitangent;
        if (Dot(-wi, its.normal) < 0.0f)
        {
            rec.inside = !its.inside;
            rec.normal = -rec.normal;
            rec.tangent = -rec.tangent;
            rec.bitangent = -rec.bitangent;
        }
        (*bsdf)->Evaluate(texture_buffer_, pixel_buffer_, seed, &rec);
    }
    else
    {
        rec.pdf = 1;
        rec.attenuation = Vec3(1);
        rec.valid = true;
    }
    return rec;
}

QUALIFIER_DEVICE Vec3 Integrator::EvaluateDirectAreaLight(const Intersection &its, const Vec3 &wo,
                                                          uint32_t *seed) const
{
    // 随机抽样一个面光源，按面积抽样面光源上一点
    if (num_area_light_ == 0)
        return Vec3(0);

    const uint32_t id_area_light = static_cast<uint32_t>(RandomFloat(seed) * num_area_light_),
                   target = area_light_id_buffer_[id_area_light];
    Intersection its_pre;
    instance_buffer_[target].SamplePoint(primitive_buffer_, seed, &its_pre);

    // 抽样得到的面光源上一点与当前着色点之间不能被其它物体遮挡
    const Vec3 d_vec = its.position - its_pre.position;
    Intersection its_test = accel_->TraceRay(Ray(its_pre.position, Normalize(d_vec)), bsdf_buffer_,
                                             texture_buffer_, pixel_buffer_, seed);
    const float distance = Length(d_vec);
    if (its_test.distance + kEpsilonDistance < distance)
        return Vec3(0);
    const Vec3 wi = Normalize(its.position - its_pre.position);
    const float cos_theta_prime = Dot(wi, its_pre.normal);
    if (cos_theta_prime < kEpsilon)
        return Vec3(0);
    if (Dot(-wi, its.normal) < kEpsilon)
        return Vec3(0);

    // 根据多重重要抽样（MIS，multiple importance sampling）合并按表面积进行抽样得到的阴影光线贡献的直接光照
    Bsdf **bsdf = bsdf_buffer_ + its.id_bsdf;
    const SamplingRecord rec = EvaluateRay(wi, wo, its, bsdf, seed);
    if (!rec.valid)
        return Vec3(0);

    const float pdf_area = its_pre.pdf_area / num_area_light_,
                pdf_direct = pdf_area * (distance * distance) / cos_theta_prime,
                weight_direct = MisWeight(pdf_direct, rec.pdf);
    Bsdf **bsdf_pre = bsdf_buffer_ + its_pre.id_bsdf;
    // 场景中的面光源以类似漫反射的形式向各个方向均匀地发光
    const Vec3 radiance = (*bsdf_pre)->GetRadiance(rec.texcoord, texture_buffer_, pixel_buffer_);
    return weight_direct * (radiance * rec.attenuation / pdf_direct);
}

QUALIFIER_DEVICE Vec3 Integrator::EvaluateDirectOtherLight(const Intersection &its, const Vec3 &wo,
                                                           uint32_t *seed) const
{
    if (num_emitter_ == 0)
        return Vec3(0);

    // 随机抽样一个点光源或平行光源
    const uint32_t id_emitter = static_cast<uint32_t>(RandomFloat(seed) * num_emitter_);

    Vec3 wi, radiance;
    if (!emitter_buffer_[id_emitter]->GetRadiance(its.position, accel_, bsdf_buffer_,
                                                  texture_buffer_, pixel_buffer_, seed,
                                                  &radiance, &wi))
        return Vec3(0);

    Bsdf **bsdf = bsdf_buffer_ + its.id_bsdf;
    const SamplingRecord rec = EvaluateRay(wi, wo, its, bsdf, seed);
    if (!rec.valid)
        return Vec3(0);
    else
        return radiance * rec.attenuation;
}

QUALIFIER_DEVICE float Integrator::PdfDirectLight(const Intersection &its_pre, const Vec3 &wi,
                                                  const float cos_theta_prime) const
{
    const float pdf_area = its_pre.pdf_area / num_area_light_,
                distance_sqr_ = its_pre.distance * its_pre.distance;
    return pdf_area * distance_sqr_ / cos_theta_prime;
}