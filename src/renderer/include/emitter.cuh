#pragma once

#include "rtcore.cuh"
#include "tensor.cuh"
#include "texture.cuh"
#include "utils.cuh"

namespace csrt
{

class Emitter
{
public:
    enum class Type
    {
        kNone,
        kPoint,
        kSpot,
        kDirectional,
        kSun,
        kEnvMap,
        kConstant,
    };

    struct Data
    {
        struct Point
        {
            Vec3 position = {};
            Vec3 intensity = {};
        };

        struct Spot
        {
            float cutoff_angle = 0;
            float cos_cutoff_angle = 0;
            float uv_factor = 0;
            float beam_width = 0;
            float cos_beam_width = 0;
            float transition_width_rcp = 0;
            Texture *texture = nullptr;
            Vec3 intensity = {};
            Vec3 position = {};
            Mat4 to_local = {};
        };

        struct Directional
        {
            Vec3 direction = {};
            Vec3 radiance = {};
        };

        struct Sun
        {
            float cos_cutoff_angle = 0;
            Texture *texture = nullptr;
            Vec3 direction = {};
            Vec3 radiance = {};
        };

        struct EnvMap
        {
            int width = 0;
            int height = 0;
            float normalization = 0;
            Texture *radiance = nullptr;
            float *cdf_cols;    // 像素列的累积分布函数
            float *cdf_rows;    // 像素行的累积分布函数
            float *weight_rows; // 像素行的权重
            Mat4 to_world = {};
            Mat4 to_local = {};
        };

        struct Constant
        {
            Vec3 radiance = {};
        };

        Type type;
        union
        {
            Point point;
            Spot spot;
            Directional directional;
            Sun sun;
            EnvMap envmap;
            Constant constant;
        };

        QUALIFIER_D_H Data();
        QUALIFIER_D_H ~Data() {}
        QUALIFIER_D_H Data(const Emitter::Data &info);
        QUALIFIER_D_H void operator=(const Emitter::Data &info);
    };

    struct Info
    {
        struct Spot
        {
            float cutoff_angle = 0;
            float beam_width = 0;
            uint32_t id_texture = kInvalidId;
            Vec3 intensity = {};
            Mat4 to_world = {};
        };

        struct Sun
        {
            float cos_cutoff_angle = 0;
            uint32_t id_texture = kInvalidId;
            Vec3 direction = {};
            Vec3 radiance = {};
        };

        struct EnvMap
        {
            uint32_t id_radiance = kInvalidId;
            Mat4 to_world = {};
        };

        Type type;
        union
        {
            Emitter::Data::Point point;
            Spot spot;
            Emitter::Data::Directional directional;
            Sun sun;
            EnvMap envmap;
            Emitter::Data::Constant constant;
        };

        QUALIFIER_D_H Info();
        QUALIFIER_D_H ~Info() {}
        QUALIFIER_D_H Info(const Info &info);
        QUALIFIER_D_H void operator=(const Info &info);
    };

    struct SampleRec
    {
        bool valid = false;
        bool harsh = true;
        float distance = kMaxFloat;
        Vec3 wi = {};
    };

    QUALIFIER_D_H Emitter();
    QUALIFIER_D_H Emitter(const uint32_t id, const Emitter::Info &info,
                          TLAS *tlas, Texture *texture_buffer);

    QUALIFIER_D_H void InitEnvMap(const int width, const int height,
                                  const float normalization, float *data);

    QUALIFIER_D_H Emitter::SampleRec
    Sample(const Vec3 &origin, const float xi_0, const float xi_1) const;
    QUALIFIER_D_H Vec3 Evaluate(const SampleRec &rec) const;
    QUALIFIER_D_H float Pdf(const Vec3 &look_dir) const;
    QUALIFIER_D_H Vec3 Evaluate(const Vec3 &look_dir) const;

    static void CreateEnvMapCdfPdf(const int width, const int height,
                                   const Texture &radiance,
                                   std::vector<float> *cdf_cols,
                                   std::vector<float> *cdf_rows,
                                   std::vector<float> *weight_rows,
                                   float *normalization);

private:
    QUALIFIER_D_H Emitter::SampleRec
    SamplePoint(const Vec3 &origin, const float xi_0, const float xi_1) const;
    QUALIFIER_D_H Emitter::SampleRec
    SampleSpot(const Vec3 &origin, const float xi_0, const float xi_1) const;
    QUALIFIER_D_H Emitter::SampleRec
    SampleSun(const Vec3 &origin, const float xi_0, const float xi_1) const;
    QUALIFIER_D_H Emitter::SampleRec
    SampleEnvMap(const Vec3 &origin, const float xi_0, const float xi_1) const;

    QUALIFIER_D_H Vec3 EvaluateSpot(const SampleRec &rec) const;
    QUALIFIER_D_H Vec3 EvaluateEnvMap(const SampleRec &rec) const;

    QUALIFIER_D_H float PdfEnvMap(const Vec3 &look_dir) const;

    QUALIFIER_D_H Vec3 EvaluateSun(const Vec3 &look_dir) const;
    QUALIFIER_D_H Vec3 EvaluateEnvMap(const Vec3 &look_dir) const;

    uint32_t id_;
    TLAS *tlas_;
    Data data_;
};

} // namespace csrt