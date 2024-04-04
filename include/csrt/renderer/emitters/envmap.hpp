#ifndef CSRT__RENDERER__EMITTERS__ENVMAP_HPP
#define CSRT__RENDERER__EMITTERS__ENVMAP_HPP

#include "../../rtcore/scene.hpp"
#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../textures/texture.hpp"

namespace csrt
{

struct EmitterSampleRec;

struct EnvMapInfo
{
    uint32_t id_radiance = kInvalidId;
    Mat4 to_world = {};
};

struct EnvMapData
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


void CreateEnvMapCdfPdf(const int width, const int height,
                        const Texture &radiance, std::vector<float> *cdf_cols,
                        std::vector<float> *cdf_rows,
                        std::vector<float> *weight_rows, float *normalization);

QUALIFIER_D_H void SampleEnvMap(const EnvMapData &data, const Vec3 &origin,
                                const float xi_0, const float xi_1,
                                EmitterSampleRec *rec);

QUALIFIER_D_H Vec3 EvaluateEnvMap(const EnvMapData &data,
                                  const EmitterSampleRec *rec);

QUALIFIER_D_H Vec3 EvaluateEnvMap(const EnvMapData &data, const Vec3 &look_dir);

QUALIFIER_D_H float PdfEnvMap(const EnvMapData &data, const Vec3 &look_dir);

} // namespace csrt

#endif