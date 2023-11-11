#pragma once

#include <vector>

#include "bsdf.cuh"
#include "camera.cuh"
#include "integrator.cuh"
#include "rtcore.cuh"
#include "texture.cuh"
#include "utils.cuh"

namespace rt
{

class Renderer
{
public:
    Renderer(const BackendType backend_type);
    ~Renderer();

    void AddTexture(const Texture::Info &info);
    void AddBsdf(const Bsdf::Info &info);
    void AddSceneInfo(const std::vector<uint32_t> &map_id_instance_bsdf,
                      Instance *instances, float *list_pdf_area_instance,
                      TLAS *tlas);
    void
    SetAreaLightInfo(const std::vector<uint32_t> map_id_area_light_instance,
                     const std::vector<float> list_area_light_weight);
    void SetCamera(const Camera::Info &info);
    void SetIntegrator(const Integrator::Info &info);

    void Commit();

    void Draw(float *frame) const;

private:
    void CommitTextures();
    void CommitBsdfs();
    void CommitIntegrator();

    void CheckTexture(const uint32_t id, const bool allow_invalid);
    void CheckBsdf(const uint32_t id, const bool allow_invalid);

    BackendType backend_type_;

    std::vector<Texture::Info> list_texture_info_;
    Texture *list_texture_;
    float *list_pixel_;

    std::vector<Bsdf::Info> list_bsdf_info_;
    Bsdf *list_bsdf_;

    uint32_t num_area_light_;
    // 从面光源ID到相应实例ID的映射
    uint32_t *map_id_area_light_instance_;
    // 从实例ID到相应面光源ID的映射
    uint32_t *map_id_instance_area_light_;
    // 面光源抽样权重的累积分布函数
    float *cdf_area_light_;

    uint32_t num_instance_;
    TLAS *tlas_;
    Instance *instances_;
    // 从实例ID到相应BSDF ID的映射
    uint32_t *map_instance_bsdf_;
    // 场景中所有实例按面积均匀抽样时的概率（面积的倒数）
    float *list_pdf_area_instance_;

    Integrator::Info info_integrator_;
    Integrator *integrator_;

    Camera *camera_;
};

} // namespace rt
