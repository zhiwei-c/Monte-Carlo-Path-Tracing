#pragma once

#include <vector>

#include "../rtcore/scene.cuh"
#include "../utils.cuh"
#include "bsdfs/bsdf.cuh"
#include "camera.cuh"
#include "emitters/emitter.cuh"
#include "integrator.cuh"
#include "textures/texture.cuh"

namespace csrt
{

class Renderer
{
public:
    Renderer(const BackendType backend_type);
    ~Renderer();

    void AddTexture(const TextureData &info);
    void AddBsdf(const BsdfInfo &info);
    void AddSceneInfo(Instance *instances, float *list_pdf_area_instance,
                      const std::vector<uint32_t> &map_instance_bsdf,
                      TLAS *tlas);
    void AddEmitter(const EmitterInfo &info);
    void
    SetAreaLightInfo(const std::vector<uint32_t> map_id_area_light_instance,
                     const std::vector<float> list_area_light_weight);
    void SetCamera(const Camera::Info &info);
    void SetIntegrator(const Integrator::Info &info);

    void Commit();

    void Draw(float *frame) const;
#ifdef ENABLE_VIEWER
    void Draw(const uint32_t index_frame, float *accum, float *frame) const;
#endif

private:
    void CommitTextures();
    void CommitBsdfs();
    void CommitEmitters();
    void CommitIntegrator();

    void CheckTexture(const uint32_t id, const bool allow_invalid);
    void CheckBsdf(const uint32_t id, const bool allow_invalid);

    BackendType backend_type_;

    uint32_t num_instance_;
    uint32_t num_area_light_;
    uint32_t id_sun_;
    uint32_t id_envmap_;

    float *pixels_;
    Texture *textures_;
    Bsdf *bsdfs_;
    Instance *instances_;
    Emitter *emitters_;
    float *data_env_map_;
    TLAS *tlas_;
    Integrator *integrator_;
    Camera *camera_;
    // 面光源抽样权重的累积分布函数
    float *cdf_area_light_;
    // 场景中所有实例按面积均匀抽样时的概率（面积的倒数）
    float *list_pdf_area_instance_;
    // 从实例ID到相应BSDF ID的映射
    uint32_t *map_instance_bsdf_;
    // 从实例ID到相应面光源ID的映射
    uint32_t *map_instance_area_light_;
    // 从面光源ID到相应实例ID的映射
    uint32_t *map_area_light_instance_;

    Integrator::Info info_integrator_;

    std::vector<TextureData> list_texture_data_;
    std::vector<BsdfInfo> list_bsdf_info_;
    std::vector<EmitterInfo> list_emitter_info_;
};

} // namespace csrt
