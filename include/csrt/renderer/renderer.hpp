#ifndef CSRT__RENDERER__RENDERER_HPP
#define CSRT__RENDERER__RENDERER_HPP

#include <vector>

#include "../rtcore/scene.hpp"
#include "../utils.hpp"
#include "bsdfs/bsdf.hpp"
#include "camera.hpp"
#include "emitters/emitter.hpp"
#include "integrators/integrator.hpp"
#include "medium/medium.hpp"
#include "textures/texture.hpp"

namespace csrt
{

struct RendererConfig
{
    BackendType backend_type;
    Camera::Info camera;
    IntegratorInfo integrator;
    std::vector<TextureInfo> textures;
    std::vector<BsdfInfo> bsdfs;
    std::vector<MediumInfo> media;
    std::vector<InstanceInfo> instances;
    std::vector<EmitterInfo> emitters;
};

class Renderer
{
public:
    Renderer(const RendererConfig &config);
    ~Renderer() { ReleaseData(); }

    void Draw(float *frame) const;
#ifdef ENABLE_VIEWER
    void Draw(const uint32_t index_frame, float *frame, float *frame_srgb) const;
#endif

private:
    void ReleaseData();

    void CommitTextures(const std::vector<TextureInfo> &list_texture_info);
    void CommitBsdfs(const size_t num_texture,
                     const std::vector<BsdfInfo> &list_bsdf_info);
    void CommitMedia(const std::vector<MediumInfo> &list_medium_info);
    void CommitEmitters(const std::vector<TextureInfo> &list_texture_info,
                        const std::vector<EmitterInfo> &list_emitter_info,
                        uint32_t *id_sun, uint32_t *id_envmap);
    void CommitIntegrator(const IntegratorInfo &integrator_info,
                          const uint32_t num_area_light,
                          const uint32_t num_emitter, const uint32_t id_sun,
                          const uint32_t id_envmap);

    BackendType backend_type_;
    Scene *scene_;
    Camera *camera_;
    Texture *textures_;
    Bsdf *bsdfs_;
    Medium *media_;
    Emitter *emitters_;
    Integrator *integrator_;

    // 从实例ID到相应BSDF ID的映射
    uint32_t *map_instance_bsdf_;
    // 从面光源ID到相应实例ID的映射
    uint32_t *map_area_light_instance_;
    // 从实例ID到相应面光源ID的映射
    uint32_t *map_instance_area_light_;
    // 面光源抽样权重的累积分布函数
    float *cdf_area_light_;
    // 位图纹理像素数据
    float *pixels_;
    // 环境映射纹理数据
    float *data_env_map_;
    // Kulla-Conty LUT
    float *brdf_avg_buffer_;
    // Kulla-Conty LUT
    float *albedo_avg_buffer_;
};

} // namespace csrt

#endif