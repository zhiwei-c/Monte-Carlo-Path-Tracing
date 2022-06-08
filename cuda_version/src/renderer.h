#pragma once

#include <vector>
#include <unordered_set>
#include <device_launch_parameters.h>

#include "core/camera.h"
#include "core/bsdfs.h"
#include "utils/timer.h"
#include "utils/model_loader.h"

__global__ void RenderInit(int max_x, int max_y, int resolution, curandState *rand_state);

__global__ void RenderProcess(int width, int height, int resolution, curandState *rand_state, Camera *camera,
                              Integrator *integrator, float *frame_data, volatile int *progress);

class Renderer
{
public:
    Renderer()
        : texture_list_(nullptr), bsdf_list_(nullptr), mesh_list_(nullptr), shapebvh_list_(nullptr),
          scenebvh_node_list_(nullptr), scenebvh_(nullptr), integrator_(nullptr), camera_(nullptr),
          emitter_idx_list_(nullptr), d_rand_state_(nullptr), env_map_info_(nullptr), env_map_(nullptr),
          bsdf_info_list_({}), texture_info_list_({}), shape_info_list_({}), emitter_shape_idx_list_({}),
          bvhnode_list_({}), texture_bitmap_data_({})
    {
    }

    ~Renderer();

    void AddBsdfInfo(const BsdfInfo &bsdf_info)
    {
        bsdf_info_list_.push_back(bsdf_info);
        if (bsdf_info.type == kAreaLight)
            emitter_bsdf_idx_list_.insert(bsdf_info_list_.size() - 1);
    }

    void AddTextureInfo(TextureInfo *texture_info)
    {
        texture_info_list_.push_back(texture_info);
    }

    void AddShapeInfo(ShapeInfo *shape_info)
    {
        shape_info_list_.push_back(shape_info);
        if (emitter_bsdf_idx_list_.count(shape_info->bsdf_idx))
            emitter_shape_idx_list_.push_back(shape_info_list_.size() - 1);
    }

    void AddIntegratorInfo(const IntegratorInfo &integrator_info)
    {
        integrator_info_ = integrator_info;
    }

    void AddCameraInfo(const CameraInfo &camera_info)
    {
        camera_info_ = camera_info;
    }

    void AddEnvMapInfo(EnvMapInfo *env_map_info)
    {
        env_map_info_ = env_map_info;
    }

    void Render(const std::string &output_filename);

private:
    void InitTextureList();

    void InitBsdfList();

    void InitVertexIndexBuffer(Vertex *&vertex_list, uvec3 *&mesh_idx_list, uint *&mesh_bsdf_idx_list, uint &mesh_num,
                               std::vector<uvec2> &mesh_idx_range_list);

    void InitShapesMeshes(std::vector<uvec2> &mesh_idx_range_list, std::vector<AABB> &mesh_aabb_list,
                          std::vector<Float> &mesh_area_list);

    void InitShapeBvh(std::vector<BvhNodeInfo> &shape_info_list);

    void InitSceneBvh(AABB &scene_aabb);

    void InitEnvMap(const AABB &scene_aabb);

    void initIntegratorCamera();

    Timer timer_;
    Texture *texture_list_;
    Bsdf **bsdf_list_;
    Shape *mesh_list_;
    ShapeBvh *shapebvh_list_;
    ShapeBvh *scenebvh_node_list_;
    SceneBvh *scenebvh_;
    Integrator *integrator_;
    Camera *camera_;
    uint *emitter_idx_list_;
    curandState *d_rand_state_;
    IntegratorInfo integrator_info_;
    EnvMapInfo *env_map_info_;
    EnvMap *env_map_;
    CameraInfo camera_info_;
    std::vector<BsdfInfo> bsdf_info_list_;
    std::vector<TextureInfo *> texture_info_list_;
    std::vector<ShapeInfo *> shape_info_list_;
    std::vector<uint> emitter_shape_idx_list_;
    std::vector<BvhNode *> bvhnode_list_;
    std::vector<float *> texture_bitmap_data_;
    std::unordered_set<uint> emitter_bsdf_idx_list_;
};