#pragma once

#include "../accelerators/accel.cuh"
#include "../renderer/camera.cuh"
#include "../renderer/integrator.cuh"
#include "../utils/config_parser.cuh"

class Painter
{
public:
    virtual ~Painter() {}

    virtual void Draw(const std::string &filename) = 0;

protected:
    Painter() {}

    Camera *camera_;
    uint32_t num_texture_;
    uint32_t num_bsdf_;
    uint32_t num_primitive_;
    uint32_t num_emitter_;
    uint32_t num_area_light_;
    float *brdf_avg_buffer_;
    float *albedo_avg_buffer_;
    float *pixel_buffer_;
    Texture **texture_buffer_;
    Bsdf **bsdf_buffer_;
    Emitter **emitter_buffer_;
    Primitive *primitive_buffer_;
    Instance *instance_buffer_;
    uint32_t *area_light_id_buffer_;
    EnvMap *env_map_;
    Sun *sun_;
    BvhNode *bvh_node_buffer_;
    Accel *accel_;
    Integrator *integrator_;
};

class CpuPainter : public Painter
{
public:
    CpuPainter(BvhBuilder::Type bvh_type, const SceneInfo &info);
    ~CpuPainter() override;

    void Draw(const std::string &filename) override;
};

#ifdef ENABLE_CUDA

class CudaPainter : public Painter
{
public:
    CudaPainter(BvhBuilder::Type bvh_type, const SceneInfo &info);
    virtual ~CudaPainter() override;

    virtual void Draw(const std::string &filename) override;

protected:
    dim3 num_blocks_;
    dim3 threads_per_block_;
};

#ifdef ENABLE_VIEWER
class CudaViewer : public CudaPainter
{
public:
    CudaViewer(int argc, char **argv, BvhBuilder::Type bvh_type, const SceneInfo &info);
    ~CudaViewer() override;

    void Draw(const std::string &filename) override;
};
#endif

#endif