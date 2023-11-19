#pragma once

#include <vector>

#include "bvh_builder.cuh"
#include "instance.cuh"
#include "primitive.cuh"
#include "utils.cuh"

namespace csrt
{

class TLAS
{
public:
    QUALIFIER_D_H TLAS();
    QUALIFIER_D_H TLAS(const Instance *instances, const BvhNode *node_buffer);

    QUALIFIER_D_H Hit Intersect(Ray *ray) const;

private:
    const BvhNode *nodes_;
    const Instance *instances_;
};

class Scene
{
public:
    Scene(const BackendType backend_type);
    ~Scene();

    void AddInstance(const Instance::Info &info);
    void Commit();

    TLAS *GetTlas() { return tlas_; };
    Instance *GetInstances() { return instances_; }
    float *GetPdfAreaList() { return list_pdf_area_; }

private:
    void CommitPrimitives();
    void CommitInstances();

    void CommitCube(const uint32_t id);
    void CommitSphere(const uint32_t id);
    void CommitRectangle(const uint32_t id);
    void CommitMeshes(const uint32_t id);

    void SetupMeshes(Instance::Info::Meshes info_meshes,
                     std::vector<Primitive::Data> *list_data_primitve,
                     std::vector<float> *areas);

    BackendType backend_type_;
    TLAS *tlas_;
    Instance *instances_;
    BLAS *list_blas_;
    float *list_pdf_area_;
    Primitive *primitives_;
    BvhNode *nodes_;
    uint64_t num_primitive_;
    uint64_t num_node_;
    std::vector<Instance::Info> list_info_instance_;
    std::vector<uint64_t> list_offset_primitive_;
    std::vector<uint64_t> list_offset_node_;
};

} // namespace csrt