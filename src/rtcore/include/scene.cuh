#pragma once

#include <vector>

#include "bvh_builder.cuh"
#include "instance.cuh"
#include "primitive.cuh"
#include "utils.cuh"

namespace rt
{

class TLAS
{
public:
    QUALIFIER_D_H TLAS();
    QUALIFIER_D_H TLAS(const uint64_t offset_node, const BvhNode *node_buffer,
                       const Instance *instances);

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

    void CommitCube(const uint32_t id, const Instance::Info::Cube &info);
    void CommitSphere(const uint32_t id, const Instance::Info::Sphere &info);
    void CommitRectangle(const uint32_t id,
                         const Instance::Info::Rectangle &info);
    void CommitMeshes(const uint32_t id, Instance::Info::Meshes info);

    void SetupMeshes(Instance::Info::Meshes info,
                     std::vector<Primitive::Data> *data_primitves,
                     std::vector<float> *areas);

    BackendType backend_type_;
    TLAS *tlas_;
    Instance *instances_;
    float *list_pdf_area_;
    BLAS *blas_buffer_;
    uint64_t num_primitive_;
    Primitive *primitive_buffer_;
    std::vector<uint64_t> list_offset_primitive_;
    uint64_t num_node_;
    BvhNode *node_buffer_;
    std::vector<uint64_t> list_offset_node_;
    std::vector<Instance::Info> list_info_instance_;
};

} // namespace rt