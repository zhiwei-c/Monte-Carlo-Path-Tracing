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
    QUALIFIER_D_H TLAS(BvhNode *nodes, Instance *instances);

    QUALIFIER_D_H void Intersect(Ray *ray, Hit *hit) const;

private:
    BvhNode *nodes_;
    Instance *instances_;
};

class Scene
{
public:
    Scene(const BackendType backend_type);
    ~Scene();

    void AddInstance(const Instance::Info &info);
    TLAS *Commit();

private:
    void AddCube(const Instance::Info::Cube &info);

    void AddSphere(const Instance::Info::Sphere &info);

    void AddRectangle(const Instance::Info::Rectangle &info);

    void AddMeshes(Instance::Info::Meshes info);

    BackendType backend_type_;
    TLAS *tlas_;
    Instance *instance_buffer_;
    std::vector<Instance *> instances_;
    std::vector<BLAS *> blas_buffer_;
    std::vector<BvhNode *> nodes_;
    std::vector<Primitive *> primitives_;
};

} // namespace rt