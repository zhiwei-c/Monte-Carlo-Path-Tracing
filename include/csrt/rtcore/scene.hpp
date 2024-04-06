#ifndef CSRT__RTCORE__SCENE_HPP
#define CSRT__RTCORE__SCENE_HPP

#include <vector>

#include "../utils.hpp"
#include "accel/bvh_builder.hpp"
#include "accel/tlas.hpp"
#include "instance.hpp"
#include "primitives/primitive.hpp"

namespace csrt
{

class Scene
{
public:
    Scene(const BackendType backend_type,
          const std::vector<InstanceInfo> &list_info_instance);
    ~Scene() { ReleaseData(); }

    TLAS *GetTlas() const { return tlas_; };
    Instance *GetInstances() const { return instances_; }
    float *GetPdfAreaList() const { return list_pdf_area_; }

private:
    void ReleaseData();

    void CommitPrimitives(const std::vector<InstanceInfo> &list_info_instance);
    void CommitInstances(const std::vector<InstanceInfo> &list_info_instance);

    void CommitRectangle(InstanceInfo info);
    void CommitCube(InstanceInfo info);
    void CommitMeshes(InstanceInfo info);
    void CommitSphere(const InstanceInfo &info);

    BackendType backend_type_;
    Instance *instances_;
    Primitive *primitives_;
    BvhNode *nodes_;
    TLAS *tlas_;
    BLAS *list_blas_;
    // 场景中所有实例按面积均匀抽样时的概率（面积的倒数）
    float *list_pdf_area_;
};

} // namespace csrt

#endif