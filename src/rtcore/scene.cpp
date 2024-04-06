#include "csrt/rtcore/scene.hpp"

#include <exception>

namespace
{

using namespace csrt;

uint64_t g_num_primitive;
uint64_t g_num_node;
std::vector<uint64_t> g_list_offset_primitive;
std::vector<uint64_t> g_list_offset_node;

void SetupMeshes(const MeshesInfo &info,
                 std::vector<PrimitiveData> *list_data_primitve,
                 std::vector<float> *areas)
{
    try
    {
        const uint32_t num_primitive_local =
            static_cast<uint32_t>(info.indices.size());
        *list_data_primitve = std::vector<PrimitiveData>(num_primitive_local);
        *areas = std::vector<float>(num_primitive_local);
        for (uint32_t i = 0; i < num_primitive_local; ++i)
        {
            (*list_data_primitve)[i].type = PrimitiveType::kTriangle;
            const Uvec3 indices = info.indices[i];
            TriangleData &triangle = (*list_data_primitve)[i].triangle;

            if (info.texcoords.empty())
            {
                triangle.texcoords[0] = {0, 0};
                triangle.texcoords[1] = {1, 0};
                triangle.texcoords[2] = {1, 1};
            }
            else
            {
                for (int j = 0; j < 3; ++j)
                    triangle.texcoords[j] = info.texcoords[indices[j]];
            }

            for (int j = 0; j < 3; ++j)
                triangle.positions[j] = info.positions[indices[j]];

            const Vec3 v0v1 = triangle.positions[1] - triangle.positions[0],
                       v0v2 = triangle.positions[2] - triangle.positions[0];
            const Vec3 normal_geom = Cross(v0v1, v0v2);
            (*areas)[i] = Length(normal_geom);

            if (info.normals.empty())
            {
                const Vec3 normal = Normalize(normal_geom);
                for (int j = 0; j < 3; ++j)
                    triangle.normals[j] = normal;
            }
            else
            {
                for (int j = 0; j < 3; ++j)
                    triangle.normals[j] = info.normals[indices[j]];
            }

            if (info.tangents.empty() && info.bitangents.empty())
            {
                const Vec2 uv_delta_01 =
                               triangle.texcoords[1] - triangle.texcoords[0],
                           uv_delta_02 =
                               triangle.texcoords[2] - triangle.texcoords[0];
                const float r = 1.0f / (uv_delta_01.y * uv_delta_02.x -
                                        uv_delta_01.x * uv_delta_02.y);
                const Vec3 tangent = Normalize(
                    (uv_delta_01.y * v0v2 - uv_delta_02.y * v0v1) * r);
                for (int j = 0; j < 3; ++j)
                {
                    triangle.bitangents[j] =
                        Normalize(Cross(triangle.normals[j], tangent));
                    triangle.tangents[j] = Normalize(
                        Cross(triangle.bitangents[j], triangle.normals[j]));
                }
            }
            else if (info.tangents.empty())
            {
                for (int j = 0; j < 3; ++j)
                {
                    triangle.bitangents[j] = info.bitangents[indices[j]];
                    triangle.tangents[j] = Normalize(
                        Cross(triangle.bitangents[j], triangle.normals[j]));
                    triangle.bitangents[j] = Normalize(
                        Cross(triangle.normals[j], triangle.tangents[j]));
                }
            }
            else
            {
                for (int j = 0; j < 3; ++j)
                {
                    triangle.tangents[j] = info.tangents[indices[j]];
                    triangle.bitangents[j] = Normalize(
                        Cross(triangle.normals[j], triangle.tangents[j]));
                    triangle.tangents[j] = Normalize(
                        Cross(triangle.bitangents[j], triangle.normals[j]));
                }
            }
        }
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when add 'meshes' instance.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

} // namespace

namespace csrt
{

Scene::Scene(const BackendType backend_type,
             const std::vector<InstanceInfo> &list_info_instance)
    : backend_type_(backend_type), instances_(nullptr), primitives_(nullptr),
      nodes_(nullptr), tlas_(nullptr), list_blas_(nullptr),
      list_pdf_area_(nullptr)
{
    try
    {
        g_num_primitive = 0;
        g_num_node = 0;
        g_list_offset_primitive = {};
        g_list_offset_node = {};

        CommitPrimitives(list_info_instance);
        CommitInstances(list_info_instance);
    }
    catch (const MyException &e)
    {
        ReleaseData();
        std::ostringstream oss;
        oss << "error when commit scene.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Scene::ReleaseData()
{
    DeleteArray(backend_type_, instances_);
    DeleteArray(backend_type_, primitives_);
    DeleteArray(backend_type_, nodes_);
    DeleteElement(backend_type_, tlas_);
    DeleteArray(backend_type_, list_blas_);
    DeleteArray(backend_type_, list_pdf_area_);
}

void Scene::CommitPrimitives(
    const std::vector<InstanceInfo> &list_info_instance)
{
    try
    {
        const uint32_t num_instance =
            static_cast<uint32_t>(list_info_instance.size());
        for (uint32_t i = 0; i < num_instance; ++i)
        {
            switch (list_info_instance[i].type)
            {
            case InstanceType::kSphere:
                CommitSphere(list_info_instance[i]);
                break;
            case InstanceType::kDisk:
                CommitDisk(list_info_instance[i]);
                break;
            case InstanceType::kCylinder:
                CommitCylinder(list_info_instance[i]);
                break;
            case InstanceType::kRectangle:
                CommitRectangle(list_info_instance[i]);
                break;
            case InstanceType::kCube:
                CommitCube(list_info_instance[i]);
                break;
            case InstanceType::kMeshes:
                CommitMeshes(list_info_instance[i]);
                break;
            default:
                throw MyException("unknow instance type.");
                break;
            }
        }
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when commit geometry.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Scene::CommitRectangle(InstanceInfo info)
{
    info.meshes.texcoords = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    info.meshes.positions = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
    info.meshes.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    info.meshes.indices = {{0, 1, 2}, {2, 3, 0}};
    try
    {
        CommitMeshes(info);
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when add 'reactangle' instance.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Scene::CommitCube(InstanceInfo info)
{
    info.meshes.texcoords = {{0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1},
                             {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0},
                             {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1},
                             {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}};
    info.meshes.positions = {
        {1, -1, -1}, {1, -1, 1},  {-1, -1, 1},  {-1, -1, -1}, {1, 1, -1},
        {-1, 1, -1}, {-1, 1, 1},  {1, 1, 1},    {1, -1, -1},  {1, 1, -1},
        {1, 1, 1},   {1, -1, 1},  {1, -1, 1},   {1, 1, 1},    {-1, 1, 1},
        {-1, -1, 1}, {-1, -1, 1}, {-1, 1, 1},   {-1, 1, -1},  {-1, -1, -1},
        {1, 1, -1},  {1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}};
    info.meshes.normals = {
        {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, 1, 0},  {0, 1, 0},
        {0, 1, 0},  {0, 1, 0},  {1, 0, 0},  {1, 0, 0},  {1, 0, 0},  {1, 0, 0},
        {0, 0, 1},  {0, 0, 1},  {0, 0, 1},  {0, 0, 1},  {-1, 0, 0}, {-1, 0, 0},
        {-1, 0, 0}, {-1, 0, 0}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}};
    info.meshes.indices = {{0, 1, 2},    {3, 0, 2},    {4, 5, 6},
                           {7, 4, 6},    {8, 9, 10},   {11, 8, 10},
                           {12, 13, 14}, {15, 12, 14}, {16, 17, 18},
                           {19, 16, 18}, {20, 21, 22}, {23, 20, 22}};
    try
    {
        CommitMeshes(info);
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when add 'cube' instance.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Scene::CommitMeshes(InstanceInfo info)
{
    if (info.meshes.indices.empty())
    {
        throw MyException(
            "cannot find vertex index info when adding instance to scene.");
    }

    if (info.meshes.positions.empty())
    {
        throw MyException("cannot find vertex position info when adding "
                          "instance to scene.");
    }

    for (Vec3 &position : info.meshes.positions)
        position = TransformPoint(info.to_world, position);

    if (!info.meshes.normals.empty())
    {
        const Mat4 normal_to_world = info.to_world.Transpose().Inverse();
        for (Vec3 &normal : info.meshes.normals)
            normal = TransformVector(normal_to_world, normal);
    }

    if (!info.meshes.tangents.empty())
    {
        for (Vec3 &tangent : info.meshes.tangents)
            tangent = TransformVector(info.to_world, tangent);
    }

    if (!info.meshes.bitangents.empty())
    {
        for (Vec3 &bitangent : info.meshes.bitangents)
            bitangent = TransformVector(info.to_world, bitangent);
    }

    try
    {
        std::vector<PrimitiveData> list_data_primitve;
        std::vector<float> areas;
        SetupMeshes(info.meshes, &list_data_primitve, &areas);
        const uint32_t num_primitive_local =
            static_cast<uint32_t>(list_data_primitve.size());

        Primitive *primitives = MallocArray<Primitive>(
            backend_type_, g_num_primitive + num_primitive_local);
        CopyArray(backend_type_, primitives, primitives_, g_num_primitive);
        DeleteArray(backend_type_, primitives_);
        std::vector<AABB> aabbs(num_primitive_local);
        for (uint32_t i = 0; i < num_primitive_local; ++i)
        {
            primitives[g_num_primitive + i] =
                Primitive(i, list_data_primitve[i]);
            aabbs[i] = primitives[g_num_primitive + i].aabb();
        }
        primitives_ = primitives;
        g_list_offset_primitive.push_back(g_num_primitive);
        g_num_primitive += num_primitive_local;

        std::vector<BvhNode> list_node = BvhBuilder::Build(aabbs, areas);
        const uint64_t num_node_local = list_node.size();
        BvhNode *nodes =
            MallocArray<BvhNode>(backend_type_, g_num_node + num_node_local);
        CopyArray(backend_type_, nodes, nodes_, g_num_node);
        DeleteArray(backend_type_, nodes_);
        for (uint64_t i = 0; i < num_node_local; ++i)
            nodes[g_num_node + i] = list_node[i];
        nodes_ = nodes;
        g_list_offset_node.push_back(g_num_node);
        g_num_node += num_node_local;
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when add 'meshes' instance.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Scene::CommitSphere(const InstanceInfo &info)
{
    try
    {
        PrimitiveData data_primitive;
        data_primitive.type = PrimitiveType::kSphere;
        data_primitive.sphere.radius = info.sphere.radius;
        data_primitive.sphere.center = info.sphere.center;
        data_primitive.sphere.to_world = info.to_world;

        Primitive *primitives =
            MallocArray<Primitive>(backend_type_, g_num_primitive + 1);
        CopyArray(backend_type_, primitives, primitives_, g_num_primitive);
        DeleteArray(backend_type_, primitives_);
        primitives[g_num_primitive] = Primitive(0, data_primitive);
        std::vector<AABB> aabbs = {primitives[g_num_primitive].aabb()};
        primitives_ = primitives;
        g_list_offset_primitive.push_back(g_num_primitive);
        ++g_num_primitive;

        const Vec3 center_world =
                       TransformPoint(info.to_world, info.sphere.center),
                   boundary_local = info.sphere.center +
                                    Vec3{info.sphere.radius, 0.0f, 0.0f},
                   boundary_world =
                       TransformPoint(info.to_world, boundary_local);
        const float radius_world = Length(center_world - boundary_world);
        std::vector<float> areas = {4.0f * kPi * Sqr(radius_world)};

        std::vector<BvhNode> list_node = BvhBuilder::Build(aabbs, areas);
        const uint64_t num_node_local = list_node.size();
        BvhNode *nodes =
            MallocArray<BvhNode>(backend_type_, g_num_node + num_node_local);
        CopyArray(backend_type_, nodes, nodes_, g_num_node);
        DeleteArray(backend_type_, nodes_);
        for (uint64_t i = 0; i < num_node_local; ++i)
            nodes[g_num_node + i] = list_node[i];
        nodes_ = nodes;
        g_list_offset_node.push_back(g_num_node);
        g_num_node += num_node_local;
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when add 'sphere' instance.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Scene::CommitDisk(const InstanceInfo &info)
{
    try
    {
        PrimitiveData data_primitive;
        data_primitive.type = PrimitiveType::kDisk;
        data_primitive.disk.to_world = info.to_world;

        Primitive *primitives =
            MallocArray<Primitive>(backend_type_, g_num_primitive + 1);
        CopyArray(backend_type_, primitives, primitives_, g_num_primitive);
        DeleteArray(backend_type_, primitives_);
        primitives[g_num_primitive] = Primitive(0, data_primitive);
        std::vector<AABB> aabbs = {primitives[g_num_primitive].aabb()};
        primitives_ = primitives;
        g_list_offset_primitive.push_back(g_num_primitive);
        ++g_num_primitive;

        const Vec3 center_world = TransformPoint(info.to_world, Vec3{0}),
                   boundary_world =
                       TransformPoint(info.to_world, Vec3{0.5f, 0, 0});
        const float radius_world = Length(center_world - boundary_world);
        std::vector<float> areas = {kPi * Sqr(radius_world)};

        std::vector<BvhNode> list_node = BvhBuilder::Build(aabbs, areas);
        const uint64_t num_node_local = list_node.size();
        BvhNode *nodes =
            MallocArray<BvhNode>(backend_type_, g_num_node + num_node_local);
        CopyArray(backend_type_, nodes, nodes_, g_num_node);
        DeleteArray(backend_type_, nodes_);
        for (uint64_t i = 0; i < num_node_local; ++i)
            nodes[g_num_node + i] = list_node[i];
        nodes_ = nodes;
        g_list_offset_node.push_back(g_num_node);
        g_num_node += num_node_local;
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when add 'disk' instance.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Scene::CommitCylinder(const InstanceInfo &info)
{
    try
    {
        PrimitiveData data_primitive;
        data_primitive.type = PrimitiveType::kCylinder;

        data_primitive.cylinder.to_world =
            LocalToWorld(Normalize(info.cylinder.p1 - info.cylinder.p0));
        data_primitive.cylinder.to_world =
            Mul(Translate(info.cylinder.p0), data_primitive.cylinder.to_world);
        data_primitive.cylinder.to_world =
            Mul(info.to_world, data_primitive.cylinder.to_world);
        data_primitive.cylinder.length =
            Length(TransformPoint(
                       data_primitive.cylinder.to_world,
                       {0, 0, Length(info.cylinder.p1 - info.cylinder.p0)}) -
                   TransformPoint(data_primitive.cylinder.to_world, {0, 0, 0}));
        data_primitive.cylinder.radius =
            Length(TransformPoint(data_primitive.cylinder.to_world,
                                  {info.cylinder.radius, 0, 0}) -
                   TransformPoint(data_primitive.cylinder.to_world, {0, 0, 0}));

        Primitive *primitives =
            MallocArray<Primitive>(backend_type_, g_num_primitive + 1);
        CopyArray(backend_type_, primitives, primitives_, g_num_primitive);
        DeleteArray(backend_type_, primitives_);
        primitives[g_num_primitive] = Primitive(0, data_primitive);
        std::vector<AABB> aabbs = {primitives[g_num_primitive].aabb()};
        primitives_ = primitives;
        g_list_offset_primitive.push_back(g_num_primitive);
        ++g_num_primitive;

        std::vector<float> areas = {k2Pi * Sqr(data_primitive.cylinder.radius)};

        std::vector<BvhNode> list_node = BvhBuilder::Build(aabbs, areas);
        const uint64_t num_node_local = list_node.size();
        BvhNode *nodes =
            MallocArray<BvhNode>(backend_type_, g_num_node + num_node_local);
        CopyArray(backend_type_, nodes, nodes_, g_num_node);
        DeleteArray(backend_type_, nodes_);
        for (uint64_t i = 0; i < num_node_local; ++i)
            nodes[g_num_node + i] = list_node[i];
        nodes_ = nodes;
        g_list_offset_node.push_back(g_num_node);
        g_num_node += num_node_local;
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when add 'disk' instance.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Scene::CommitInstances(const std::vector<InstanceInfo> &list_info_instance)
{
    try
    {
        const uint32_t num_instance =
            static_cast<uint32_t>(list_info_instance.size());

        //
        // 生成顶层加速结构的节点
        //
        std::vector<AABB> aabbs(num_instance);
        std::vector<float> areas(num_instance);
        for (uint32_t i = 0; i < num_instance; ++i)
        {
            const uint64_t index = g_list_offset_node[i];
            aabbs[i] = nodes_[index].aabb;
            areas[i] = nodes_[index].area;
        }

        list_pdf_area_ = MallocArray(backend_type_, areas);
        for (uint32_t i = 0; i < num_instance; ++i)
            list_pdf_area_[i] = 1.0f / list_pdf_area_[i];

        std::vector<BvhNode> list_node = BvhBuilder::Build(aabbs, areas);
        const uint64_t num_node_local = list_node.size();
        BvhNode *nodes =
            MallocArray<BvhNode>(backend_type_, g_num_node + num_node_local);
        for (uint64_t i = 0; i < num_node_local; ++i)
            nodes[i] = list_node[i];
        for (uint64_t i = 0; i < g_num_node; ++i)
            nodes[num_node_local + i] = nodes_[i];
        DeleteArray(backend_type_, nodes_);
        nodes_ = nodes;
        for (uint32_t i = 0; i < num_instance; ++i)
            g_list_offset_node[i] += num_node_local;

        //
        // 生成底层加速结构和实例
        //
        instances_ = MallocArray<Instance>(backend_type_, num_instance);
        list_blas_ = MallocArray<BLAS>(backend_type_, num_instance);
        for (uint32_t i = 0; i < num_instance; ++i)
        {
            list_blas_[i] = BLAS(g_list_offset_node[i], nodes_,
                                 g_list_offset_primitive[i], primitives_);
            instances_[i] =
                Instance(i, list_info_instance[i].id_medium_int,
                         list_info_instance[i].id_medium_ext, list_blas_);
        }

        tlas_ = MallocElement<TLAS>(backend_type_);
        *tlas_ = TLAS(instances_, nodes_);
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when commit geometry.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

} // namespace csrt