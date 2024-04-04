#include "csrt/rtcore/scene.hpp"

#include <exception>

namespace csrt
{

Scene::Scene(const BackendType backend_type)
    : backend_type_(backend_type), num_primitive_(0), num_node_(0),
      tlas_(nullptr), instances_(nullptr), list_blas_(nullptr),
      list_pdf_area_(nullptr), primitives_(nullptr), nodes_(nullptr)
{
}

Scene::~Scene()
{
    DeleteArray(backend_type_, nodes_);
    DeleteArray(backend_type_, primitives_);
    DeleteArray(backend_type_, list_pdf_area_);
    DeleteArray(backend_type_, list_blas_);
    DeleteArray(backend_type_, instances_);
    DeleteElement(backend_type_, tlas_);
}

void Scene::AddInstance(const Instance::Info &info)
{
    try
    {
        switch (info.type)
        {
        case Instance::Type::kCube:
        case Instance::Type::kSphere:
        case Instance::Type::kRectangle:
        case Instance::Type::kMeshes:
            break;
        default:
            throw MyException("unknow instance type.");
            break;
        }
        list_info_instance_.push_back(info);
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when add instance to scene.\n\t" << e.what();
        throw MyException(oss.str());
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when add instance to scene.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Scene::Commit()
{
    try
    {
        num_primitive_ = 0;
        num_node_ = 0;
        list_offset_primitive_ = {};
        list_offset_node_ = {};

        DeleteArray(backend_type_, nodes_);
        DeleteArray(backend_type_, primitives_);
        DeleteArray(backend_type_, list_pdf_area_);
        DeleteArray(backend_type_, list_blas_);
        DeleteArray(backend_type_, instances_);
        DeleteElement(backend_type_, tlas_);

        CommitPrimitives();
        CommitInstances();
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when commit scene.\n\t" << e.what();
        throw MyException(oss.str());
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when commit scene.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Scene::CommitPrimitives()
{
    try
    {
        const uint32_t num_instance =
            static_cast<uint32_t>(list_info_instance_.size());
        for (uint32_t i = 0; i < num_instance; ++i)
        {
            switch (list_info_instance_[i].type)
            {
            case Instance::Type::kCube:
                CommitCube(i);
                break;
            case Instance::Type::kSphere:
                CommitSphere(i);
                break;
            case Instance::Type::kRectangle:
                CommitRectangle(i);
                break;
            case Instance::Type::kMeshes:
                CommitMeshes(i);
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
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when commit geometry.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Scene::CommitInstances()
{
    try
    {
        //
        // 生成顶层加速结构的节点
        //
        const uint32_t num_instance =
            static_cast<uint32_t>(list_info_instance_.size());
        std::vector<AABB> aabbs(num_instance);
        std::vector<float> areas(num_instance);
        for (uint32_t i = 0; i < num_instance; ++i)
        {
            const uint64_t index = list_offset_node_[i];
            aabbs[i] = nodes_[index].aabb;
            areas[i] = nodes_[index].area;
        }

        list_pdf_area_ = MallocArray(backend_type_, areas);
        for (uint32_t i = 0; i < num_instance; ++i)
            list_pdf_area_[i] = 1.0f / list_pdf_area_[i];

        std::vector<BvhNode> list_node = BvhBuilder::Build(aabbs, areas);
        const uint64_t num_node_local = list_node.size();
        BvhNode *nodes =
            MallocArray<BvhNode>(backend_type_, num_node_ + num_node_local);
        for (uint64_t i = 0; i < num_node_local; ++i)
            nodes[i] = list_node[i];
        for (uint64_t i = 0; i < num_node_; ++i)
            nodes[num_node_local + i] = nodes_[i];
        DeleteArray(backend_type_, nodes_);
        nodes_ = nodes;
        for (uint32_t i = 0; i < num_instance; ++i)
            list_offset_node_[i] += num_node_local;

        //
        // 生成底层加速结构和实例
        //
        instances_ = MallocArray<Instance>(backend_type_, num_instance);
        list_blas_ = MallocArray<BLAS>(backend_type_, num_instance);
        for (uint32_t i = 0; i < num_instance; ++i)
        {
            list_blas_[i] = BLAS(list_offset_node_[i], nodes_,
                                 list_offset_primitive_[i], primitives_);
            instances_[i] = Instance(i, list_blas_);
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
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when commit geometry.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Scene::CommitCube(const uint32_t id)
{
    Instance::Info::Meshes &info_meshes = list_info_instance_[id].meshes;
    info_meshes.texcoords = {{0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1},
                             {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0},
                             {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1},
                             {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}};
    info_meshes.positions = {
        {1, -1, -1}, {1, -1, 1},  {-1, -1, 1},  {-1, -1, -1}, {1, 1, -1},
        {-1, 1, -1}, {-1, 1, 1},  {1, 1, 1},    {1, -1, -1},  {1, 1, -1},
        {1, 1, 1},   {1, -1, 1},  {1, -1, 1},   {1, 1, 1},    {-1, 1, 1},
        {-1, -1, 1}, {-1, -1, 1}, {-1, 1, 1},   {-1, 1, -1},  {-1, -1, -1},
        {1, 1, -1},  {1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}};
    info_meshes.normals = {
        {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, 1, 0},  {0, 1, 0},
        {0, 1, 0},  {0, 1, 0},  {1, 0, 0},  {1, 0, 0},  {1, 0, 0},  {1, 0, 0},
        {0, 0, 1},  {0, 0, 1},  {0, 0, 1},  {0, 0, 1},  {-1, 0, 0}, {-1, 0, 0},
        {-1, 0, 0}, {-1, 0, 0}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}};
    info_meshes.indices = {{0, 1, 2},    {3, 0, 2},    {4, 5, 6},
                           {7, 4, 6},    {8, 9, 10},   {11, 8, 10},
                           {12, 13, 14}, {15, 12, 14}, {16, 17, 18},
                           {19, 16, 18}, {20, 21, 22}, {23, 20, 22}};
    try
    {
        CommitMeshes(id);
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when add 'cube' instance.\n\t" << e.what();
        throw MyException(oss.str());
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when add 'cube' instance.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Scene::CommitSphere(const uint32_t id)
{
    try
    {
        const Instance::Info::Sphere &info_sphere =
            list_info_instance_[id].sphere;
        const Mat4 to_world = list_info_instance_[id].to_world;

        PrimitiveData data_primitive;
        data_primitive.type = PrimitiveType::kSphere;
        data_primitive.sphere.radius = info_sphere.radius;
        data_primitive.sphere.center = info_sphere.center;
        data_primitive.sphere.to_world = to_world;

        Primitive *primitives =
            MallocArray<Primitive>(backend_type_, num_primitive_ + 1);
        CopyArray(backend_type_, primitives, primitives_, num_primitive_);
        DeleteArray(backend_type_, primitives_);
        primitives[num_primitive_] = Primitive(0, data_primitive);
        std::vector<AABB> aabbs = {primitives[num_primitive_].aabb()};
        primitives_ = primitives;
        list_offset_primitive_.push_back(num_primitive_);
        ++num_primitive_;

        const Vec3 center_world = TransformPoint(to_world, info_sphere.center),
                   boundary_local = info_sphere.center +
                                    Vec3{info_sphere.radius, 0.0f, 0.0f},
                   boundary_world = TransformPoint(to_world, boundary_local);
        const float radius_world = Length(center_world - boundary_world);
        std::vector<float> areas = {4.0f * kPi * Sqr(radius_world)};

        std::vector<BvhNode> list_node = BvhBuilder::Build(aabbs, areas);
        const uint64_t num_node_local = list_node.size();
        BvhNode *nodes =
            MallocArray<BvhNode>(backend_type_, num_node_ + num_node_local);
        CopyArray(backend_type_, nodes, nodes_, num_node_);
        DeleteArray(backend_type_, nodes_);
        for (uint64_t i = 0; i < num_node_local; ++i)
            nodes[num_node_ + i] = list_node[i];
        nodes_ = nodes;
        list_offset_node_.push_back(num_node_);
        num_node_ += num_node_local;
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when add 'sphere' instance.\n\t" << e.what();
        throw MyException(oss.str());
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when add 'sphere' instance.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Scene::CommitRectangle(const uint32_t id)
{
    Instance::Info::Meshes &info_meshes = list_info_instance_[id].meshes;
    info_meshes.texcoords = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    info_meshes.positions = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
    info_meshes.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    info_meshes.indices = {{0, 1, 2}, {2, 3, 0}};
    try
    {
        CommitMeshes(id);
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when add 'reactangle' instance.\n\t" << e.what();
        throw MyException(oss.str());
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when add 'reactangle' instance.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Scene::CommitMeshes(const uint32_t id)
{
    Instance::Info::Meshes &info_meshes = list_info_instance_[id].meshes;
    const Mat4 to_world = list_info_instance_[id].to_world;

    if (info_meshes.indices.empty())
    {
        throw MyException(
            "cannot find vertex index info when adding instance to scene.");
    }

    if (info_meshes.positions.empty())
    {
        throw MyException("cannot find vertex position info when adding "
                             "instance to scene.");
    }

    for (Vec3 &position : info_meshes.positions)
        position = TransformPoint(to_world, position);

    if (!info_meshes.normals.empty())
    {
        const Mat4 normal_to_world = to_world.Transpose().Inverse();
        for (Vec3 &normal : info_meshes.normals)
            normal = TransformVector(normal_to_world, normal);
    }

    if (!info_meshes.tangents.empty())
    {
        for (Vec3 &tangent : info_meshes.tangents)
            tangent = TransformVector(to_world, tangent);
    }

    if (!info_meshes.bitangents.empty())
    {
        for (Vec3 &bitangent : info_meshes.bitangents)
            bitangent = TransformVector(to_world, bitangent);
    }

    try
    {
        std::vector<PrimitiveData> list_data_primitve;
        std::vector<float> areas;
        SetupMeshes(info_meshes, &list_data_primitve, &areas);
        const uint32_t num_primitive_local =
            static_cast<uint32_t>(list_data_primitve.size());

        Primitive *primitives = MallocArray<Primitive>(
            backend_type_, num_primitive_ + num_primitive_local);
        CopyArray(backend_type_, primitives, primitives_, num_primitive_);
        DeleteArray(backend_type_, primitives_);
        std::vector<AABB> aabbs(num_primitive_local);
        for (uint32_t i = 0; i < num_primitive_local; ++i)
        {
            primitives[num_primitive_ + i] =
                Primitive(i, list_data_primitve[i]);
            aabbs[i] = primitives[num_primitive_ + i].aabb();
        }
        primitives_ = primitives;
        list_offset_primitive_.push_back(num_primitive_);
        num_primitive_ += num_primitive_local;

        std::vector<BvhNode> list_node = BvhBuilder::Build(aabbs, areas);
        const uint64_t num_node_local = list_node.size();
        BvhNode *nodes =
            MallocArray<BvhNode>(backend_type_, num_node_ + num_node_local);
        CopyArray(backend_type_, nodes, nodes_, num_node_);
        DeleteArray(backend_type_, nodes_);
        for (uint64_t i = 0; i < num_node_local; ++i)
            nodes[num_node_ + i] = list_node[i];
        nodes_ = nodes;
        list_offset_node_.push_back(num_node_);
        num_node_ += num_node_local;
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when add 'meshes' instance.\n\t" << e.what();
        throw MyException(oss.str());
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when add 'meshes' instance.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Scene::SetupMeshes(Instance::Info::Meshes info_meshes,
                        std::vector<PrimitiveData> *list_data_primitve,
                        std::vector<float> *areas)
{
    try
    {
        const uint32_t num_primitive_local =
            static_cast<uint32_t>(info_meshes.indices.size());
        *list_data_primitve = std::vector<PrimitiveData>(num_primitive_local);
        *areas = std::vector<float>(num_primitive_local);
        for (uint32_t i = 0; i < num_primitive_local; ++i)
        {
            (*list_data_primitve)[i].type = PrimitiveType::kTriangle;
            const Uvec3 indices = info_meshes.indices[i];
            TriangleData &triangle = (*list_data_primitve)[i].triangle;

            if (info_meshes.texcoords.empty())
            {
                triangle.texcoords[0] = {0, 0};
                triangle.texcoords[1] = {1, 0};
                triangle.texcoords[2] = {1, 1};
            }
            else
            {
                for (int j = 0; j < 3; ++j)
                    triangle.texcoords[j] = info_meshes.texcoords[indices[j]];
            }

            for (int j = 0; j < 3; ++j)
                triangle.positions[j] = info_meshes.positions[indices[j]];

            const Vec3 v0v1 = triangle.positions[1] - triangle.positions[0],
                       v0v2 = triangle.positions[2] - triangle.positions[0];
            const Vec3 normal_geom = Cross(v0v1, v0v2);
            (*areas)[i] = Length(normal_geom);

            if (info_meshes.normals.empty())
            {
                const Vec3 normal = Normalize(normal_geom);
                for (int j = 0; j < 3; ++j)
                    triangle.normals[j] = normal;
            }
            else
            {
                for (int j = 0; j < 3; ++j)
                    triangle.normals[j] = info_meshes.normals[indices[j]];
            }

            if (info_meshes.tangents.empty() && info_meshes.bitangents.empty())
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
            else if (info_meshes.tangents.empty())
            {
                for (int j = 0; j < 3; ++j)
                {
                    triangle.bitangents[j] = info_meshes.bitangents[indices[j]];
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
                    triangle.tangents[j] = info_meshes.tangents[indices[j]];
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
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when add 'meshes' instance.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

} // namespace csrt