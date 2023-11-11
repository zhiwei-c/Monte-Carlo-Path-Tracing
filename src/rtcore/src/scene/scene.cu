#include "scene.cuh"

#include <exception>

namespace rt
{

Scene::Scene(const BackendType backend_type)
    : backend_type_(backend_type), tlas_(nullptr), instances_(nullptr),
      list_pdf_area_(nullptr), num_primitive_(0), num_node_(0),
      primitive_buffer_(nullptr), node_buffer_(nullptr), blas_buffer_(nullptr)
{
}

Scene::~Scene()
{
    DeleteArray(backend_type_, primitive_buffer_);
    DeleteArray(backend_type_, node_buffer_);
    DeleteArray(backend_type_, blas_buffer_);
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
            throw std::exception("unknow instance type.");
            break;
        }
        list_info_instance_.push_back(info);
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when add instance to scene.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
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

        DeleteArray(backend_type_, primitive_buffer_);
        DeleteArray(backend_type_, node_buffer_);
        DeleteArray(backend_type_, blas_buffer_);
        DeleteArray(backend_type_, instances_);
        DeleteElement(backend_type_, tlas_);

        CommitPrimitives();
        CommitInstances();
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when commit scene.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
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
                CommitCube(i, list_info_instance_[i].cube);
                break;
            case Instance::Type::kSphere:
                CommitSphere(i, list_info_instance_[i].sphere);
                break;
            case Instance::Type::kRectangle:
                CommitRectangle(i, list_info_instance_[i].rectangle);
                break;
            case Instance::Type::kMeshes:
                CommitMeshes(i, list_info_instance_[i].meshes);
                break;
            default:
                throw std::exception("unknow instance type.");
                break;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when commit geometry.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
    }
}

void rt::Scene::CommitInstances()
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
            aabbs[i] = node_buffer_[list_offset_node_[i]].aabb;
            areas[i] = node_buffer_[list_offset_node_[i]].area;
        }

        DeleteArray(backend_type_, list_pdf_area_);
        list_pdf_area_ = MallocArray(backend_type_, areas);
        for (uint32_t i = 0; i < num_instance; ++i)
            list_pdf_area_[i] = 1.0f / list_pdf_area_[i];

        std::vector<BvhNode> nodes = BvhBuilder::Build(aabbs, areas);
        const uint64_t num_node_local = nodes.size();
        BvhNode *node_buffer =
            MallocArray<BvhNode>(backend_type_, num_node_ + num_node_local);
        for (uint64_t i = 0; i < num_node_local; ++i)
            node_buffer[i] = nodes[i];
        for (uint64_t i = 0; i < num_node_; ++i)
            node_buffer[num_node_local + i] = node_buffer_[i];
        DeleteArray(backend_type_, node_buffer_);
        node_buffer_ = node_buffer;

        for (uint32_t i = 0; i < num_instance; ++i)
            list_offset_node_[i] += num_node_local;
        list_offset_node_.push_back(0);

        num_node_ += num_node_local;

        //
        // 生成底层加速结构和实例
        //

        blas_buffer_ = MallocArray<BLAS>(backend_type_, num_instance);
        instances_ = MallocArray<Instance>(backend_type_, num_instance);
        for (uint32_t i = 0; i < num_instance; ++i)
        {
            blas_buffer_[i] =
                BLAS(list_offset_node_[i], node_buffer_,
                     list_offset_primitive_[i], primitive_buffer_);
            instances_[i] = Instance(i, blas_buffer_);
        }

        tlas_ = MallocElement<TLAS>(backend_type_);
        *tlas_ =
            TLAS(list_offset_node_[num_instance], node_buffer_, instances_);
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when commit geometry.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
    }
}

void Scene::CommitCube(const uint32_t id, const Instance::Info::Cube &info)
{
    Instance::Info::Meshes info_meshes;
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
    info_meshes.to_world = info.to_world;
    try
    {
        CommitMeshes(id, info_meshes);
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when add 'cube' instance.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
    }
}

void Scene::CommitSphere(const uint32_t id, const Instance::Info::Sphere &info)
{
    try
    {
        Primitive::Data data_primitive;
        data_primitive.type = Primitive::Type::kSphere;
        data_primitive.sphere.radius = info.radius;
        data_primitive.sphere.center = info.center;
        data_primitive.sphere.to_world = info.to_world;
        data_primitive.sphere.normal_to_world =
            info.to_world.Transpose().Inverse();
        data_primitive.sphere.to_local = info.to_world.Inverse();

        Primitive *primitive_buffer =
            MallocArray<Primitive>(backend_type_, num_primitive_ + 1);
        CopyArray(backend_type_, primitive_buffer, primitive_buffer_,
                  num_primitive_);
        DeleteArray(backend_type_, primitive_buffer_);
        primitive_buffer[num_primitive_] = Primitive(0, data_primitive);
        std::vector<AABB> aabbs = {primitive_buffer[num_primitive_].aabb()};
        primitive_buffer_ = primitive_buffer;
        list_offset_primitive_.push_back(num_primitive_);
        ++num_primitive_;

        const float radius_world =
            Length(TransformPoint(info.to_world, info.center) -
                   TransformPoint(info.to_world,
                                  info.center + Vec3{info.radius, 0.0f, 0.0f}));
        std::vector<float> areas = {4.0f * kPi * Sqr(radius_world)};

        std::vector<BvhNode> nodes = BvhBuilder::Build(aabbs, areas);
        const uint64_t num_node_local = nodes.size();
        BvhNode *node_buffer =
            MallocArray<BvhNode>(backend_type_, num_node_ + num_node_local);
        CopyArray(backend_type_, node_buffer, node_buffer_, num_node_);
        DeleteArray(backend_type_, node_buffer_);
        for (uint64_t i = 0; i < num_node_local; ++i)
            node_buffer[num_node_ + i] = nodes[i];
        node_buffer_ = node_buffer;
        list_offset_node_.push_back(num_node_);
        num_node_ += num_node_local;
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when add 'sphere' instance.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
    }
}

void Scene::CommitRectangle(const uint32_t id,
                            const Instance::Info::Rectangle &info)
{
    Instance::Info::Meshes info_meshes;
    info_meshes.texcoords = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    info_meshes.positions = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
    info_meshes.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    info_meshes.indices = {{0, 1, 2}, {2, 3, 0}};
    info_meshes.to_world = info.to_world;
    try
    {
        CommitMeshes(id, info_meshes);
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when add 'reactangle' instance.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
    }
}

void Scene::CommitMeshes(const uint32_t id, Instance::Info::Meshes info)
{
    if (info.indices.empty())
    {
        throw std::exception(
            "cannot find vertex index info when adding instance to scene.");
    }

    if (info.positions.empty())
    {
        throw std::exception("cannot find vertex position info when adding "
                             "instance to scene.");
    }

    for (Vec3 &position : info.positions)
        position = TransformPoint(info.to_world, position);

    if (!info.normals.empty())
    {
        const Mat4 normal_to_world = info.to_world.Transpose().Inverse();
        for (Vec3 &normal : info.normals)
            normal = TransformVector(normal_to_world, normal);
    }

    if (!info.tangents.empty())
    {
        for (Vec3 &tangent : info.tangents)
            tangent = TransformVector(info.to_world, tangent);
    }

    if (!info.bitangents.empty())
    {
        for (Vec3 &bitangent : info.bitangents)
            bitangent = TransformVector(info.to_world, bitangent);
    }

    try
    {
        std::vector<Primitive::Data> data_primitves;
        std::vector<float> areas;
        SetupMeshes(info, &data_primitves, &areas);
        const uint32_t num_primitive_local =
            static_cast<uint32_t>(data_primitves.size());

        Primitive *primitive_buffer = MallocArray<Primitive>(
            backend_type_, num_primitive_ + num_primitive_local);
        CopyArray(backend_type_, primitive_buffer, primitive_buffer_,
                  num_primitive_);
        DeleteArray(backend_type_, primitive_buffer_);
        primitive_buffer_ = primitive_buffer;

        std::vector<AABB> aabbs(num_primitive_local);
        for (uint32_t i = 0; i < num_primitive_local; ++i)
        {
            primitive_buffer_[num_primitive_ + i] =
                Primitive(i, data_primitves[i]);
            aabbs[i] = primitive_buffer_[num_primitive_ + i].aabb();
        }
        list_offset_primitive_.push_back(num_primitive_);
        num_primitive_ += num_primitive_local;

        std::vector<BvhNode> nodes = BvhBuilder::Build(aabbs, areas);
        const uint64_t num_node_local = nodes.size();
        BvhNode *node_buffer =
            MallocArray<BvhNode>(backend_type_, num_node_ + num_node_local);
        CopyArray(backend_type_, node_buffer, node_buffer_, num_node_);
        DeleteArray(backend_type_, node_buffer_);
        for (uint64_t i = 0; i < num_node_local; ++i)
            node_buffer[num_node_ + i] = nodes[i];
        node_buffer_ = node_buffer;
        list_offset_node_.push_back(num_node_);
        num_node_ += num_node_local;
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when add 'meshes' instance.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
    }
}

void Scene::SetupMeshes(Instance::Info::Meshes info,
                        std::vector<Primitive::Data> *data_primitves,
                        std::vector<float> *areas)
{
    try
    {
        const uint32_t num_primitive_local =
            static_cast<uint32_t>(info.indices.size());
        *data_primitves = std::vector<Primitive::Data>(num_primitive_local);
        *areas = std::vector<float>(num_primitive_local);
        for (uint32_t i = 0; i < num_primitive_local; ++i)
        {
            (*data_primitves)[i].type = Primitive::Type::kTriangle;
            const Uvec3 index = info.indices[i];
            Primitive::Data::Triangle &triangle = (*data_primitves)[i].triangle;

            if (info.texcoords.empty())
            {
                triangle.texcoords[0] = {0, 0}, triangle.texcoords[1] = {1, 0},
                triangle.texcoords[2] = {1, 1};
            }
            else
            {
                for (int j = 0; j < 3; ++j)
                    triangle.texcoords[j] = info.texcoords[index[j]];
            }

            for (int j = 0; j < 3; ++j)
                triangle.positions[j] = info.positions[index[j]];

            triangle.v0v1 = triangle.positions[1] - triangle.positions[0],
            triangle.v0v2 = triangle.positions[2] - triangle.positions[0];
            const Vec3 normal_geom = Cross(triangle.v0v1, triangle.v0v2);
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
                    triangle.normals[j] = info.normals[index[j]];
            }

            if (info.tangents.empty() && info.bitangents.empty())
            {
                const Vec2 uv_delta_01 =
                               triangle.texcoords[1] - triangle.texcoords[0],
                           uv_delta_02 =
                               triangle.texcoords[2] - triangle.texcoords[0];
                const float r = 1.0f / (uv_delta_02.x * uv_delta_01.y -
                                        uv_delta_01.x * uv_delta_02.y);
                const Vec3 tangent = Normalize((uv_delta_01.v * triangle.v0v2 -
                                                uv_delta_02.v * triangle.v0v1) *
                                               r),
                           bitangent =
                               Normalize((uv_delta_02.u * triangle.v0v1 -
                                          uv_delta_01.u * triangle.v0v2) *
                                         r);
                for (int j = 0; j < 3; ++j)
                {
                    triangle.tangents[j] = tangent;
                    triangle.bitangents[j] = bitangent;
                }
            }
            else if (info.tangents.empty())
            {
                for (int j = 0; j < 3; ++j)
                {
                    triangle.bitangents[j] = info.bitangents[index[j]];
                    triangle.tangents[j] = Normalize(
                        Cross(triangle.bitangents[j], triangle.normals[j]));
                }
            }
            else if (info.bitangents.empty())
            {
                for (int j = 0; j < 3; ++j)
                {
                    triangle.tangents[j] = info.tangents[index[j]];
                    triangle.bitangents[j] = Normalize(
                        Cross(triangle.normals[j], triangle.tangents[j]));
                }
            }
            else
            {
                for (int j = 0; j < 3; ++j)
                {
                    triangle.tangents[j] = info.tangents[index[j]];
                    triangle.bitangents[j] = info.bitangents[index[j]];
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when add 'meshes' instance.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
    }
}

} // namespace rt