#include "scene.cuh"

#include <exception>

namespace rt
{

Scene::Scene(const BackendType backend_type)
    : tlas_(nullptr), instance_buffer_(nullptr), backend_type_(backend_type)
{
}

Scene::~Scene()
{
    for (Primitive *&primitives : primitives_)
        DeleteArray(backend_type_, primitives);

    for (BvhNode *&node : nodes_)
        DeleteArray(backend_type_, node);

    for (BLAS *&blas : blas_buffer_)
        DeleteElement(backend_type_, blas);

    for (Instance *&instance : instances_)
        DeleteElement(backend_type_, instance);

    DeleteArray(backend_type_, instance_buffer_);
    DeleteElement(backend_type_, tlas_);
}

void Scene::AddInstance(const Instance::Info &info)
{
    try
    {
        switch (info.type)
        {
        case Instance::Type::kCube:
            AddCube(info.cube);
            break;
        case Instance::Type::kSphere:
            AddSphere(info.sphere);
            break;
        case Instance::Type::kRectangle:
            AddRectangle(info.rectangle);
            break;
        case Instance::Type::kMeshes:
            AddMeshes(info.meshes);
            break;
        default:
            throw std::exception("unknow instance type.");
            break;
        }
    }
    catch (const std::exception &e)
    {
        throw e;
    }
}

TLAS *Scene::Commit()
{
    const size_t num = instances_.size();
    DeleteArray(backend_type_, instance_buffer_);
    instance_buffer_ = MallocArray<Instance>(backend_type_, num);
    for (size_t i = 0; i < num; ++i)
        instance_buffer_[i] = *instances_[i];

    std::vector<AABB> aabbs(num);
    std::vector<float> areas(num);
    for (size_t i = 0; i < num; ++i)
    {
        aabbs[i] = nodes_[i][0].aabb;
        areas[i] = nodes_[i][0].area;
    }
    std::vector<BvhNode> nodes_temp;
    try
    {
        nodes_temp = BvhBuilder::Build(aabbs, areas);
    }
    catch (const std::exception &e)
    {
        throw e;
    }
    BvhNode *nodes = MallocArray<BvhNode>(backend_type_, nodes_temp);
    nodes_.push_back(nodes);

    DeleteElement(backend_type_, tlas_);
    tlas_ = MallocElement<TLAS>(backend_type_);
    *tlas_ = TLAS(nodes, instance_buffer_);

    return tlas_;
}

void Scene::AddCube(const Instance::Info::Cube &info)
{
    Instance::Info::Meshes info_meshes;
    info_meshes.texcoords = {{0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0},
                             {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0},
                             {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}};
    info_meshes.positions = {{1, -1, -1}, {1, -1, 1},  {-1, -1, 1},  {-1, -1, -1}, {1, 1, -1},
                             {-1, 1, -1}, {-1, 1, 1},  {1, 1, 1},    {1, -1, -1},  {1, 1, -1},
                             {1, 1, 1},   {1, -1, 1},  {1, -1, 1},   {1, 1, 1},    {-1, 1, 1},
                             {-1, -1, 1}, {-1, -1, 1}, {-1, 1, 1},   {-1, 1, -1},  {-1, -1, -1},
                             {1, 1, -1},  {1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}};
    info_meshes.normals = {{0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, 1, 0},  {0, 1, 0},
                           {0, 1, 0},  {0, 1, 0},  {1, 0, 0},  {1, 0, 0},  {1, 0, 0},  {1, 0, 0},
                           {0, 0, 1},  {0, 0, 1},  {0, 0, 1},  {0, 0, 1},  {-1, 0, 0}, {-1, 0, 0},
                           {-1, 0, 0}, {-1, 0, 0}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}};
    info_meshes.indices = {{0, 1, 2},    {3, 0, 2},    {4, 5, 6},    {7, 4, 6},
                           {8, 9, 10},   {11, 8, 10},  {12, 13, 14}, {15, 12, 14},
                           {16, 17, 18}, {19, 16, 18}, {20, 21, 22}, {23, 20, 22}};
    info_meshes.to_world = info.to_world;
    try
    {
        AddMeshes(info_meshes);
    }
    catch (const std::exception &e)
    {
        throw e;
    }
}

void Scene::AddSphere(const Instance::Info::Sphere &info)
{
    Primitive::Info info_primitive;
    info_primitive.type = Primitive::Type::kSphere;
    info_primitive.sphere.radius = info.radius;
    info_primitive.sphere.center = info.center;
    info_primitive.sphere.to_world = info.to_world;
    info_primitive.sphere.normal_to_world = info.to_world.Transpose().Inverse();
    info_primitive.sphere.to_local = info.to_world.Inverse();

    Primitive *primitive = MallocArray<Primitive>(backend_type_, 1);
    *primitive = Primitive(0, info_primitive);
    primitives_.push_back(primitive);

    std::vector<AABB> aabbs = {primitive->aabb()};

    const float radius_world =
        Length(Mul(info.to_world, {info.center, 1.0f}).position() -
               Mul(info.to_world, {info.center + Vec3{1.0f, 0.0f, 0.0f}, 1.0f}).position());
    std::vector<float> areas = {4.0f * kPi * Sqr(radius_world)};

    std::vector<BvhNode> nodes_temp;
    try
    {
        nodes_temp = BvhBuilder::Build(aabbs, areas);
    }
    catch (const std::exception &e)
    {
        throw e;
    }
    BvhNode *nodes = MallocArray<BvhNode>(backend_type_, nodes_temp);
    nodes_.push_back(nodes);

    BLAS *blas = MallocElement<BLAS>(backend_type_);
    *blas = BLAS(nodes, primitive);
    blas_buffer_.push_back(blas);

    Instance *instance = MallocElement<Instance>(backend_type_);
    const uint32_t id_instance = static_cast<uint32_t>(instances_.size());
    *instance = Instance(id_instance, blas);
    instances_.push_back(instance);
}

void Scene::AddRectangle(const Instance::Info::Rectangle &info)
{
    Instance::Info::Meshes info_meshes;
    info_meshes.texcoords = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    info_meshes.positions = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
    info_meshes.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    info_meshes.indices = {{0, 1, 2}, {2, 3, 0}};
    info_meshes.to_world = info.to_world;
    try
    {
        AddMeshes(info_meshes);
    }
    catch (const std::exception &e)
    {
        throw e;
    }
}

void Scene::AddMeshes(Instance::Info::Meshes info)
{
    if (info.indices.empty())
        throw std::exception("cannot find vertex index info when adding instance to scene.");

    if (info.positions.empty())
        throw std::exception("cannot find vertex position info when adding instance to scene.");
    for (Vec3 &position : info.positions)
        position = Mul(info.to_world, {position, 1.0f}).position();

    if (!info.normals.empty())
    {
        const Mat4 normal_to_world = info.to_world.Transpose().Inverse();
        for (Vec3 &normal : info.normals)
            normal = Mul(normal_to_world, {normal, 0.0f}).direction();
    }

    if (!info.tangents.empty())
    {
        for (Vec3 &tangent : info.tangents)
            tangent = Mul(info.to_world, {tangent, 0.0f}).direction();
    }

    if (!info.bitangents.empty())
    {
        for (Vec3 &bitangent : info.bitangents)
            bitangent = Mul(info.to_world, {bitangent, 0.0f}).direction();
    }

    const size_t num = info.indices.size();
    std::vector<Primitive::Info> info_primitves(num);
    std::vector<float> areas(num);
    for (size_t i = 0; i < num; ++i)
    {
        info_primitves[i].type = Primitive::Type::kTriangle;
        const Uvec3 index = info.indices[i];
        Primitive::Info::Triangle &triangle = info_primitves[i].triangle;
        for (int j = 0; j < 3; ++j)
        {
            triangle.texcoords[j] = info.texcoords[index[j]];
            triangle.positions[j] = info.positions[index[j]];
        }

        triangle.v0v1 = triangle.positions[1] - triangle.positions[0],
        triangle.v0v2 = triangle.positions[2] - triangle.positions[0];
        const Vec3 normal_geom = Cross(triangle.v0v1, triangle.v0v2);
        areas[i] = Length(normal_geom);

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
            const Vec2 uv_delta_01 = triangle.texcoords[1] - triangle.texcoords[0],
                       uv_delta_02 = triangle.texcoords[2] - triangle.texcoords[0];
            const float r = 1.0f / (uv_delta_02.x * uv_delta_01.y - uv_delta_01.x * uv_delta_02.y);
            const Vec3 tangent = Normalize(
                           (uv_delta_01.v * triangle.v0v2 - uv_delta_02.v * triangle.v0v1) * r),
                       bitangent = Normalize(
                           (uv_delta_02.u * triangle.v0v1 - uv_delta_01.u * triangle.v0v2) * r);
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
                triangle.tangents[j] =
                    Normalize(Cross(triangle.bitangents[j], triangle.normals[j]));
            }
        }
        else if (info.bitangents.empty())
        {
            for (int j = 0; j < 3; ++j)
            {
                triangle.tangents[j] = info.tangents[index[j]];
                triangle.bitangents[j] =
                    Normalize(Cross(triangle.normals[j], triangle.tangents[j]));
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

    Primitive *primitives = MallocArray<Primitive>(backend_type_, num);
    std::vector<AABB> aabbs(num);
    for (size_t i = 0; i < num; ++i)
    {
        primitives[i] = Primitive(i, info_primitves[i]);
        aabbs[i] = primitives[i].aabb();
    }
    primitives_.push_back(primitives);

    std::vector<BvhNode> nodes_temp;
    try
    {
        nodes_temp = BvhBuilder::Build(aabbs, areas);
    }
    catch (const std::exception &e)
    {
        throw e;
    }
    BvhNode *nodes = MallocArray<BvhNode>(backend_type_, nodes_temp);
    nodes_.push_back(nodes);

    BLAS *blas = MallocElement<BLAS>(backend_type_);
    *blas = BLAS(nodes, primitives);
    blas_buffer_.push_back(blas);

    Instance *instance = MallocElement<Instance>(backend_type_);
    const uint32_t id_instance = static_cast<uint32_t>(instances_.size());
    *instance = Instance(id_instance, blas);
    instances_.push_back(instance);
}

} // namespace rt