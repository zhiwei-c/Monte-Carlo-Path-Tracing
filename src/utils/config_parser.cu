#include "config_parser.cuh"

#include <filesystem>
#include <algorithm>
#include <cstdio>

#include "math.cuh"
#include "model_loader.cuh"
#include "image_io.cuh"
#include "../bsdfs/ior.cuh"

namespace mitsuba_parser
{

    bool ReadBoolean(const pugi::xml_node &parent_node,
                     const std::vector<std::string> &valid_names, bool defalut_value);

    int ReadInt(const pugi::xml_node &parent_node, const std::vector<std::string> &valid_names,
                int defalut_value);

    float ReadFloat(const pugi::xml_node &parent_node,
                    const std::vector<std::string> &valid_names, float defalut_value);

    Vec3 ReadVec3(const pugi::xml_node &parent_node, const std::vector<std::string> &valid_names,
                  const Vec3 &defalut_value);

    Vec3 ReadVec3(const pugi::xml_node &vec3_node, const Vec3 &defalut_value,
                  const char *value_name);

    Mat4 ReadMat4(const pugi::xml_node &matrix_node);

    Mat4 ReadTransform4(const pugi::xml_node &transform_node);

    bool GetChildByName(const pugi::xml_node &parent_node,
                        const std::vector<std::string> &valid_names, pugi::xml_node *child_node);
} // namespace mitsuba_parser

SceneInfo ConfigParser::LoadConfig(const std::string &filename)
{
    info_ = {};

    if (filename.empty())
    {
        fprintf(stderr, "[warning] no config file, load default.\n");
        return LoadDefault();
    }
    std::filesystem::path filepath = filename;
    if (!std::filesystem::exists(filepath))
    {
        fprintf(stderr, "[error] cannot find config file: '%s', load default config instead.\n",
                filename.c_str());
        exit(1);
    }
    if (GetSuffix(filename) != "xml")
    {
        fprintf(stderr, "[error] only support mitsuba xml format config file.\n");
        exit(1);
    }
    config_file_directory_ = GetDirectory(filename);

    //
    // 解析 XML
    //
    fprintf(stderr, "[info] read config file: '%s'.\n", filename.c_str());
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(filename.c_str());
    if (!result)
    {
        fprintf(stderr, "[error] read config file failed.\n");
        exit(1);
    }
    pugi::xml_node scene_node = doc.child("scene");

    //
    // 解析一些预先定义的参数和相机参数
    //
    default_mp_.clear();
    for (pugi::xml_node node : scene_node.children("default"))
    {
        std::string name = node.attribute("name").value();
        std::string value = node.attribute("value").value();
        default_mp_["$" + name] = value;
    }
    ReadCamera(scene_node.child("sensor"));

    //
    // 解析除面光源外的其它光源
    //
    for (pugi::xml_node emitter_node : scene_node.children("emitter"))
        ReadEmitter(emitter_node);

    //
    // 解析纹理
    //
    id_texture_mp_.clear();
    for (pugi::xml_node texture_node : scene_node.children("texture"))
        ReadTexture(texture_node, 1.0f, 1.0f);

    //
    // 解析BSDF
    //
    id_bsdf_mp_.clear();
    for (pugi::xml_node bsdf_node : scene_node.children("bsdf"))
        ReadBsdf(bsdf_node, "", kInvalidId, kInvalidId, false);

    //
    // 解析物体
    //
    for (pugi::xml_node shape_node : scene_node.children("shape"))
        ReadShape(shape_node);

    return info_;
}

SceneInfo ConfigParser::LoadDefault()
{

    info_.texture_info_buffer = {
        // 0 - White
        Texture::Info::CreateConstant({0.725f, 0.71f, 0.68f}),
        // 1 - Green
        Texture::Info::CreateConstant({0.14f, 0.45f, 0.091f}),
        // 2 - Red
        Texture::Info::CreateConstant({0.63f, 0.065f, 0.05f}),
        // 3 - Light radiance
        Texture::Info::CreateConstant({17.0f, 12.0f, 4.0f}),
        // 4 - 1.0f
        Texture::Info::CreateConstant({1.0f}),
        // 5 - 0.001f
        Texture::Info::CreateConstant({0.01f}),
        // 6 - 0.1f
        Texture::Info::CreateConstant({0.1f}),
        // 7 - 0.3f
        Texture::Info::CreateConstant({0.3f}),
    };

    Vec3 eta_ag, k_ag;
    ior_lut::LookupConductorIor("Ag", &eta_ag, &k_ag);
    Vec3 eta_au, k_au;
    ior_lut::LookupConductorIor("Au", &eta_au, &k_au);
    float eta_bk7;
    ior_lut::LookupDielectricIor("bk7", &eta_bk7);

    info_.bsdf_info_buffer = {
        // 0 White
        Bsdf::Info::CreateDiffuse(0, true, kInvalidId, kInvalidId),
        // 1 Green
        Bsdf::Info::CreateDiffuse(1, true, kInvalidId, kInvalidId),
        // 2 Red
        Bsdf::Info::CreateDiffuse(2, true, kInvalidId, kInvalidId),
        // 3 Light
        Bsdf::Info::CreateAreaLight(3, true, kInvalidId, kInvalidId),
        // 4 Conductor
        Bsdf::Info::CreateConductor(5, 4, eta_ag, k_ag, true, kInvalidId, kInvalidId),
        // 5 Rough Conductor
        Bsdf::Info::CreateConductor(7, 4, eta_au, k_au, true, kInvalidId, kInvalidId),
        // 6 Dielectric
        Bsdf::Info::CreateDielectric(6, 4, 4, eta_bk7, false, true, kInvalidId, kInvalidId)};

    // 0 Floor
    CreateRectangle({{0, 1, 0, 0},
                     {0, 0, 2, 0},
                     {1, 0, 0, 0},
                     {0, 0, 0, 1}},
                    0);
    // 1 Ceiling
    CreateRectangle({{-1, 0, 0, 0},
                     {0, 0, -2, 2},
                     {0, -1, 0, 0},
                     {0, 0, 0, 1}},
                    0);
    // 2 BackWall
    CreateRectangle({{0, 1, 0, 0},
                     {1, 0, 0, 1},
                     {0, 0, -2, -1},
                     {0, 0, 0, 1}},
                    0);
    // 3 RightWall
    CreateRectangle({{0, 0, 2, 1},
                     {1, 0, 0, 1},
                     {0, 1, 0, 0},
                     {0, 0, 0, 1}},
                    1);
    // 4 LeftWall
    CreateRectangle({{0, 0, -2, -1},
                     {1, 0, 0, 1},
                     {0, -1, 0, 0},
                     {0, 0, 0, 1}},
                    2);
    // 5 Light
    CreateRectangle({{0.235, 0, 0, -0.005},
                     {0, 0, -0.0893, 1.98},
                     {0, 0.19, 0, -0.03},
                     {0, 0, 0, 1}},
                    3);

    // 6 Tall Box
    CreateCube({{0.286776, 0.098229, 0, -0.335439},
                {0, 0, -0.6, 0.6},
                {-0.0997984, 0.282266, 0, -0.291415},
                {0, 0, 0, 1}},
               4);

    // 7 Short Box
    CreateCube({{0.0851643, 0.289542, 0, 0.328631},
                {0, 0, -0.3, 0.3},
                {-0.284951, 0.0865363, 0, 0.374592},
                {0, 0, 0, 1}},
               5);

    CreateSphere({0.328631014f, 0.75f, 0.374592006f}, 0.15f, Mat4(), 6);

    return info_;
}

void ConfigParser::ReadCamera(pugi::xml_node sensor_node)
{
    std::string type = sensor_node.attribute("type").as_string();
    if (type != "perspective")
    {
        fprintf(stderr, "[error] only support 'perspective' sensor.\n");
        exit(1);
    }

    float focal_length = 50.0f;
    std::string fov_axis = "x";
    for (pugi::xml_node node : sensor_node.children("string"))
    {
        switch (Hash(node.attribute("name").value()))
        {
        case "focalLength"_hash:
        {
            std::string foval_length_str = node.attribute("value").as_string();
            foval_length_str = foval_length_str.substr(0, foval_length_str.size() - 2);
            focal_length = std::stof(foval_length_str);
            break;
        }
        case "fovAxis"_hash:
        {
            fov_axis = node.attribute("value").as_string();
        }
        }
    }

    uint64_t width = 768, height = 576;
    for (pugi::xml_node node : sensor_node.child("film").children("integer"))
    {
        switch (Hash(node.attribute("name").value()))
        {
        case "width"_hash:
        {
            std::string value_str = node.attribute("value").value();
            if (!value_str.empty())
            {
                if (value_str[0] == '$')
                {
                    if (!default_mp_.count(value_str))
                    {
                        fprintf(stderr, "[error] cannot find '%s' from config file.\n",
                                value_str.c_str());
                        exit(1);
                    }
                    value_str = default_mp_.at(value_str);
                }
                width = std::stoi(value_str);
            }
            break;
        }
        case "height"_hash:
        {
            std::string value_str = node.attribute("value").value();
            if (!value_str.empty())
            {
                if (value_str[0] == '$')
                {
                    if (!default_mp_.count(value_str))
                    {
                        fprintf(stderr, "[error] cannot find '%s' from config file.\n",
                                value_str.c_str());
                        exit(1);
                    }
                    value_str = default_mp_.at(value_str);
                }
                height = std::stoi(value_str);
            }
            break;
        }
        default:
            break;
        }
    }

    float fov_x = mitsuba_parser::ReadFloat(sensor_node, {"fov"}, -1.0f);
    switch (Hash(fov_axis.c_str()))
    {
    case "x"_hash:
    {
        if (fov_x <= 0.0f)
            fov_x = 2.0f * atanf(36.0f * 0.5f / focal_length) * 180.0f * kPiInv;
        break;
    }
    case "y"_hash:
    {
        if (fov_x <= 0.0f)
            fov_x = 2.0f * atanf(24.0f * 0.5f / focal_length) * 180.0f * kPiInv;
        fov_x = fov_x * width / height;
        break;
    }
    case "smaller"_hash:
    {
        if (width > height)
        {
            if (fov_x <= 0.0f)
                fov_x = 2.0f * atanf(24.0f * 0.5f / focal_length) * 180.0f * kPiInv;
            fov_x = fov_x * width / height;
        }
        break;
    }
    default:
    {
        fprintf(stderr, "[error] unsupport fov axis type '%s'\n", fov_axis.c_str());
        exit(1);
    }
    }

    uint32_t spp = 4;
    for (pugi::xml_node node : sensor_node.child("sampler").children("integer"))
    {
        switch (Hash(node.attribute("name").value()))
        {
        case "sampleCount"_hash:
        case "sample_count"_hash:
        {
            std::string value_str = node.attribute("value").value();
            if (!value_str.empty())
            {
                if (value_str[0] == '$')
                {
                    if (!default_mp_.count(value_str))
                    {
                        fprintf(stderr, "[error] cannot find '%s' from config file.\n",
                                value_str.c_str());
                        exit(1);
                    }
                    value_str = default_mp_.at(value_str);
                }
                spp = std::stoi(value_str);
            }
            break;
        }
        default:
            break;
        }
    }

    Vec3 eye = {0.0f, 0.0f, 0.0f},
         look_at = {0.0f, 0.0f, 1.0f},
         up = {0.0f, 1.0f, 0.0f};
    if (sensor_node.child("transform"))
    {
        const Mat4 to_world = mitsuba_parser::ReadTransform4(sensor_node.child("transform"));
        eye = TransfromPoint(to_world, eye);
        look_at = TransfromPoint(to_world, look_at);
        up = TransfromVector(to_world, up);
    }

    info_.camera = Camera(spp, width, height, fov_x, eye, look_at, up);
}

void ConfigParser::CreateSphere(const Vec3 &center, const float radius, const Mat4 &to_world,
                                const uint64_t id_bsdf)
{
    const uint64_t index_offset = info_.primitive_buffer.size(),
                   num_primitives = 1;
    info_.primitive_buffer.emplace_back(center, radius, to_world, id_bsdf);
    info_.instance_buffer.emplace_back(
        info_.instance_buffer.size(),
        info_.bsdf_info_buffer[id_bsdf].type == Bsdf::Type::kAreaLight, index_offset,
        num_primitives, info_.primitive_buffer.data());
}

void ConfigParser::CreateDisk(const Mat4 &to_world, const uint64_t id_bsdf)
{
    const uint64_t index_offset = info_.primitive_buffer.size(),
                   num_primitives = 1;
    info_.primitive_buffer.emplace_back(to_world, id_bsdf);
    info_.instance_buffer.emplace_back(
        info_.instance_buffer.size(),
        info_.bsdf_info_buffer[id_bsdf].type == Bsdf::Type::kAreaLight, index_offset,
        num_primitives, info_.primitive_buffer.data());
}

void ConfigParser::CreateRectangle(const Mat4 &to_world, const uint64_t id_bsdf)
{
    const Uvec3 indices[] = {{0, 1, 2}, {2, 3, 0}};
    std::vector<Vertex> vertices = {
        {{0, 0}, {-1, -1, 0}, {0, 0, 1}},
        {{1, 0}, {1, -1, 0}, {0, 0, 1}},
        {{1, 1}, {1, 1, 0}, {0, 0, 1}},
        {{0, 1}, {-1, 1, 0}, {0, 0, 1}},
    };

    const Mat4 normal_to_world = to_world.Transpose().Inverse();
    for (uint64_t i = 0; i < vertices.size(); ++i)
    {
        vertices[i].pos = TransfromPoint(to_world, vertices[i].pos);
        vertices[i].normal = TransfromVector(normal_to_world, vertices[i].normal);
    }

    const uint64_t index_offset = info_.primitive_buffer.size(),
                   num_primitives = sizeof(indices) / sizeof(Uvec3);
    Vertex v[3];
    for (int i = 0; i < num_primitives; ++i)
    {
        for (int j = 0; j < 3; ++j)
            v[j] = vertices[indices[i][j]];
        info_.primitive_buffer.emplace_back(v, id_bsdf);
    }

    info_.instance_buffer.emplace_back(
        info_.instance_buffer.size(),
        info_.bsdf_info_buffer[id_bsdf].type == Bsdf::Type::kAreaLight, index_offset,
        num_primitives, info_.primitive_buffer.data());
}

void ConfigParser::CreateCube(const Mat4 &to_world, const uint64_t id_bsdf)
{
    const Uvec3 indices[] = {
        // bottom
        {0, 1, 2},
        {3, 0, 2},
        // top
        {4, 5, 6},
        {7, 4, 6},
        // right
        {8, 9, 10},
        {11, 8, 10},
        // front
        {12, 13, 14},
        {15, 12, 14},
        // left
        {16, 17, 18},
        {19, 16, 18},
        // back
        {20, 21, 22},
        {23, 20, 22},
    };
    std::vector<Vertex> vertices = {
        // bottom
        {{0, 1}, {1, -1, -1}, {0, -1, 0}},
        {{1, 1}, {1, -1, 1}, {0, -1, 0}},
        {{1, 0}, {-1, -1, 1}, {0, -1, 0}},
        {{0, 0}, {-1, -1, -1}, {0, -1, 0}},
        // top
        {{0, 1}, {1, 1, -1}, {0, 1, 0}},
        {{1, 1}, {-1, 1, -1}, {0, 1, 0}},
        {{1, 0}, {-1, 1, 1}, {0, 1, 0}},
        {{0, 0}, {1, 1, 1}, {0, 1, 0}},
        // right
        {{0, 1}, {1, -1, -1}, {1, 0, 0}},
        {{1, 1}, {1, 1, -1}, {1, 0, 0}},
        {{1, 0}, {1, 1, 1}, {1, 0, 0}},
        {{0, 0}, {1, -1, 1}, {1, 0, 0}},
        // front
        {{0, 1}, {1, -1, 1}, {0, 0, 1}},
        {{1, 1}, {1, 1, 1}, {0, 0, 1}},
        {{1, 0}, {-1, 1, 1}, {0, 0, 1}},
        {{0, 0}, {-1, -1, 1}, {0, 0, 1}},
        // left
        {{0, 1}, {-1, -1, 1}, {-1, 0, 0}},
        {{1, 1}, {-1, 1, 1}, {-1, 0, 0}},
        {{1, 0}, {-1, 1, -1}, {-1, 0, 0}},
        {{0, 0}, {-1, -1, -1}, {-1, 0, 0}},
        // back
        {{0, 1}, {1, 1, -1}, {0, 0, -1}},
        {{1, 1}, {1, -1, -1}, {0, 0, -1}},
        {{1, 0}, {-1, -1, -1}, {0, 0, -1}},
        {{0, 0}, {-1, 1, -1}, {0, 0, -1}},
    };
    const Mat4 normal_to_world = to_world.Transpose().Inverse();
    for (uint64_t i = 0; i < vertices.size(); ++i)
    {
        vertices[i].pos = TransfromPoint(to_world, vertices[i].pos);
        vertices[i].normal = TransfromVector(normal_to_world, vertices[i].normal);
    }

    const uint64_t index_offset = info_.primitive_buffer.size(),
                   num_primitives = sizeof(indices) / sizeof(Uvec3);
    Vertex v[3];
    for (int i = 0; i < num_primitives; ++i)
    {
        for (int j = 0; j < 3; ++j)
            v[j] = vertices[indices[i][j]];
        info_.primitive_buffer.emplace_back(v, id_bsdf);
    }

    info_.instance_buffer.emplace_back(
        info_.instance_buffer.size(),
        info_.bsdf_info_buffer[id_bsdf].type == Bsdf::Type::kAreaLight, index_offset,
        num_primitives, info_.primitive_buffer.data());
}

void ConfigParser::CreateMeshes(const std::string &filename, const int index_shape,
                                const Mat4 &to_world, const bool flip_texcoords,
                                const bool face_normals, const uint64_t id_bsdf)
{
    ModelLoader loader;
    std::vector<Primitive> local_primitive_buffer =
        GetSuffix(filename) == "serialized"
            ? loader.Load(filename, index_shape, to_world, flip_texcoords, face_normals, id_bsdf)
            : loader.Load(filename, to_world, flip_texcoords, face_normals, id_bsdf);

    const uint64_t index_offset = info_.primitive_buffer.size(),
                   num_primitives = local_primitive_buffer.size();

    info_.primitive_buffer.insert(info_.primitive_buffer.end(), local_primitive_buffer.begin(),
                                  local_primitive_buffer.end());

    info_.instance_buffer.emplace_back(
        info_.instance_buffer.size(),
        info_.bsdf_info_buffer[id_bsdf].type == Bsdf::Type::kAreaLight, index_offset,
        num_primitives, info_.primitive_buffer.data());
}

uint64_t ConfigParser::ReadTexture(const pugi::xml_node &parent_node,
                                   const std::vector<std::string> &valid_names,
                                   const float defalut_value)
{
    pugi::xml_node texture_node;
    if (mitsuba_parser::GetChildByName(parent_node, valid_names, &texture_node))
    {
        return ReadTexture(texture_node, 1.0f, defalut_value);
    }
    else
    {
        const uint64_t index = info_.texture_info_buffer.size();
        const std::string id = "texture_" + std::to_string(index);
        id_texture_mp_[id] = index;
        info_.texture_info_buffer.push_back(Texture::Info::CreateConstant(Vec3(defalut_value)));
        return index;
    }
}

uint64_t ConfigParser::ReadTexture(const pugi::xml_node &texture_node, const float scale,
                                   const float defalut_value)
{
    std::string id = texture_node.attribute("id").as_string();
    if (!texture_node)
    {
        const float index = info_.texture_info_buffer.size();
        if (id.empty())
            id = "texture_" + std::to_string(index);
        id_texture_mp_[id] = index;
        info_.texture_info_buffer.push_back(Texture::Info::CreateConstant(Vec3(scale * defalut_value)));
        return index;
    }

    uint64_t id_texture = kInvalidId;
    std::string node_name = texture_node.name();
    switch (Hash(node_name.c_str()))
    {
    case "scale"_hash:
    {
        const float local_scale = mitsuba_parser::ReadFloat(texture_node, {"scale"}, 1.0f);
        id_texture = ReadTexture(texture_node.child("texture"), scale * local_scale, defalut_value);
        break;
    }
    case "ref"_hash:
    {
        if (!id_texture_mp_.count(id))
        {
            fprintf(stderr, "[error] cannot find texture with id '%s'.\n", id.c_str());
            exit(1);
        }
        id_texture = id_texture_mp_.at(id);
        break;
    }
    case "rgb"_hash:
    {
        const float index = info_.texture_info_buffer.size();
        if (id.empty())
            id = "texture_" + std::to_string(index);
        const Vec3 value = mitsuba_parser::ReadVec3(texture_node, Vec3(defalut_value), nullptr);
        id_texture_mp_[id] = index;
        info_.texture_info_buffer.push_back(Texture::Info::CreateConstant(scale * value));
        id_texture = index;
        break;
    }
    case "float"_hash:
    {
        const float index = info_.texture_info_buffer.size();
        if (id.empty())
            id = "texture_" + std::to_string(index);
        const float value = texture_node.attribute("value").as_float(defalut_value);
        id_texture_mp_[id] = index;
        info_.texture_info_buffer.push_back(Texture::Info::CreateConstant(Vec3(scale * value)));
        id_texture = index;
        break;
    }
    case "texture"_hash:
    {
        switch (Hash(texture_node.attribute("type").value()))
        {
        case "checkerboard"_hash:
        {
            Vec3 color0 = {0.4f, 0.4f, 0.4f};
            pugi::xml_node node_tmp;
            if (mitsuba_parser::GetChildByName(texture_node, {"color0"}, &node_tmp))
            {
                if (std::string(node_tmp.name()) == "rgb" ||
                    std::string(node_tmp.name()) == "float")
                {
                    color0 = mitsuba_parser::ReadVec3(texture_node, {"color0"}, 0.4f);
                }
                else
                {
                    fprintf(stderr, "[warning] not support texture inside 'checkerboard'");
                    fprintf(stderr, ", use default color instead.\n");
                }
            }

            Vec3 color1 = {0.2f, 0.2f, 0.2f};
            if (mitsuba_parser::GetChildByName(texture_node, {"color1"}, &node_tmp))
            {
                if (std::string(node_tmp.name()) == "rgb" ||
                    std::string(node_tmp.name()) == "float")
                {
                    color1 = mitsuba_parser::ReadVec3(texture_node, {"color1"}, 0.2f);
                }
                else
                {
                    fprintf(stderr, "[warning] not support texture inside 'checkerboard'");
                    fprintf(stderr, ", use default color instead.\n");
                }
            }

            Mat4 to_uv = mitsuba_parser::ReadTransform4(texture_node.child("transform"));
            float u_offset = mitsuba_parser::ReadFloat(texture_node, {"uoffset"}, 0.0f),
                  v_offset = mitsuba_parser::ReadFloat(texture_node, {"voffset"}, 0.0f),
                  u_scale = mitsuba_parser::ReadFloat(texture_node, {"uscale"}, 1.0f),
                  v_scale = mitsuba_parser::ReadFloat(texture_node, {"vscale"}, 1.0f);
            to_uv = Mul(Translate(Vec3{u_offset, v_offset, 0.0f}), to_uv);
            to_uv = Mul(Scale(Vec3{u_scale, v_scale, 1.0f}), to_uv);

            const float index = info_.texture_info_buffer.size();
            if (id.empty())
                id = "texture_" + std::to_string(index);
            id_texture_mp_[id] = index;
            info_.texture_info_buffer.push_back(Texture::Info::CreateCheckerboard(
                scale * color0, scale * color1, to_uv));
            id_texture = index;
            break;
        }
        case "bitmap"_hash:
        {
            pugi::xml_node child_node;
            if (!mitsuba_parser::GetChildByName(texture_node, {"filename"}, &child_node))
            {
                fprintf(stderr, "[error] cannot find filename for bitmap texture.\n");
                exit(1);
            }
            const float gamma = mitsuba_parser::ReadFloat(texture_node, {"gamma"}, -1.0f);
            const uint64_t index = info_.texture_info_buffer.size();
            if (id.empty())
                id = "texture_" + std::to_string(index);
            id_texture = ReadBitmap(config_file_directory_ + child_node.attribute("value").as_string(),
                                    id, gamma, scale, nullptr);
            break;
        }
        default:
        {
            fprintf(stderr, "[error] unsupport texture type '%s'.\n",
                    texture_node.attribute("type").value());
            exit(1);
            break;
        }
        }
        break;
    }
    default:
    {
        fprintf(stderr, "[error] unsupport texture type '%s'.\n",
                texture_node.attribute("type").value());
        exit(1);
        break;
    }
    }
    return id_texture;
}

uint64_t ConfigParser::ReadBsdf(pugi::xml_node bsdf_node, std::string id, uint64_t id_opacity,
                                uint64_t id_bumpmap, bool twosided)
{
    if (id.empty())
        id = bsdf_node.attribute("id").as_string();
    std::string type = bsdf_node.attribute("type").as_string();
    switch (Hash(type.c_str()))
    {
    case "bumpmap"_hash:
    {
        id_bumpmap = ReadTexture(bsdf_node.child("texture"), 1.0f, 1.0f);
        return ReadBsdf(bsdf_node.child("bsdf"), id, id_opacity, id_bumpmap, twosided);
    }
    case "mask"_hash:
    {
        id_opacity = ReadTexture(bsdf_node, {"opacity"}, 1.0f);
        return ReadBsdf(bsdf_node.child("bsdf"), id, id_opacity, id_bumpmap, twosided);
    }
    case "twosided"_hash:
    {
        return ReadBsdf(bsdf_node.child("bsdf"), id, id_opacity, id_bumpmap, true);
    }
    case "coating"_hash:
    case "roughcoating"_hash:
    case "phong"_hash:
    case "ward"_hash:
    case "mixturebsdf"_hash:
    case "blendbsdf"_hash:
    case "difftrans"_hash:
    case "hk"_hash:
    case "irawan"_hash:
    case "null"_hash:
    {
        fprintf(stderr, "[error] not support bsdf type '%s'.\n", type.c_str());
        exit(1);
    }
    default:
    {
        break;
    }
    }

    const uint64_t index = info_.bsdf_info_buffer.size();
    if (id.empty())
        id = "bsdf_" + std::to_string(index);

    uint64_t id_bsdf = index;
    bool is_thin_dielectric = false;
    switch (Hash(type.c_str()))
    {
    case "roughdiffuse"_hash:
    {
        bool use_fast_aaprox = mitsuba_parser::ReadBoolean(bsdf_node, {"useFastApprox"}, false);
        uint64_t id_reflectance = ReadTexture(bsdf_node, {"reflectance"}, 0.5f);
        uint64_t id_roughness = ReadTexture(bsdf_node, {"alpha"}, 0.2f);
        info_.bsdf_info_buffer.push_back(Bsdf::Info::CreateRoughDiffuse(
            id_reflectance, id_roughness, use_fast_aaprox, twosided, id_opacity, id_bumpmap));
        break;
    }
    case "diffuse"_hash:
    {
        uint64_t id_reflectance = ReadTexture(bsdf_node, {"reflectance"}, 0.5f);
        info_.bsdf_info_buffer.push_back(Bsdf::Info::CreateDiffuse(id_reflectance, twosided,
                                                                   id_opacity, id_bumpmap));
        break;
    }
    case "thindielectric"_hash:
    {
        is_thin_dielectric = true;
    }
    case "dielectric"_hash:
    case "roughdielectric"_hash:
    {
        twosided = true;
        const float int_ior = ReadDielectricIor(bsdf_node, {"int_ior", "intIOR"}, 1.5046f);
        const float ext_ior = ReadDielectricIor(bsdf_node, {"ext_ior", "extIOR"}, 1.000277f);
        uint64_t id_roughness;
        if (type == "roughdielectric")
            id_roughness = ReadTexture(bsdf_node, {"alpha", "alpha_u", "alphaU"}, 0.1f);
        else
            id_roughness = ReadTexture(bsdf_node, std::vector<std::string>{}, 0.001f);
        const uint64_t id_specular_reflectance = ReadTexture(
            bsdf_node, {"specularReflectance", "specular_reflectance"}, 1.0f);
        const uint64_t id_specular_transmittance = ReadTexture(
            bsdf_node, {"specularTransmittance", "specular_transmittance"}, 1.0f);
        if (is_thin_dielectric)
            info_.bsdf_info_buffer.push_back(Bsdf::Info::CreateDielectric(
                id_roughness, id_specular_reflectance, id_specular_transmittance,
                int_ior / ext_ior, is_thin_dielectric, twosided, id_opacity, id_bumpmap));
        else
            info_.bsdf_info_buffer.push_back(Bsdf::Info::CreateDielectric(
                id_roughness, id_specular_reflectance, id_specular_transmittance,
                int_ior / ext_ior, is_thin_dielectric, twosided, id_opacity, id_bumpmap));
        break;
    }
    case "conductor"_hash:
    case "roughconductor"_hash:
    {
        uint64_t id_roughness;
        if (type == "roughconductor")
            id_roughness = ReadTexture(bsdf_node, {"alpha", "alpha_u", "alphaU"}, 0.1f);
        else
            id_roughness = ReadTexture(bsdf_node, std::vector<std::string>{}, 0.001f);
        const uint64_t id_specular_reflectance = ReadTexture(
            bsdf_node, {"specularReflectance", "specular_reflectance"}, 1.0f);
        Vec3 eta, k;
        ReadConductorIor(bsdf_node, &eta, &k);
        info_.bsdf_info_buffer.push_back(Bsdf::Info::CreateConductor(
            id_roughness, id_specular_reflectance, eta, k, twosided, id_opacity, id_bumpmap));
        break;
    }
    case "plastic"_hash:
    case "roughplastic"_hash:
    {
        const float int_ior = ReadDielectricIor(bsdf_node, {"int_ior", "intIOR"}, 1.5046f);
        const float ext_ior = ReadDielectricIor(bsdf_node, {"ext_ior", "extIOR"}, 1.000277f);
        uint64_t id_roughness;
        if (type == "roughplastic")
            id_roughness = ReadTexture(bsdf_node, {"alpha", "alpha_u", "alphaU"}, 0.1f);
        else
            id_roughness = ReadTexture(bsdf_node, std::vector<std::string>{}, 0.001f);
        const uint64_t id_diffuse_reflectance = ReadTexture(
            bsdf_node, {"diffuseReflectance", "diffuse_reflectance"}, 1.0f);
        const uint64_t id_specular_reflectance = ReadTexture(
            bsdf_node, {"specularReflectance", "specular_reflectance"}, 1.0f);
        info_.bsdf_info_buffer.push_back(Bsdf::Info::CreatePlastic(
            id_roughness, id_diffuse_reflectance, id_specular_reflectance, int_ior / ext_ior,
            twosided, id_opacity, id_bumpmap));
        break;
    }
    default:
    {
        fprintf(stderr, "[warning] unsupport bsdf type '%s', use default 'diffuse' instead.\n",
                type.c_str());
        uint64_t id_reflectance = ReadTexture(bsdf_node, std::vector<std::string>{}, 0.5f);
        info_.bsdf_info_buffer.push_back(Bsdf::Info::CreateDiffuse(id_reflectance, twosided,
                                                                   id_opacity, id_bumpmap));
        break;
    }
    }
    id_bsdf_mp_[id] = id_bsdf;
    return id_bsdf;
}

void ConfigParser::ReadShape(pugi::xml_node shape_node)
{
    std::string id = shape_node.attribute("id").value();
    const uint64_t index = info_.instance_buffer.size();
    if (id.empty())
        id = "shape_" + std::to_string(index);

    std::string type = shape_node.attribute("type").value();
    Mat4 to_world = mitsuba_parser::ReadTransform4(shape_node.child("transform"));

    bool flip_texcoords = false, flip_normals = false, face_normals = false;
    for (pugi::xml_node node : shape_node.children("boolean"))
    {
        switch (Hash(node.attribute("name").value()))
        {
        case "flip_normals"_hash:
        case "flipNormals"_hash:
            flip_normals = node.attribute("value").as_bool(false);
            break;
        case "face_normals"_hash:
        case "faceNormals"_hash:
            face_normals = node.attribute("value").as_bool(false);
            break;
        default:
            break;
        }
    }

    uint64_t id_bsdf;
    if (shape_node.child("emitter"))
    {
        pugi::xml_node emitter_node = shape_node.child("emitter");
        assert(emitter_node.attribute("type").value() == std::string("area") ||
               emitter_node.attribute("type").value() == std::string("constant"));
        if (!emitter_node.child("rgb"))
        {
            fprintf(stderr, "[error] cannot find radiance for area light '%s'.\n", id.c_str());
            exit(1);
        }

        const Vec3 radiance = mitsuba_parser::ReadVec3(emitter_node, {"radiance"}, Vec3(1));
        const uint64_t index_texture = info_.texture_info_buffer.size();
        const std::string texture_id = "texture_" + std::to_string(index_texture);
        id_texture_mp_[texture_id] = info_.texture_info_buffer.size();
        info_.texture_info_buffer.push_back(Texture::Info::CreateConstant(radiance));
        size_t id_radiance = id_texture_mp_[texture_id];

        id_bsdf = info_.bsdf_info_buffer.size();
        info_.bsdf_info_buffer.push_back(Bsdf::Info::CreateAreaLight(id_radiance, false, kInvalidId,
                                                                     kInvalidId));
        id_bsdf_mp_[id] = id_bsdf;
    }
    else if (shape_node.child("bsdf"))
    {
        id_bsdf = ReadBsdf(shape_node.child("bsdf"), "", kInvalidId, kInvalidId, false);
    }
    else
    {
        bool found = false;
        for (pugi::xml_node ref_node : shape_node.children("ref"))
        {
            std::string bsdf_id = shape_node.child("ref").attribute("id").value();
            if (id_bsdf_mp_.count(bsdf_id))
            {
                id_bsdf = id_bsdf_mp_.at(bsdf_id);
                found = true;
                break;
            }
        }
        if (!found)
        {
            id_bsdf = ReadBsdf(pugi::xml_node(), "", kInvalidId, kInvalidId, false);
        }
    }

    int index_shape = shape_node.child("integer").attribute("value").as_int(0);

    switch (Hash(type.c_str()))
    {
    case "cube"_hash:
    {
        CreateCube(to_world, id_bsdf);
        break;
    }
    case "rectangle"_hash:
    {
        CreateRectangle(to_world, id_bsdf);
        break;
    }
    case "sphere"_hash:
    {
        CreateSphere(mitsuba_parser::ReadVec3(shape_node, {"center"}, Vec3(0)),
                     shape_node.child("float").attribute("value").as_float(1.0), to_world,
                     id_bsdf);
        break;
    }
    case "disk"_hash:
    {
        CreateDisk(to_world, id_bsdf);
        break;
    }
    case "obj"_hash:
    {
        flip_texcoords = mitsuba_parser::ReadBoolean(shape_node,
                                                     {"flip_tex_coords", "flipTexCoords"}, true);
    }
    case "gltf"_hash:
    case "ply"_hash:
    case "serialized"_hash:
    {
        std::string filename = shape_node.child("string").attribute("value").as_string();
        CreateMeshes(config_file_directory_ + filename, index_shape, to_world, flip_texcoords,
                     face_normals, id_bsdf);
        break;
    }
    default:
    {
        fprintf(stderr, "[warning] unsupported shape type '%s', ignore it.\n", type.c_str());
        break;
    }
    }
}

void ConfigParser::ReadEmitter(pugi::xml_node emitter_node)
{
    std::string type = emitter_node.attribute("type").as_string();
    switch (Hash(type.c_str()))
    {
    case "constant"_hash:
    {
        const uint64_t id_radiance = ReadTexture(emitter_node, {"radiance"}, 1.0f);
        info_.env_map = new EnvMap(id_radiance, 1.0f, Mat4());
        break;
    }
    case "envmap"_hash:
    {
        const std::string filename = emitter_node.child("string").attribute("value").as_string();
        const float gamma = mitsuba_parser::ReadFloat(emitter_node, {"gamma"}, -1.0f);
        int width_target = static_cast<int>(info_.camera.width() * 360 / info_.camera.fov_x());
        const uint64_t id_radiance = ReadBitmap(config_file_directory_ + filename,
                                                filename, gamma, 1.0f, &width_target);
        Mat4 to_world = mitsuba_parser::ReadTransform4(emitter_node.child("transform"));
        float scale = mitsuba_parser::ReadFloat(emitter_node, {"scale"}, 1.0f);
        info_.env_map = new EnvMap(id_radiance, scale, to_world);
        break;
    }
    case "sun"_hash:
    case "sky"_hash:
    case "sunsky"_hash:
    {
        Vec3 sun_direction(0);
        pugi::xml_node sun_direction_node;
        if (mitsuba_parser::GetChildByName(emitter_node, {"sunDirection"}, &sun_direction_node))
        {
            sun_direction = {sun_direction_node.attribute("x").as_float(),
                             sun_direction_node.attribute("y").as_float(),
                             sun_direction_node.attribute("z").as_float()};
        }
        else
        {
            LocationDate location_date;
            location_date.year = mitsuba_parser::ReadInt(emitter_node, {"year"}, 2010);
            location_date.month = mitsuba_parser::ReadInt(emitter_node, {"month"}, 7);
            location_date.day = mitsuba_parser::ReadInt(emitter_node, {"day"}, 10);
            location_date.hour = mitsuba_parser::ReadFloat(emitter_node, {"hour"}, 15);
            location_date.minute = mitsuba_parser::ReadFloat(emitter_node, {"minute"}, 0);
            location_date.second = mitsuba_parser::ReadFloat(emitter_node, {"second"}, 0);
            location_date.latitude = mitsuba_parser::ReadFloat(emitter_node, {"latitude"}, 35.6894);
            location_date.longitude = mitsuba_parser::ReadFloat(emitter_node, {"longitude"},
                                                                139.6917);
            location_date.timezone = mitsuba_parser::ReadFloat(emitter_node, {"timezone"}, 9);
            sun_direction = GetSunDirection(location_date);
        }

        Vec3 albedo = mitsuba_parser::ReadVec3(emitter_node, {"albedo"}, Vec3(0.15));

        double turbidity = mitsuba_parser::ReadFloat(emitter_node, {"turbidity"}, 3);
        turbidity = fminf(fmaxf(turbidity, 10.0f), 1.0f);

        double stretch = mitsuba_parser::ReadFloat(emitter_node, {"stretch"}, 1);
        stretch = std::min(std::max(turbidity, 2.0), 1.0);

        int resolution = mitsuba_parser::ReadInt(emitter_node, {"resolution"}, 512);

        double sun_scale = mitsuba_parser::ReadFloat(emitter_node, {"sunScale"}, 1);
        double sky_scale = mitsuba_parser::ReadFloat(emitter_node, {"skyScale"}, 1);
        double sun_radius_scale = mitsuba_parser::ReadFloat(emitter_node, {"sunRadiusScale"}, 1);
        bool extend = mitsuba_parser::ReadBoolean(emitter_node, {"extend"}, true);

        int width = resolution, height = resolution / 2, channel = 3;
        if (type == "sky" || type == "sunsky")
        {
            std::vector<float> data;
            CreateSkyTexture(sun_direction, albedo, turbidity, stretch, sky_scale,
                             extend, width, height, &data);
            const uint64_t offset = info_.pixel_buffer.size();
            info_.pixel_buffer.insert(info_.pixel_buffer.end(), data.begin(), data.end());
            const uint64_t index = info_.texture_info_buffer.size();
            id_texture_mp_["sky_texture"] = index;
            info_.texture_info_buffer.push_back(Texture::Info::CreateBitmap(offset, width, height,
                                                                            channel));
            info_.env_map = new EnvMap(index, 1.0f, Mat4());
        }
        if (type == "sun" || type == "sunsky")
        {
            Vec3 sun_radiance;
            std::vector<float> data;
            CreateSunTexture(sun_direction, turbidity, sun_scale, sun_radius_scale, width, height,
                             &sun_radiance, &data);
            const uint64_t offset = info_.pixel_buffer.size();
            info_.pixel_buffer.insert(info_.pixel_buffer.end(), data.begin(), data.end());
            const uint64_t index = info_.texture_info_buffer.size();
            id_texture_mp_["sun_texture"] = index;
            info_.texture_info_buffer.push_back(Texture::Info::CreateBitmap(
                offset, width, height, channel));

            info_.emitter_info_buffer.push_back(Emitter::Info::CreateSun(
                -sun_direction, sun_radiance, sun_radius_scale, index));
        }
        break;
    }
    case "directional"_hash:
    {
        Mat4 to_world = mitsuba_parser::ReadTransform4(emitter_node.child("transform"));
        Vec3 direction = mitsuba_parser::ReadVec3(emitter_node, {"direction"}, Vec3{0, 0, 1});
        direction = TransfromVector(to_world.Transpose().Inverse(), direction);
        Vec3 radiance = mitsuba_parser::ReadVec3(emitter_node, {"radiance", "irradiance"}, Vec3(1));
        info_.emitter_info_buffer.push_back(Emitter::Info::CreateDirctional(direction, radiance));
        break;
    }
    case "spot"_hash:
    {
        Vec3 intensity = mitsuba_parser::ReadVec3(emitter_node, {"intensity"}, Vec3(1));
        float cutoff_angle = mitsuba_parser::ReadFloat(
            emitter_node, {"cutoff_angle", "cutoffAngle"}, 20);
        float beam_width = mitsuba_parser::ReadFloat(
            emitter_node, {"beamWidth", "beam_width"}, cutoff_angle * 0.75f);
        Mat4 to_world = mitsuba_parser::ReadTransform4(emitter_node.child("transform"));
        uint64_t id_texture = kInvalidId;
        if (emitter_node.child("texture"))
            id_texture = ReadTexture(emitter_node.child("texture"), 1.0f, 1.0f);
        info_.emitter_info_buffer.push_back(Emitter::Info::CreateSpotLight(
            to_world, intensity, ToRadians(cutoff_angle), ToRadians(beam_width), id_texture));
        break;
    }
    default:
    {
        fprintf(stderr, "[warning] unsupport emitter '%s', ignore it.\n", type.c_str());
        break;
    }
    }
}

uint64_t ConfigParser::ReadBitmap(const std::string &filename, const std::string &id, float gamma,
                                  float scale, int *width_max)
{
    int width, height, channel;
    std::vector<float> data;
    image_io::Read(filename, gamma, &width, &height, &channel, &data, width_max);
    for (float &item : data)
        item *= scale;

    const uint64_t offset = info_.pixel_buffer.size();
    info_.pixel_buffer.insert(info_.pixel_buffer.end(), data.begin(), data.end());

    const uint64_t index = info_.texture_info_buffer.size();
    id_texture_mp_[id] = index;
    info_.texture_info_buffer.push_back(Texture::Info::CreateBitmap(offset, width, height,
                                                                    channel));
    return index;
}

float ConfigParser::ReadDielectricIor(const pugi::xml_node &parent_node,
                                      const std::vector<std::string> &valid_names,
                                      float defalut_value)
{
    pugi::xml_node ior_node;
    mitsuba_parser::GetChildByName(parent_node, valid_names, &ior_node);
    if (ior_node.name() == std::string("string"))
    {
        std::string material_name = ior_node.attribute("value").as_string();
        float ior = 0.0;
        if (!ior_lut::LookupDielectricIor(material_name, &ior))
        {
            fprintf(stderr, "[error] unsupported  material '%s'.\n", material_name.c_str());
            exit(1);
        }
        return ior;
    }
    else
    {
        return ior_node.attribute("value").as_float(defalut_value);
    }
}

void ConfigParser::ReadConductorIor(const pugi::xml_node &parent_node, Vec3 *eta, Vec3 *k)
{
    pugi::xml_node child_node;
    if (mitsuba_parser::GetChildByName(parent_node, {"material"}, &child_node))
    {
        std::string material_name = child_node.attribute("value").as_string();
        if (!ior_lut::LookupConductorIor(material_name, eta, k))
        {
            fprintf(stderr, "[error] unsupported  material '%s'.\n", material_name.c_str());
            exit(1);
        }
    }
    else if (mitsuba_parser::GetChildByName(parent_node, {"eta"}, &child_node))
    {
        *eta = mitsuba_parser::ReadVec3(child_node, Vec3{1.0, 1.0, 1.0}, nullptr);
        if (!mitsuba_parser::GetChildByName(parent_node, {"k"}, &child_node))
        {
            fprintf(stderr, "[error] cannot find 'k for Conductor bsdf '%s'.\n",
                    parent_node.attribute("id").as_string());
            exit(1);
        }
        *k = mitsuba_parser::ReadVec3(child_node, Vec3{1.0, 1.0, 1.0}, nullptr);
    }
    else
    {
        ior_lut::LookupConductorIor("Cu", eta, k);
    }
}

bool mitsuba_parser::ReadBoolean(const pugi::xml_node &parent_node,
                                 const std::vector<std::string> &valid_names,
                                 bool defalut_value)
{
    pugi::xml_node boolean_node;
    GetChildByName(parent_node, valid_names, &boolean_node);
    return boolean_node.attribute("value").as_bool(defalut_value);
}

int mitsuba_parser::ReadInt(const pugi::xml_node &parent_node,
                            const std::vector<std::string> &valid_names,
                            int defalut_value)
{
    pugi::xml_node int_node;
    GetChildByName(parent_node, valid_names, &int_node);
    return int_node.attribute("value").as_int(defalut_value);
}

float mitsuba_parser::ReadFloat(const pugi::xml_node &parent_node,
                                const std::vector<std::string> &valid_names,
                                float defalut_value)
{
    pugi::xml_node double_node;
    GetChildByName(parent_node, valid_names, &double_node);
    return double_node.attribute("value").as_float(defalut_value);
}

Vec3 mitsuba_parser::ReadVec3(const pugi::xml_node &parent_node,
                              const std::vector<std::string> &valid_names,
                              const Vec3 &defalut_value)
{
    pugi::xml_node vec3_node;
    GetChildByName(parent_node, valid_names, &vec3_node);
    if (!vec3_node)
        return defalut_value;
    else
        return ReadVec3(vec3_node, defalut_value, nullptr);
}

Vec3 mitsuba_parser::ReadVec3(const pugi::xml_node &vec3_node, const Vec3 &defalut_value,
                              const char *value_name)
{
    if (vec3_node.attribute("value") || value_name != nullptr)
    {
        std::string str_buffer = vec3_node.attribute(
                                              value_name != nullptr ? value_name : "value")
                                     .as_string();
        const int space_count = static_cast<int>(std::count(str_buffer.begin(),
                                                            str_buffer.end(), ' '));
        if (space_count == 0)
        {
            const float value = vec3_node.attribute("value").as_float(defalut_value.x);
            return Vec3(value);
        }
        else if (space_count == 2)
        {
            Vec3 result(0);
            const int Comma_count = static_cast<int>(std::count(str_buffer.begin(),
                                                                str_buffer.end(), ','));
            if (Comma_count == 0)
            {
                sscanf(str_buffer.c_str(), "%f %f %f", &result[0], &result[1], &result[2]);
                return result;
            }
            else
            {
                sscanf(str_buffer.c_str(), "%f, %f, %f", &result[0], &result[1], &result[2]);
                return result;
            }
        }
        else
        {
            return defalut_value;
        }
    }
    else
    {
        const Vec3 value = {vec3_node.attribute("x").as_float(defalut_value.x),
                            vec3_node.attribute("y").as_float(defalut_value.y),
                            vec3_node.attribute("z").as_float(defalut_value.z)};
        return value;
    }
}

Mat4 mitsuba_parser::ReadMat4(const pugi::xml_node &matrix_node)
{
    if (!matrix_node.attribute("value"))
        return Mat4();

    Mat4 result;
    std::string str_buffer = matrix_node.attribute("value").as_string();
    const int space_count = static_cast<int>(std::count(str_buffer.begin(),
                                                        str_buffer.end(), ' '));
    if (space_count == 8)
    {
        sscanf(str_buffer.c_str(), "%f %f %f %f %f %f %f %f %f",
               &result[0][0], &result[0][0], &result[0][1],
               &result[1][0], &result[1][1], &result[1][2],
               &result[2][0], &result[2][1], &result[2][2]);
    }
    else
    {
        sscanf(str_buffer.c_str(), "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f",
               &result[0][0], &result[0][1], &result[0][2], &result[0][3],
               &result[1][0], &result[1][1], &result[1][2], &result[1][3],
               &result[2][0], &result[2][1], &result[2][2], &result[2][3],
               &result[3][0], &result[3][1], &result[3][2], &result[3][3]);
    }
    return result;
}

Mat4 mitsuba_parser::ReadTransform4(const pugi::xml_node &transform_node)
{
    Mat4 result;
    if (!transform_node)
        return result;

    for (pugi::xml_node node : transform_node.children())
    {
        const char *name = node.name();
        switch (Hash(name))
        {
        case "translate"_hash:
        {
            Vec3 translate = ReadVec3(node, Vec3(0), nullptr);
            result = Mul(Translate(translate), result);
            break;
        }
        case "rotate"_hash:
        {
            Vec3 axis = ReadVec3(node, Vec3(0), nullptr);
            float angle = node.attribute("angle").as_float(0);
            result = Mul(Rotate(ToRadians(angle), axis), result);
            break;
        }
        case "scale"_hash:
        {
            Vec3 scale = ReadVec3(node, Vec3(1), nullptr);
            result = Mul(Scale(scale), result);
            break;
        }
        case "matrix"_hash:
        {
            Mat4 matrix = ReadMat4(node);
            result = Mul(matrix, result);
            break;
        }
        case "lookat"_hash:
        {
            Vec3 origin = ReadVec3(node, Vec3{0, 0, 0}, "origin"),
                 target = ReadVec3(node, Vec3{1, 0, 0}, "target"),
                 up = ReadVec3(node, Vec3{0, 1, 0}, "up");
            result = Mul(LookAtLH(origin, target, up).Inverse(), result);
            break;
        }
        default:
        {
            fprintf(stderr, "[warning] unsupport transform type '%s', ignore it.\n", name);
            break;
        }
        }
    }
    return result;
}

bool mitsuba_parser::GetChildByName(const pugi::xml_node &parent_node,
                                    const std::vector<std::string> &valid_names,
                                    pugi::xml_node *child_node)
{
    for (const std::string &name : valid_names)
    {
        for (pugi::xml_node node : parent_node.children())
        {
            const std::string node_name = node.attribute("name").value();
            if (node_name == name)
            {
                if (child_node != nullptr)
                    *child_node = node;
                return true;
            }
        }
    }
    return false;
}
