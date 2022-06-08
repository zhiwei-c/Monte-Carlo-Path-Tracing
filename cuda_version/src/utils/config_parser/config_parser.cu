#include "config_parser.h"

Renderer *renderer = nullptr;
uint bsdf_cnt;
uint texture_cnt;
std::string xml_directory;
std::map<std::string, uint> m_id_to_m_idx;

Renderer *ParseRenderConfig(const std::string &config_path)
{
    bsdf_cnt = 0;
    texture_cnt = 0;
    renderer = new Renderer();
    xml_directory = GetDirectory(ConvertBackSlash(config_path));
    //
    auto file_doc = new rapidxml::file<>(config_path.c_str());
    auto xml_doc = new rapidxml::xml_document<>();
    xml_doc->parse<0>(file_doc->data());
    auto node_scene = xml_doc->first_node("scene");

    //
    auto node_bsdf = node_scene->first_node("bsdf");
    while (node_bsdf)
    {
        ParseBsdf(node_bsdf);
        node_bsdf = node_bsdf->next_sibling("bsdf");
    }

    //
    auto node_shape = node_scene->first_node("shape");
    while (node_shape)
    {
        ParseShape(node_shape);
        node_shape = node_shape->next_sibling("shape");
    }
    //
    ParseIntegrator(node_scene->first_node("integrator"));

    auto camera_info = ParseCamera(node_scene->first_node("sensor"));
    //
    ParseEnvmap(node_scene->first_node("emitter"), camera_info);
    //
    return renderer;
}

//======================================================================================================

void ParseIntegrator(rapidxml::xml_node<> *node_integrator)
{
    auto type = GetAttri(node_integrator, "type").value();
    auto max_depth = GetInt(node_integrator, "maxDepth", true).value_or(-1);
    auto rr_depth = GetInt(node_integrator, "rrDepth", true).value_or(5);

    switch (Hash(type.c_str()))
    {
    case "path"_hash:
        break;
    default:
        std::cerr << "[warning] " << GetTreeName(node_integrator) << std::endl
                  << "\tcannot handle integrator type \"" << type << "\", use path" << std::endl;
        break;
    }
    renderer->AddIntegratorInfo(IntegratorInfo(max_depth, rr_depth));
}

//======================================================================================================

CameraInfo ParseCamera(rapidxml::xml_node<> *node_sensor)
{

    auto type = GetAttri(node_sensor, "type").value();
    if (type != "perspective")
    {
        std::cerr << "[error] " << GetTreeName(node_sensor) << std::endl
                  << "\tcannot handle sensor except from perspective" << std::endl;
        exit(1);
    }

    CameraInfo camera_info;

    auto fov_width = GetFloat(node_sensor, "fov", false).value();
    auto eye_pos = gvec3(0, 0, 0);
    auto look_at = gvec3(0, 0, 1);
    auto up = gvec3(0, 1, 0);
    if (auto node_lookat = GetChild(node_sensor, "toWorld")->first_node("lookat");
        node_lookat)
    {
        auto origin_str = GetAttri(node_lookat, "origin").value();
        sscanf(origin_str.c_str(), "%lf, %lf, %lf", &eye_pos[0], &eye_pos[1], &eye_pos[2]);

        auto target_str = GetAttri(node_lookat, "target").value();
        sscanf(target_str.c_str(), "%lf, %lf, %lf", &look_at[0], &look_at[1], &look_at[2]);

        auto up_str = GetAttri(node_lookat, "up").value();
        sscanf(up_str.c_str(), "%lf, %lf, %lf", &up[0], &up[1], &up[2]);
    }
    else if (auto to_world = GetToWorld(node_sensor);
             to_world)
    {
        eye_pos = TransfromPt(*to_world, eye_pos);
        look_at = TransfromPt(*to_world, look_at);
        up = TransfromDir(*to_world, up);
    }
    camera_info.eye_pos = eye_pos;
    camera_info.look_dir = glm::normalize(look_at - eye_pos);
    camera_info.up = up;

    auto node_film = node_sensor->first_node("film");
    camera_info.width = GetInt(node_film, "width").value_or(768);
    camera_info.height = GetInt(node_film, "height").value_or(576);
    camera_info.fov_height = fov_width * camera_info.height / camera_info.width;
    camera_info.gamma = GetFloat(node_film, "gamma").value_or(-1);
    auto node_sampler = node_sensor->first_node("sampler");
    camera_info.sample_count = GetInt(node_sampler, "sampleCount", false).value();

    renderer->AddCameraInfo(camera_info);
    return camera_info;
}

//======================================================================================================

void ParseEnvmap(rapidxml::xml_node<> *node_envmap, const CameraInfo &camera_info)
{
    if (!node_envmap)
        return;

    auto envmap_type = GetAttri(node_envmap, "type").value();
    switch (Hash(envmap_type.c_str()))
    {
    case "envmap"_hash:
    {
        auto env_map_info = new EnvMapInfo();
        auto to_world = GetToWorld(node_envmap);
        if (to_world)
        {
            env_map_info->to_local = new gmat4(glm::inverse(*to_world));
            delete to_world;
            to_world = nullptr;
        }
        auto node_filename = GetChild(node_envmap, "filename", false);
        auto filename = xml_directory + GetAttri(node_filename, "value").value();
        auto gamma = GetFloat(node_envmap, "gamma").value_or(-1);
        auto radiance = new TextureInfo(filename, gamma);

        auto new_height = static_cast<int>(camera_info.height * 180.0 / camera_info.fov_height);
        if (new_height < camera_info.fov_height)
        {
            auto new_width = static_cast<int>(new_height * camera_info.width / camera_info.height);
            ImageResize(radiance->width, radiance->height, new_width, new_height, radiance->channel, radiance->colors);
            radiance->height = new_height;
            radiance->width = new_width;
        }
        renderer->AddTextureInfo(radiance);
        env_map_info->radiance_idx = texture_cnt++;
        renderer->AddEnvMapInfo(env_map_info);
        break;
    }
    case "constant"_hash:
    {
        auto env_map_info = new EnvMapInfo();
        auto node_radiance = GetChild(node_envmap, "radiance", false);
        auto color = GetVec3(node_radiance);
        auto radiance = new TextureInfo(color);
        renderer->AddTextureInfo(radiance);
        env_map_info->radiance_idx = texture_cnt++;
        renderer->AddEnvMapInfo(env_map_info);
        break;
    }
    default:
    {
        if (node_envmap->next_sibling("emitter"))
            ParseEnvmap(node_envmap->next_sibling("emitter"), camera_info);
        else
        {
            auto env_map_info = new EnvMapInfo();
            auto radiance = new TextureInfo(vec3(0.3));
            renderer->AddTextureInfo(radiance);
            env_map_info->radiance_idx = texture_cnt++;
            renderer->AddEnvMapInfo(env_map_info);
            std::cout << "[warning] " << GetTreeName(node_envmap) << std::endl
                      << "\tunsupported emitter type, use default envmap instead." << std::endl;
        }
        break;
    }
    }
}

//========================================================================================================================

void ParseShape(rapidxml::xml_node<> *node_shape)
{
    std::string ref;
    if (auto node_emitter = node_shape->first_node("emitter"); node_emitter)
    {
        ref = "unnamed_" + std::to_string(bsdf_cnt);
        if (node_emitter->next_sibling("emitter"))
        {
            std::cerr << "[error] " << GetTreeName(node_emitter) << std::endl
                      << "\tfind multiple emitter info" << std::endl;
            exit(1);
        }
        if (GetAttri(node_emitter, "type").value() != "area")
        {
            std::cerr << "[error] " << GetTreeName(node_emitter) << std::endl
                      << "\tcannot handle shape emitter except from area" << std::endl;
            exit(1);
        }
        auto node_radiance = GetChild(node_emitter, "radiance");
        auto radiance = GetVec3(node_radiance);
        auto radiance_info = new TextureInfo(radiance);
        renderer->AddTextureInfo(radiance_info);

        auto area_light_info = BsdfInfo();
        area_light_info.type = kAreaLight;
        area_light_info.radiance_idx = texture_cnt++;
        renderer->AddBsdfInfo(area_light_info);
        m_id_to_m_idx[ref] = bsdf_cnt++;
    }
    else if (auto node_ref = node_shape->first_node("ref"); node_ref)
    {
        if (node_ref->next_sibling("ref"))
        {
            std::cerr << "[error] " << GetTreeName(node_ref) << std::endl
                      << "\tfind multiple ref" << std::endl;
            exit(1);
        }
        ref = GetAttri(node_ref, "id").value();
    }
    else
    {
        ref = "unnamed_" + std::to_string(bsdf_cnt);
        if (auto node_bsdf = node_shape->first_node("bsdf"); node_bsdf)
            ParseBsdf(node_bsdf, &ref);
        else
        {
            std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
                      << "\tcannot find supported bsdf info" << std::endl;
            exit(1);
        }
    }

    if (m_id_to_m_idx.find(ref) == m_id_to_m_idx.end())
    {

        std::cerr << "[error] " << GetTreeName(node_shape) << std::endl
                  << "\tcannot find bsdf with name: \"" << ref << "\"" << std::endl;
        exit(1);
    }
    auto bsdf_idx = m_id_to_m_idx[ref];

    auto type = GetAttri(node_shape, "type").value();
    auto to_world = GetToWorld(node_shape);
    auto flip_normals = GetBoolean(node_shape, "flipNormals").value_or(false);

    switch (Hash(type.c_str()))
    {
    case "cube"_hash:
        renderer->AddShapeInfo(new ShapeInfo(kCube, flip_normals, to_world, bsdf_idx));
        break;
    case "disk"_hash:
        renderer->AddShapeInfo(new ShapeInfo(kDisk, flip_normals, to_world, bsdf_idx));
        break;
    case "obj"_hash:
    {
        auto face_normals = GetBoolean(node_shape, "faceNormals").value_or(false);
        auto filename = xml_directory + GetAttri(GetChild(node_shape, "filename", false), "value").value();
        auto flip_tex_coords = GetBoolean(node_shape, "flipTexCoords").value_or(true);
        renderer->AddShapeInfo(new ShapeInfo(filename, face_normals, flip_normals, flip_tex_coords, to_world, bsdf_idx));
        break;
    }
    case "ply"_hash:
    {
        auto face_normals = GetBoolean(node_shape, "faceNormals").value_or(false);
        auto filename = xml_directory + GetAttri(GetChild(node_shape, "filename", false), "value").value();
        auto flip_tex_coords = GetBoolean(node_shape, "flipTexCoords").value_or(true);
        renderer->AddShapeInfo(new ShapeInfo(filename, face_normals, flip_normals, flip_tex_coords, to_world, bsdf_idx));
        break;
    }
    case "rectangle"_hash:
        renderer->AddShapeInfo(new ShapeInfo(kRectangle, flip_normals, to_world, bsdf_idx));
        break;
    case "sphere"_hash:
    {
        auto radius = GetFloat(node_shape, "radius").value_or(1);
        auto center = GetPoint(node_shape, "center").value_or(vec3(0));
        renderer->AddShapeInfo(new ShapeInfo(center, radius, flip_normals, to_world, bsdf_idx));
        break;
    }
    default:
        std::cerr << "[warning] " << GetTreeName(node_shape) << std::endl
                  << "\tcannot handle shape type, ignore " << std::endl;
        if (to_world)
            delete to_world;
        break;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

TextureInfo *ParseTextureOrOther(rapidxml::xml_node<> *node_parent, const std::string &name)
{
    auto node = GetChild(node_parent, name, true);
    if (!node)
        return nullptr;

    TextureInfo *new_texture = nullptr;
    auto node_type = node->name();
    switch (Hash(node_type))
    {
    case "float"_hash:
    {
        auto value = std::stof(GetAttri(node, "value").value());
        new_texture = new TextureInfo(value);
        break;
    }
    case "rgb"_hash:
    case "vec3"_hash:
    {
        auto value = GetVec3(node);
        new_texture = new TextureInfo(value);
        break;
    }
    case "texture"_hash:
    {
        new_texture = ParseTexture(node);
        break;
    }
    default:
        std::cerr << "[error] " << GetTreeName(node) << std::endl
                  << "\tcannot handle texture" << std::endl;
        exit(1);
    }
    return new_texture;
}

TextureInfo *ParseTexture(rapidxml::xml_node<> *node_texture)
{
    auto texture_type = GetAttri(node_texture, "type").value();
    switch (Hash(texture_type.c_str()))
    {
    case "bitmap"_hash:
    {
        auto node_filename = GetChild(node_texture, "filename", false);
        auto filename = xml_directory + ConvertBackSlash(GetAttri(node_filename, "value").value());
        auto gamma = GetFloat(node_texture, "gamma").value_or(-1);
        return new TextureInfo(filename, gamma);
        break;
    }
    default:
        std::cerr << "[error] " << GetTreeName(node_texture) << std::endl
                  << "\tcannot handle texture type except from bitmap, ignore it" << std::endl;
        break;
    }
    return nullptr;
}
