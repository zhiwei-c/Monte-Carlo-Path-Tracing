#include <map>
#include <vector>
#include <string>
#include <optional>
#include <iostream>

#include "rapidxml/rapidxml_utils.hpp"
#include "glm/gtx/matrix_query.hpp"

#include "file_path.h"
#include "../renderer.h"
#include "../core/ior.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////

Renderer *renderer = nullptr;
uint material_cnt;
uint texture_cnt;
std::string xml_directory;
std::map<std::string, uint> m_id_to_m_idx;

///////////////////////////////////////////////////////////////////////////////////////////////////////

static void ParseBsdf(rapidxml::xml_node<> *node_bsdf, const std::string *id_default = nullptr);

static bool ParseCoating(const std::string &id, rapidxml::xml_node<> *&node_bsdf, std::string &bsdf_type);

static MaterialInfo ParseDiffuse(rapidxml::xml_node<> *node_diffuse);

static MaterialInfo ParseDielectric(rapidxml::xml_node<> *node_dielectric, bool thin);

static MaterialInfo ParseRoughDielectric(rapidxml::xml_node<> *node_rough_dielectric);

static MaterialInfo ParseConductor(rapidxml::xml_node<> *node_conductor);

static MaterialInfo ParseRoughConductor(rapidxml::xml_node<> *node_rough_conductor);

static MaterialInfo ParsePlastic(rapidxml::xml_node<> *node_plastic);

static MaterialInfo ParseRoughPlastic(rapidxml::xml_node<> *node_rough_plastic);

//============================================================================================================

static void ParseShape(rapidxml::xml_node<> *node_shape);

static void ParseIntegrator(rapidxml::xml_node<> *node_integrator);

static CameraInfo ParseCamera(rapidxml::xml_node<> *node_sensor);

static void ParseEnvmap(rapidxml::xml_node<> *node_envmap, const CameraInfo &camera_info);

//============================================================================================================

static Float GetIor(rapidxml::xml_node<> *node_parent, std::string ior_type, std::string default_material_name);

static MicrofacetDistribType GetDistrbType(const std::string &name);

static TextureInfo *ParseTextureOrOther(rapidxml::xml_node<> *node_parent, const std::string &name);

static TextureInfo *ParseTexture(rapidxml::xml_node<> *node_texture);

static vec3 GetVec3(rapidxml::xml_node<> *node_vec3);

static gmat4 *GetToWorld(rapidxml::xml_node<> *node_parent);

static std::optional<bool> GetBoolean(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);

static std::optional<int> GetInt(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);

static std::optional<Float> GetFloat(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);

static std::optional<vec3> GetPoint(rapidxml::xml_node<> *node_parent, const std::string &name, bool not_exist_ok = true);

static std::optional<std::string> GetString(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);

//================================================================

static std::string GetTreeName(rapidxml::xml_node<> *node);

static std::optional<std::string> GetAttri(rapidxml::xml_node<> *node, std::string key, bool not_exist_ok = false);

static rapidxml::xml_node<> *GetChild(rapidxml::xml_node<> *node, std::string name, bool not_exist_ok = true);

///////////////////////////////////////////////////////////////////////////////////////////////////////

Renderer *ParseRenderConfig(const std::string &config_path)
{
    material_cnt = 0;
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
            auto new_colors = std::vector<float>(new_height * new_width * radiance->channel);
            stbir_resize_float(radiance->colors.data(), radiance->width, radiance->height, 0,
                               &new_colors[0], new_width, new_height, 0, radiance->channel);
            radiance->colors = new_colors;
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

//======================================================================================================

void ParseBsdf(rapidxml::xml_node<> *node_bsdf, const std::string *id_default)
{
    auto bsdf_type = GetAttri(node_bsdf, "type").value();

    auto bump_map_idx = static_cast<uint>(-1);
    //处理凹凸贴图
    if (bsdf_type == "bumpmap")
    {
        auto node_bump = node_bsdf->first_node("texture");
        if (node_bump->first_node("texture"))
            node_bump = node_bump->first_node("texture");
        auto bump_map = ParseTexture(node_bump);
        renderer->AddTextureInfo(bump_map);
        bump_map_idx = texture_cnt++;

        node_bsdf = node_bsdf->first_node("bsdf");
        if (!node_bsdf)
        {
            std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
                      << "\tnot enough bsdf information" << std::endl;
            exit(1);
        }
        bsdf_type = GetAttri(node_bsdf, "type").value();
    }

    //材质 ID
    auto id = id_default ? *id_default : GetAttri(node_bsdf, "id").value();

    //处理 mask
    auto opacity_idx = static_cast<uint>(-1);
    if (bsdf_type == "mask")
    {
        auto node_opacity = node_bsdf;
        auto opacity = ParseTextureOrOther(node_opacity, "opacity");
        if (!opacity)
        {
            std::cerr << "[error] " << GetTreeName(node_opacity) << std::endl
                      << "\tnot enough opacity information" << std::endl;
            exit(1);
        }
        if (opacity->type == kConstant &&
            (opacity->colors[0] != opacity->colors[1] ||
             opacity->colors[0] != opacity->colors[2] ||
             opacity->colors[1] != opacity->colors[2]))
        {
            std::cerr << "[error] " << GetTreeName(node_opacity) << std::endl
                      << "\tnot support different opacity for different color channel" << std::endl;
            exit(1);
        }

        renderer->AddTextureInfo(opacity);
        opacity_idx = texture_cnt++;

        node_bsdf = node_bsdf->first_node("bsdf");
        if (!node_bsdf)
        {
            std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
                      << "\tnot enough bsdf information" << std::endl;
            exit(1);
        }
        bsdf_type = GetAttri(node_bsdf, "type").value();
    }

    auto twosided = false;
    if (bsdf_type == "twosided")
    {
        twosided = true;
        node_bsdf = node_bsdf->first_node("bsdf");
        if (!node_bsdf)
        {
            std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
                      << "\tnot enough bsdf information" << std::endl;
            exit(1);
        }
        bsdf_type = GetAttri(node_bsdf, "type").value();
    }

    //处理 coating
    if (ParseCoating(id, node_bsdf, bsdf_type))
        return;

    auto material_info = MaterialInfo();
    switch (Hash(bsdf_type.c_str()))
    {
    case "diffuse"_hash:
        material_info = ParseDiffuse(node_bsdf);
        break;
    case "dielectric"_hash:
        twosided = true;
        material_info = ParseDielectric(node_bsdf, false);
        break;
    case "roughdielectric"_hash:
        twosided = true;
        material_info = ParseRoughDielectric(node_bsdf);
        break;
    case "thindielectric"_hash:
        twosided = true;
        material_info = ParseDielectric(node_bsdf, true);
        break;
    case "conductor"_hash:
        material_info = ParseConductor(node_bsdf);
        break;
    case "roughconductor"_hash:
        material_info = ParseRoughConductor(node_bsdf);
        break;
    case "plastic"_hash:
        material_info = ParsePlastic(node_bsdf);
        break;
    case "roughplastic"_hash:
        material_info = ParseRoughPlastic(node_bsdf);
        break;
    default:
        std::cerr << "[warning] " << GetTreeName(node_bsdf) << std::endl
                  << "\tconduct as diffuse" << std::endl;
        material_info = ParseDiffuse(node_bsdf);
    };

    material_info.twosided = twosided;
    material_info.bump_map_idx = bump_map_idx;
    material_info.opacity_idx = opacity_idx;
    renderer->AddMaterialInfo(material_info);
    m_id_to_m_idx[id] = material_cnt++;
}

bool ParseCoating(const std::string &id, rapidxml::xml_node<> *&node_bsdf, std::string &bsdf_type)
{

    if (bsdf_type != "coating" || bsdf_type != "roughcoating")
        return false;

    std::cerr << "[warning] not support coating bsdf, ignore it." << std::endl;
    if (auto node_ref = node_bsdf->first_node("ref"); node_ref)
    {
        if (node_ref->next_sibling("ref"))
        {
            std::cerr << "[error] " << GetTreeName(node_ref) << std::endl
                      << "\tfind multiple ref" << std::endl;
            exit(1);
        }
        auto ref_id = GetAttri(node_ref, "id").value();
        if (m_id_to_m_idx.find(ref_id) == m_id_to_m_idx.end())
        {
            std::cerr << "[error] " << GetTreeName(node_ref) << std::endl
                      << "\tcannot find existed material with id: " << ref_id << std::endl;
            exit(1);
        }
        else
        {
            m_id_to_m_idx[ref_id] = m_id_to_m_idx[id];
            return true;
        }
    }
    node_bsdf = node_bsdf->first_node("bsdf");
    if (!node_bsdf)
    {
        std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
                  << "\tnot enough bsdf information" << std::endl;
        exit(1);
    }
    bsdf_type = GetAttri(node_bsdf, "type").value();
    return false;
}

MaterialInfo ParseDiffuse(rapidxml::xml_node<> *node_diffuse)
{
    auto reflectance_idx = static_cast<uint>(-1);
    auto reflectance_info = ParseTextureOrOther(node_diffuse, "reflectance");
    if (reflectance_info)
    {
        renderer->AddTextureInfo(reflectance_info);
        reflectance_idx = texture_cnt++;
    }

    auto diffuse_info = MaterialInfo();
    diffuse_info.type = kDiffuse;
    diffuse_info.diffuse_reflectance_idx = reflectance_idx;
    return diffuse_info;
}

MaterialInfo ParseDielectric(rapidxml::xml_node<> *node_dielectric, bool thin)
{
    auto int_ior = GetIor(node_dielectric, "intIOR", "bk7");
    auto ext_ior = GetIor(node_dielectric, "extIOR", "air");

    auto specular_reflectance_idx = static_cast<uint>(-1);
    auto specular_reflectance = ParseTextureOrOther(node_dielectric, "specularReflectance");
    if (specular_reflectance)
    {
        renderer->AddTextureInfo(specular_reflectance);
        specular_reflectance_idx = texture_cnt++;
    }

    auto specular_transmittance_idx = static_cast<uint>(-1);
    auto specular_transmittance = ParseTextureOrOther(node_dielectric, "specularTransmittance");
    if (specular_transmittance)
    {
        renderer->AddTextureInfo(specular_transmittance);
        specular_transmittance_idx = texture_cnt++;
    }

    auto material_info = MaterialInfo();
    material_info.type = thin ? kThinDielectric : kDielectric;
    material_info.eta = vec3(int_ior / ext_ior);
    material_info.specular_reflectance_idx = specular_reflectance_idx;
    material_info.specular_transmittance_idx = specular_transmittance_idx;
    return material_info;
}

MaterialInfo ParseRoughDielectric(rapidxml::xml_node<> *node_rough_dielectric)
{
    auto int_ior = GetIor(node_rough_dielectric, "intIOR", "bk7");
    auto ext_ior = GetIor(node_rough_dielectric, "extIOR", "air");

    auto specular_reflectance_idx = static_cast<uint>(-1);
    auto specular_reflectance = ParseTextureOrOther(node_rough_dielectric, "specularReflectance");
    if (specular_reflectance)
    {
        renderer->AddTextureInfo(specular_reflectance);
        specular_reflectance_idx = texture_cnt++;
    }

    auto specular_transmittance_idx = static_cast<uint>(-1);
    auto specular_transmittance = ParseTextureOrOther(node_rough_dielectric, "specularTransmittance");
    if (specular_transmittance)
    {
        renderer->AddTextureInfo(specular_transmittance);
        specular_transmittance_idx = texture_cnt++;
    }

    auto distri = GetString(node_rough_dielectric, "distribution").value_or("beckmann");

    auto alpha_u_idx = static_cast<uint>(-1);
    auto alpha_u = ParseTextureOrOther(node_rough_dielectric, "alpha");
    if (!alpha_u)
        alpha_u = ParseTextureOrOther(node_rough_dielectric, "alphaU");
    if (alpha_u)
    {
        renderer->AddTextureInfo(alpha_u);
        alpha_u_idx = texture_cnt++;
    }

    auto alpha_v_idx = alpha_u_idx;
    auto alpha_v = ParseTextureOrOther(node_rough_dielectric, "alphaV");
    if (alpha_v)
    {
        renderer->AddTextureInfo(alpha_v);
        alpha_v_idx = texture_cnt++;
    }
    auto material_info = MaterialInfo();
    material_info.type = kRoughDielectric;
    material_info.eta = vec3(int_ior / ext_ior);
    material_info.specular_reflectance_idx = specular_reflectance_idx;
    material_info.specular_transmittance_idx = specular_transmittance_idx;
    material_info.distri = GetDistrbType(distri);
    material_info.alpha_u_idx = alpha_u_idx;
    material_info.alpha_v_idx = alpha_v_idx;
    return material_info;
}

MaterialInfo ParseConductor(rapidxml::xml_node<> *node_conductor)
{
    auto eta = vec3(0);
    auto k = vec3(1);
    auto ext_eta = GetIor(node_conductor, "extEta", "air");
    auto node_material = GetChild(node_conductor, "material");

    auto specular_reflectance_idx = static_cast<uint>(-1);
    auto specular_reflectance = ParseTextureOrOther(node_conductor, "specularReflectance");
    if (specular_reflectance)
    {
        renderer->AddTextureInfo(specular_reflectance);
        specular_reflectance_idx = texture_cnt++;
    }

    bool mirror = false;
    if (node_material)
    {
        auto material_name = GetAttri(node_material, "value").value();
        if (material_name == "none")
            mirror = true;
        else if (!LookupConductorIor(material_name, eta, k))
        {
            std::cerr << "[error] " << GetTreeName(node_material) << std::endl
                      << " unsupported material :" << material_name << ", "
                      << "use default Conductor material instead." << std::endl;
            exit(1);
        }
    }
    else if (node_conductor->first_node() != nullptr)
    {
        auto node_eta = GetChild(node_conductor, "eta", false);
        eta = GetVec3(node_eta);
        auto node_k = GetChild(node_conductor, "k", false);
        k = GetVec3(node_k);
    }
    else
        mirror = true;
    auto material_info = MaterialInfo();
    material_info.type = kConductor;
    material_info.mirror = mirror;
    material_info.eta = eta / ext_eta;
    material_info.k = k / ext_eta;
    material_info.specular_reflectance_idx = specular_reflectance_idx;
    return material_info;
}

MaterialInfo ParseRoughConductor(rapidxml::xml_node<> *node_rough_conductor)
{

    auto eta = vec3(0);
    auto k = vec3(1);
    auto ext_eta = GetIor(node_rough_conductor, "extEta", "air");
    auto node_material = GetChild(node_rough_conductor, "material");

    auto specular_reflectance_idx = static_cast<uint>(-1);
    auto specular_reflectance = ParseTextureOrOther(node_rough_conductor, "specularReflectance");
    if (specular_reflectance)
    {
        renderer->AddTextureInfo(specular_reflectance);
        specular_reflectance_idx = texture_cnt++;
    }

    bool mirror = false;
    if (node_material)
    {
        auto material_name = GetAttri(node_material, "value").value();
        if (material_name == "none")
            mirror = true;
        else if (!LookupConductorIor(material_name, eta, k))
        {
            std::cerr << "[error] " << GetTreeName(node_material) << std::endl
                      << " unsupported material :" << material_name << ", "
                      << "use default Conductor material instead." << std::endl;
            exit(1);
        }
    }
    else if (node_rough_conductor->first_node() != nullptr)
    {
        auto node_eta = GetChild(node_rough_conductor, "eta", false);
        eta = GetVec3(node_eta);
        auto node_k = GetChild(node_rough_conductor, "k", false);
        k = GetVec3(node_k);
    }
    else
        mirror = true;

    auto distri = GetString(node_rough_conductor, "distribution").value_or("beckmann");

    auto alpha_u_idx = static_cast<uint>(-1);
    auto alpha_u = ParseTextureOrOther(node_rough_conductor, "alpha");
    if (!alpha_u)
        alpha_u = ParseTextureOrOther(node_rough_conductor, "alphaU");
    if (alpha_u)
    {
        renderer->AddTextureInfo(alpha_u);
        alpha_u_idx = texture_cnt++;
    }

    auto alpha_v_idx = alpha_u_idx;
    auto alpha_v = ParseTextureOrOther(node_rough_conductor, "alphaV");
    if (alpha_v)
    {
        renderer->AddTextureInfo(alpha_v);
        alpha_v_idx = texture_cnt++;
    }

    auto material_info = MaterialInfo();
    material_info.type = kRoughConductor;
    material_info.mirror = mirror;
    material_info.eta = eta / ext_eta;
    material_info.k = k / ext_eta;
    material_info.specular_reflectance_idx = specular_reflectance_idx;
    material_info.distri = GetDistrbType(distri);
    material_info.alpha_u_idx = alpha_u_idx;
    material_info.alpha_v_idx = alpha_v_idx;
    return material_info;
}

MaterialInfo ParsePlastic(rapidxml::xml_node<> *node_plastic)
{
    auto int_ior = GetIor(node_plastic, "intIOR", "polypropylene");
    auto ext_ior = GetIor(node_plastic, "extIOR", "air");

    auto diffuse_reflectance_idx = static_cast<uint>(-1);
    auto diffuse_reflectance = ParseTextureOrOther(node_plastic, "diffuseReflectance");
    if (diffuse_reflectance)
    {
        renderer->AddTextureInfo(diffuse_reflectance);
        diffuse_reflectance_idx = texture_cnt++;
    }

    auto specular_reflectance_idx = static_cast<uint>(-1);
    auto specular_reflectance = ParseTextureOrOther(node_plastic, "specularReflectance");
    if (specular_reflectance)
    {
        renderer->AddTextureInfo(specular_reflectance);
        specular_reflectance_idx = texture_cnt++;
    }

    bool nonlinear = GetBoolean(node_plastic, "nonlinear").value_or(false);

    auto material_info = MaterialInfo();
    material_info.type = kPlastic;
    material_info.eta = vec3(int_ior / ext_ior);
    material_info.diffuse_reflectance_idx = diffuse_reflectance_idx;
    material_info.specular_reflectance_idx = specular_reflectance_idx;
    material_info.nonlinear = nonlinear;
    return material_info;
}

MaterialInfo ParseRoughPlastic(rapidxml::xml_node<> *node_rough_plastic)
{
    auto int_ior = GetIor(node_rough_plastic, "intIOR", "polypropylene");
    auto ext_ior = GetIor(node_rough_plastic, "extIOR", "air");

    auto diffuse_reflectance_idx = static_cast<uint>(-1);
    auto diffuse_reflectance = ParseTextureOrOther(node_rough_plastic, "diffuseReflectance");
    if (diffuse_reflectance)
    {
        renderer->AddTextureInfo(diffuse_reflectance);
        diffuse_reflectance_idx = texture_cnt++;
    }

    auto specular_reflectance_idx = static_cast<uint>(-1);
    auto specular_reflectance = ParseTextureOrOther(node_rough_plastic, "specularReflectance");
    if (specular_reflectance)
    {
        renderer->AddTextureInfo(specular_reflectance);
        specular_reflectance_idx = texture_cnt++;
    }

    bool nonlinear = GetBoolean(node_rough_plastic, "nonlinear").value_or(false);

    auto distri = GetString(node_rough_plastic, "distribution").value_or("beckmann");

    auto alpha_u_idx = static_cast<uint>(-1);
    auto alpha_u = ParseTextureOrOther(node_rough_plastic, "alpha");
    if (!alpha_u)
        alpha_u = ParseTextureOrOther(node_rough_plastic, "alphaU");
    if (alpha_u)
    {
        renderer->AddTextureInfo(alpha_u);
        alpha_u_idx = texture_cnt++;
    }

    auto alpha_v_idx = alpha_u_idx;
    auto alpha_v = ParseTextureOrOther(node_rough_plastic, "alphaV");
    if (alpha_v)
    {
        renderer->AddTextureInfo(alpha_v);
        alpha_v_idx = texture_cnt++;
    }

    auto material_info = MaterialInfo();
    material_info.type = kRoughPlastic;
    material_info.eta = vec3(int_ior / ext_ior);
    material_info.diffuse_reflectance_idx = diffuse_reflectance_idx;
    material_info.specular_reflectance_idx = specular_reflectance_idx;
    material_info.distri = GetDistrbType(distri);
    material_info.alpha_u_idx = alpha_u_idx;
    material_info.alpha_v_idx = alpha_v_idx;
    material_info.nonlinear = nonlinear;
    return material_info;
}

//========================================================================================================================

void ParseShape(rapidxml::xml_node<> *node_shape)
{
    std::string ref;
    if (auto node_emitter = node_shape->first_node("emitter"); node_emitter)
    {
        ref = "unnamed_" + std::to_string(material_cnt);
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

        auto area_light_info = MaterialInfo();
        area_light_info.type = kAreaLight;
        area_light_info.radiance_idx = texture_cnt++;
        renderer->AddMaterialInfo(area_light_info);
        m_id_to_m_idx[ref] = material_cnt++;
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
        ref = "unnamed_" + std::to_string(material_cnt);
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
                  << "\tcannot find material with name: \"" << ref << "\"" << std::endl;
        exit(1);
    }
    auto material_idx = m_id_to_m_idx[ref];

    auto type = GetAttri(node_shape, "type").value();
    auto to_world = GetToWorld(node_shape);
    auto flip_normals = GetBoolean(node_shape, "flipNormals").value_or(false);

    switch (Hash(type.c_str()))
    {
    case "cube"_hash:
        renderer->AddShapeInfo(new ShapeInfo(kCube, flip_normals, to_world, material_idx));
        break;
    case "disk"_hash:
        renderer->AddShapeInfo(new ShapeInfo(kDisk, flip_normals, to_world, material_idx));
        break;
    case "obj"_hash:
    {
        auto face_normals = GetBoolean(node_shape, "faceNormals").value_or(false);
        auto filename = xml_directory + GetAttri(GetChild(node_shape, "filename", false), "value").value();
        auto flip_tex_coords = GetBoolean(node_shape, "flipTexCoords").value_or(true);
        renderer->AddShapeInfo(new ShapeInfo(filename, face_normals, flip_normals, flip_tex_coords, to_world, material_idx));
        break;
    }
    case "ply"_hash:
    {
        auto face_normals = GetBoolean(node_shape, "faceNormals").value_or(false);
        auto filename = xml_directory + GetAttri(GetChild(node_shape, "filename", false), "value").value();
        auto flip_tex_coords = GetBoolean(node_shape, "flipTexCoords").value_or(true);
        renderer->AddShapeInfo(new ShapeInfo(filename, face_normals, flip_normals, flip_tex_coords, to_world, material_idx));
        break;
    }
    case "rectangle"_hash:
        renderer->AddShapeInfo(new ShapeInfo(kRectangle, flip_normals, to_world, material_idx));
        break;
    case "sphere"_hash:
    {
        auto radius = GetFloat(node_shape, "radius").value_or(1);
        auto center = GetPoint(node_shape, "center").value_or(vec3(0));
        renderer->AddShapeInfo(new ShapeInfo(center, radius, flip_normals, to_world, material_idx));
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

Float GetIor(rapidxml::xml_node<> *node_parent, std::string ior_type, std::string default_material_name)
{
    Float ior = 0;
    auto node_ior = GetChild(node_parent, ior_type);
    if (!node_ior)
        LookupDielectricIor(default_material_name, ior);
    else if (strcmp(node_ior->name(), "float") == 0)
        return std::stof(GetAttri(node_ior, "value").value());
    else
    {
        auto int_ior_name = GetAttri(node_ior, "value").value();
        if (!LookupDielectricIor(int_ior_name, ior))
        {
            std::cerr << "[error] " << GetTreeName(node_ior) << std::endl
                      << "\tunsupported ior material " << int_ior_name << std::endl;
            exit(1);
        }
    }
    return ior;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vec3 GetVec3(rapidxml::xml_node<> *node_vec3)
{
    if (strcmp(node_vec3->name(), "rgb") != 0)
    {
        std::cerr << "[error] " << GetTreeName(node_vec3) << std::endl
                  << "\tcannot hanle  vec3 except from rgb" << std::endl;
        exit(1);
    }

    auto value_str = GetAttri(node_vec3, "value").value();
    vec3 result;
    sscanf(value_str.c_str(), "%lf, %lf, %lf", &result[0], &result[1], &result[2]);
    return result;
}

gmat4 *GetToWorld(rapidxml::xml_node<> *node_parent)
{
    auto node_toworld = GetChild(node_parent, "toWorld");
    if (!node_toworld)
        return nullptr;

    auto node_matrix = node_toworld->first_node("matrix");
    if (!node_matrix || node_matrix->next_sibling() || node_matrix->previous_sibling())
    {
        std::cerr << "[error] " << GetTreeName(node_matrix) << std::endl
                  << "\tcannot handle transform except from matrix or find multiple matrix" << std::endl;
        exit(1);
    }
    auto matrix_str = GetAttri(node_matrix, "value").value();
    gmat4 result;
    sscanf(matrix_str.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
           &result[0][0], &result[1][0], &result[2][0], &result[3][0],
           &result[0][1], &result[1][1], &result[2][1], &result[3][1],
           &result[0][2], &result[1][2], &result[2][2], &result[3][2],
           &result[0][3], &result[1][3], &result[2][3], &result[3][3]);

    if (glm::isIdentity(result, kEpsilon))
        return nullptr;
    else
        return new gmat4(result);
}

std::optional<bool> GetBoolean(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok)
{
    auto node_boolean = GetChild(node_parent, name);
    if (!node_boolean)
    {
        if (not_exist_ok)
            return std::nullopt;
        else
        {
            std::cerr << "[error] " << GetTreeName(node_parent) << std::endl
                      << "\tcannot find child node:" << name << std::endl;
            exit(1);
        }
    }
    auto value = GetAttri(node_boolean, "value");
    if (value == "true")
        return true;
    else
        return false;
}

std::optional<int> GetInt(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok)
{
    auto node_int = GetChild(node_parent, name);
    if (!node_int)
    {
        if (not_exist_ok)
            return std::nullopt;
        else
        {
            std::cerr << "[error] " << GetTreeName(node_parent) << std::endl
                      << "\tcannot find child node: " << name << std::endl;
            exit(1);
        }
    }
    if (strcmp(node_int->name(), "integer") != 0)
    {
        std::cerr << "[error] " << GetTreeName(node_int) << std::endl
                  << "\tthe type of \"" << name << "\" provided is not integer" << std::endl;
        exit(1);
    }

    return std::stoi(GetAttri(node_int, "value").value());
}

std::optional<Float> GetFloat(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok)
{
    auto node_float = GetChild(node_parent, name);
    if (!node_float)
    {
        if (not_exist_ok)
            return std::nullopt;
        else
        {
            std::cerr << "[error] " << GetTreeName(node_parent) << std::endl
                      << "\tcannot find child node: " << name << std::endl;
            exit(1);
        }
    }

    if (strcmp(node_float->name(), "float") != 0)
    {
        std::cerr << "[error] " << GetTreeName(node_float) << std::endl
                  << "\tthe type of \"" << name << "\" provided is not float" << std::endl;
        exit(1);
    }

    return std::stof(GetAttri(node_float, "value").value());
}

std::optional<vec3> GetPoint(rapidxml::xml_node<> *node_parent, const std::string &name, bool not_exist_ok)
{
    auto node_point = GetChild(node_parent, name);
    if (!node_point)
    {
        if (not_exist_ok)
            return std::nullopt;
        else
        {
            std::cerr << "[error] " << GetTreeName(node_parent) << std::endl
                      << "\tcannot find child node: " << name << std::endl;
            exit(1);
        }
    }

    if (strcmp(node_point->name(), "point") != 0)
    {
        std::cerr << "[error] " << GetTreeName(node_point) << std::endl
                  << "\tthe type of \"" << name << "\" provided is not point" << std::endl;
        exit(1);
    }

    vec3 result;

    result.x = static_cast<Float>(std::stod(GetAttri(node_point, "x").value()));
    result.y = static_cast<Float>(std::stod(GetAttri(node_point, "y").value()));
    result.z = static_cast<Float>(std::stod(GetAttri(node_point, "z").value()));

    return result;
}

std::optional<std::string> GetString(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok)
{
    auto node_string = GetChild(node_parent, name);
    if (!node_string)
    {
        if (not_exist_ok)
            return std::nullopt;
        else
        {
            std::cerr << "[error] " << GetTreeName(node_parent) << std::endl
                      << "\tcannot find child node :" << name << std::endl;
            exit(1);
        }
    }
    else
        return GetAttri(node_string, "value");
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

rapidxml::xml_node<> *GetChild(rapidxml::xml_node<> *node, std::string name, bool not_exist_ok)
{
    for (auto child = node->first_node(); child; child = child->next_sibling())
    {
        auto child_name = GetAttri(child, "name", not_exist_ok);
        if (child_name.has_value() && child_name.value() == name)
            return child;
    }
    if (not_exist_ok)
        return nullptr;
    else
    {
        std::cerr << "[error] " << GetTreeName(node) << std::endl
                  << "\tcannot find child node :" << name << std::endl;
        exit(1);
    }
}

std::optional<std::string> GetAttri(rapidxml::xml_node<> *node, std::string key, bool not_exist_ok)
{
    auto attri = node->first_attribute(key.c_str());
    if (!attri)
    {
        if (not_exist_ok)
            return std::nullopt;
        std::cerr << "[error] " << GetTreeName(node) << std::endl
                  << "\tcannot find " << key << std::endl;
        exit(1);
    }
    if (attri->next_attribute(key.c_str()))
    {
        std::cerr << "[error] " << GetTreeName(node) << std::endl
                  << "\tfind multiple " << key << std::endl;
        exit(1);
    }
    return attri->value();
}

std::string GetTreeName(rapidxml::xml_node<> *node)
{
    if (!node || node->name_size() == 0)
        return "root";
    else
    {
        auto result = GetTreeName(node->parent()) + " --> " + node->name();
        if (auto attri_name = node->first_attribute("name"); attri_name)
            result = result + ":" + attri_name->value();
        if (auto attri_type = node->first_attribute("type"); attri_type)
            result = result + ":" + attri_type->value();
        if (auto attri_id = node->first_attribute("id"); attri_id)
            result = result + ":" + attri_id->value();
        return result;
    }
}

MicrofacetDistribType GetDistrbType(const std::string &name)
{
    switch (Hash(name.c_str()))
    {
    case "beckmann"_hash:
        return kBeckmann;
        break;
    case "ggx"_hash:
        return kGgx;
        break;
    default:
        std::cout << "[warning] unkown microfacet distribution: " << name << ", use Beckmann instead.";
        return kBeckmann;
        break;
    }
}