#include "config_parser.hpp"

#include <iostream>
#include <algorithm>
#include <unordered_map>

#include <pugixml.hpp>
#include <glm/gtx/transform.hpp>

#include "file_path.hpp"
#include "image.hpp"
#include "object_loader.hpp"
#include "serialized_loader.hpp"
#include "../core/ior.hpp"
#include "../math/coordinate.hpp"
#include "../math/math.hpp"
#include "../media/medium_data.hpp"
#include "../utils/sun_model.hpp"

NAMESPACE_BEGIN(raytracer)

static int width = 768, height = 576;
static double fov_x;
static size_t anonymous_shape_num = 0;
static size_t anonymous_medium_num = 0;
static size_t anonymous_bsdf_num = 0;
static size_t anonymous_texture_num = 0;
static std::string config_file_directory = "";
static std::unordered_map<std::string, std::string> default_mp;
static std::unordered_map<std::string, Texture *> texture_mp;
static std::unordered_map<std::string, Bsdf *> bsdf_mp;
static std::unordered_map<std::string, Medium *> medium_mp;

void ReadIntegrator(pugi::xml_node integrator_node, Renderer &renderer);
void ReadFilm(pugi::xml_node sensor_node, Renderer &renderer);
void ReadCamera(pugi::xml_node sensor_node, Renderer &renderer);
void ReadEmitter(pugi::xml_node emitter_node, Renderer &renderer);
Texture *ReadTexture(const pugi::xml_node &parent_node, const std::vector<std::string> &valid_names, Renderer &renderer,
                     double defalut_value);
Texture *ReadTexture(const pugi::xml_node &texture_node, Renderer &renderer, double defalut_value);
Bsdf *ReadBsdf(pugi::xml_node bsdf_node, Renderer &renderer);
Shape *ReadShape(pugi::xml_node shape_node, Renderer &renderer);

Medium *ReadMedium(const pugi::xml_node &medium_node, Renderer &renderer);
PhaseFunction *ReadPhaseFunction(const pugi::xml_node &parent_node);

Ndf *ReadNdf(const pugi::xml_node &parent_node, Renderer &renderer, double default_roughness, bool force_roughness = false);
double ReadDielectricIor(const pugi::xml_node &parent_node, const std::vector<std::string> &valid_names, double defalut_value);
void ReadConductorIor(const pugi::xml_node &parent_node, dvec3 *eta, dvec3 *k);

dmat4 ReadTransform4(const pugi::xml_node &transform_node);
dmat4 ReadMatrix4(const pugi::xml_node &matrix_node);
dvec3 ReadVec3(const pugi::xml_node &vec3_node, dvec3 defalut_value = {1.0, 1.0, 1.0}, const char *value_name = nullptr);
dvec3 ReadVec3(const pugi::xml_node &parent_node, const std::vector<std::string> &valid_names, dvec3 defalut_value = {1.0, 1.0, 1.0});
double ReadDouble(const pugi::xml_node &parent_node, const std::vector<std::string> &valid_names, double defalut_value);
bool ReadBoolean(const pugi::xml_node &parent_node, const std::vector<std::string> &valid_names, bool defalut_value);
int ReadInt(const pugi::xml_node &parent_node, const std::vector<std::string> &valid_names, int defalut_value);

bool GetChildByName(pugi::xml_node parent_node, const std::vector<std::string> &valid_names, pugi::xml_node *child_node = nullptr);

void LoadMitsubaConfig(const std::string &filename, Renderer &renderer)
{
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(filename.c_str());
    if (!result)
    {
        std::cerr << "[error] read \"" << filename << "\" failed";
        exit(1);
    }
    config_file_directory = GetDirectory(filename);

    pugi::xml_node scene_node = doc.child("scene");

    default_mp.clear();
    for (pugi::xml_node node : scene_node.children("default"))
    {
        std::string name = node.attribute("name").value();
        std::string value = node.attribute("value").value();
        default_mp["$" + name] = value;
    }

    ReadIntegrator(scene_node.child("integrator"), renderer);
    ReadFilm(scene_node.child("sensor"), renderer);
    ReadCamera(scene_node.child("sensor"), renderer);

    for (pugi::xml_node emitter_node : scene_node.children("emitter"))
    {
        ReadEmitter(emitter_node, renderer);
    }

    medium_mp.clear();
    anonymous_medium_num = 0;
    for (pugi::xml_node medium_node : scene_node.children("medium"))
    {
        ReadMedium(medium_node, renderer);
    }

    texture_mp.clear();
    anonymous_texture_num = 0;
    for (pugi::xml_node texture_node : scene_node.children("texture"))
    {
        Texture *texture = ReadTexture(texture_node, renderer, 1.0);
        texture_mp[texture->id()] = texture;
    }

    bsdf_mp.clear();
    anonymous_bsdf_num = 0;
    for (pugi::xml_node bsdf_node : scene_node.children("bsdf"))
    {
        Bsdf *bsdf = ReadBsdf(bsdf_node, renderer);
        if (bsdf != nullptr)
        {
            bsdf_mp[bsdf->id()] = bsdf;
        }
    }

    anonymous_shape_num = 0;
    for (pugi::xml_node shape_node : scene_node.children("shape"))
    {
        Shape *shape = ReadShape(shape_node, renderer);
        if (shape != nullptr)
        {
            renderer.AddShape(shape);
        }
    }
}

void ReadIntegrator(pugi::xml_node integrator_node, Renderer &renderer)
{
    std::string integrator_type = integrator_node.attribute("type").as_string("path");
    if (!integrator_type.empty() && integrator_type[0] == '$' && default_mp.count(integrator_type))
    {
        integrator_type = default_mp.at(integrator_type);
    }

    int max_depth = -1, rr_depth = 5;
    for (pugi::xml_node node : integrator_node.children("integer"))
    {
        switch (Hash(node.attribute("name").value()))
        {
        case "maxDepth"_hash:
        case "max_depth"_hash:
        {
            std::string value = node.attribute("value").value();
            if (!value.empty())
            {
                if (value[0] == '$')
                {
                    assert(default_mp.count(value));
                    value = default_mp.at(value);
                }
                max_depth = std::stoi(value);
            }
            break;
        }
        case "rrDepth"_hash:
        case "rr_depth"_hash:
        {
            std::string value = node.attribute("value").value();
            if (!value.empty())
            {
                if (value[0] == '$')
                {
                    assert(default_mp.count(value));
                    value = default_mp.at(value);
                }
                rr_depth = std::stoi(value);
            }
            break;
        }
        default:
            break;
        }
    }

    bool hide_emitters = ReadBoolean(integrator_node, {"hide_emitters", "hideEmitters"}, false);
    switch (Hash(integrator_type.c_str()))
    {
    case "bdpt"_hash:
        renderer.SetIntegratorInfo(IntegratorType::kBdpt, max_depth, rr_depth, hide_emitters);
        break;
    case "volpath"_hash:
        renderer.SetIntegratorInfo(IntegratorType::kVolPath, max_depth, rr_depth, hide_emitters);
        break;
    case "path"_hash:
        renderer.SetIntegratorInfo(IntegratorType::kPath, max_depth, rr_depth, hide_emitters);
        break;
    default:
        std::cerr << "[warning] unsupport integrator type \"" << integrator_type << "\""
                  << "use \"path\" instead.\n";
        renderer.SetIntegratorInfo(IntegratorType::kPath, max_depth, rr_depth, hide_emitters);
        break;
    }
}

void ReadFilm(pugi::xml_node sensor_node, Renderer &renderer)
{
    std::string type = sensor_node.attribute("type").as_string();
    if (type != "perspective")
    {
        std::cerr << "[error] only support perspective sensor";
        exit(1);
    }

    fov_x = ReadDouble(sensor_node, {"fov"}, -1);
    double focal_length = 50;
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
            fov_axis = node.attribute("value").as_string();
        default:
            break;
        }
    }

    width = 768, height = 576;
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
                    assert(default_mp.count(value_str));
                    value_str = default_mp.at(value_str);
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
                    assert(default_mp.count(value_str));
                    value_str = default_mp.at(value_str);
                }
                height = std::stoi(value_str);
            }
            break;
        }
        default:
            break;
        }
    }

    switch (Hash(fov_axis.c_str()))
    {
    case "x"_hash:
        if (fov_x <= 0)
        {
            fov_x = 2 * glm::atan(36 * 0.5 / focal_length) * 180 * kPiRcp;
        }
        break;
    case "y"_hash:
        if (fov_x <= 0)
        {
            fov_x = 2 * glm::atan(24 * 0.5 / focal_length) * 180 * kPiRcp;
        }
        fov_x = fov_x * width / height;
        break;
    case "smaller"_hash:
    {
        if (width > height)
        {
            if (fov_x <= 0)
            {
                fov_x = 2 * glm::atan(24 * 0.5 / focal_length) * 180 * kPiRcp;
            }
            fov_x = fov_x * width / height;
        }
    }
    default:
        std::cerr << "[error] unsupport fov axis \"" << fov_axis << "\"";
        exit(1);
    }
    renderer.SetFilm(width, height, fov_x);
}

void ReadCamera(pugi::xml_node sensor_node, Renderer &renderer)
{
    int spp = 4;
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
                    assert(default_mp.count(value_str));
                    value_str = default_mp.at(value_str);
                }
                spp = std::stoi(value_str);
            }
            break;
        }
        default:
            break;
        }
    }

    if (sensor_node.child("transform"))
    {
        dmat4 to_world = ReadTransform4(sensor_node.child("transform"));
        renderer.SetCamera(to_world, spp);
    }
}

void ReadEmitter(pugi::xml_node emitter_node, Renderer &renderer)
{
    std::string type = emitter_node.attribute("type").as_string();
    switch (Hash(type.c_str()))
    {
    case "sun"_hash:
    case "sky"_hash:
    case "sunsky"_hash:
    {
        auto sun_direction = dvec3(0);
        pugi::xml_node sun_direction_node;
        if (GetChildByName(emitter_node, {"sunDirection"}, &sun_direction_node))
        {
            sun_direction = {
                sun_direction_node.attribute("x").as_double(),
                sun_direction_node.attribute("y").as_double(),
                sun_direction_node.attribute("z").as_double()};
        }
        else
        {
            LocationDataInfo location_time;
            location_time.year = ReadInt(emitter_node, {"year"}, 2010);
            location_time.month = ReadInt(emitter_node, {"month"}, 7);
            location_time.day = ReadInt(emitter_node, {"day"}, 10);
            location_time.hour = ReadDouble(emitter_node, {"hour"}, 15);
            location_time.minute = ReadDouble(emitter_node, {"minute"}, 0);
            location_time.second = ReadDouble(emitter_node, {"second"}, 0);
            location_time.latitude = ReadDouble(emitter_node, {"latitude"}, 35.6894);
            location_time.longitude = ReadDouble(emitter_node, {"longitude"}, 139.6917);
            location_time.timezone = ReadDouble(emitter_node, {"timezone"}, 9);
            sun_direction = GetSunDirection(location_time);
        }

        dvec3 albedo = ReadVec3(emitter_node, {"albedo"}, dvec3(0.15));

        double turbidity = ReadDouble(emitter_node, {"turbidity"}, 3);
        turbidity = std::min(std::max(turbidity, 10.0), 1.0);

        double stretch = ReadDouble(emitter_node, {"stretch"}, 1);
        stretch = std::min(std::max(turbidity, 2.0), 1.0);

        int resolution = ReadInt(emitter_node, {"resolution"}, 512);

        double sun_scale = ReadDouble(emitter_node, {"sunScale"}, 1);
        double sky_scale = ReadDouble(emitter_node, {"skyScale"}, 1);
        double sun_radius_scale = ReadDouble(emitter_node, {"sunRadiusScale"}, 1);
        bool extend = ReadBoolean(emitter_node, {"extend"}, true);
        if (type == "sky" || type == "sunsky")
        {
            Sky *sky = new Sky(sun_direction, albedo, turbidity, stretch, sun_scale, sky_scale, sun_radius_scale,
                               resolution, extend);
            renderer.AddEmitter(sky);
        }
        if (type == "sun" || type == "sunsky")
        {
            Sun *sun = new Sun(sun_direction, turbidity, resolution, sun_scale, sun_radius_scale);
            renderer.AddEmitter(sun);
        }
        break;
    }
    case "constant"_hash:
    {
        Texture *radiance = ReadTexture(emitter_node, {"radiance"}, renderer, 1.0);
        Envmap *envmap = new Envmap(radiance, 1.0, dmat4(1));
        renderer.AddEmitter(envmap);
        break;
    }
    case "envmap"_hash:
    {
        std::string filename = emitter_node.child("string").attribute("value").as_string();
        int target_width = static_cast<int>(width * 360 / fov_x);
        Texture *radiance = LoadImage(config_file_directory + filename, filename, &target_width);
        dmat4 to_world = ReadTransform4(emitter_node.child("transform"));
        double scale = ReadDouble(emitter_node, {"scale"}, 1.0);
        Envmap *envmap = new Envmap(radiance, scale, to_world);
        renderer.AddEmitter(envmap);
        break;
    }
    case "spot"_hash:
    {
        dvec3 intensity = ReadVec3(emitter_node, {"intensity"}, dvec3(1));
        double cutoff_angle = ReadDouble(emitter_node, {"cutoff_angle", "cutoffAngle"}, 20);
        double beam_width = ReadDouble(emitter_node, {"beamWidth", "beam_width"}, cutoff_angle * 0.75);
        dmat4 to_world = ReadTransform4(emitter_node.child("transform"));
        Texture *texture = nullptr;
        if (emitter_node.child("texture"))
        {
            texture = ReadTexture(emitter_node.child("texture"), renderer, 1.0);
        }
        Emitter *emitter = new SpotLight(intensity, glm::radians(cutoff_angle), glm::radians(beam_width), texture, to_world);
        renderer.AddEmitter(emitter);
        break;
    }
    case "directional"_hash:
    {
        dvec3 radiance = ReadVec3(emitter_node, {"radiance"}, dvec3(1));
        dmat4 to_world = ReadTransform4(emitter_node.child("transform"));
        dvec3 direction = ReadVec3(emitter_node, {"direction"}, dvec3{0, 0, 1});
        direction = TransfromVec(glm::inverse(glm::transpose(to_world)), direction);
        Emitter *emitter = new DistantDirectionalEmitter(radiance, direction);
        renderer.AddEmitter(emitter);
        break;
    }
    case "point"_hash:
    {
        dvec3 intensity = ReadVec3(emitter_node, {"intensity"}, dvec3(1));
        dvec3 position = ReadVec3(emitter_node, {"position"}, dvec3(0));
        dmat4 to_world = ReadTransform4(emitter_node.child("transform"));
        position = TransfromPoint(to_world, position);

        Emitter *emitter = new PointLight(intensity, position);
        renderer.AddEmitter(emitter);
        break;
    }
    default:
        std::cerr << "[warning] unsupport emitter \"" << type << "\", ignore it\n";
        break;
    }
}

Texture *ReadTexture(const pugi::xml_node &parent_node, const std::vector<std::string> &valid_names, Renderer &renderer,
                     double defalut_value)
{
    pugi::xml_node texture_node;
    if (GetChildByName(parent_node, valid_names, &texture_node))
    {
        return ReadTexture(texture_node, renderer, defalut_value);
    }
    else
    {
        const std::string id = "anonymous_texture_" + std::to_string(anonymous_texture_num++);
        Texture *texture = new ConstantTexture(id, defalut_value);
        renderer.AddTexture(texture);
        return texture;
    }
}

Texture *ReadTexture(const pugi::xml_node &texture_node, Renderer &renderer, double defalut_value)
{
    Texture *texture = nullptr;

    std::string id = texture_node.attribute("id").as_string();
    if (!texture_node)
    {
        if (id.empty())
        {
            id = "anonymous_texture_" + std::to_string(anonymous_texture_num++);
        }
        texture = new ConstantTexture(id, defalut_value);
        renderer.AddTexture(texture);
        return texture;
    }

    std::string node_name = texture_node.name();
    switch (Hash(node_name.c_str()))
    {
    case "ref"_hash:
    {
        if (!texture_mp.count(id))
        {
            std::cerr << "[error] cannot find texture with id \"" << id << "\"";
            exit(1);
        }
        texture = texture_mp.at(id);
        break;
    }
    case "rgb"_hash:
    {
        if (id.empty())
        {
            id = "anonymous_texture_" + std::to_string(anonymous_texture_num++);
        }

        dvec3 value = ReadVec3(texture_node, dvec3(defalut_value));
        texture = new ConstantTexture(id, value);
        renderer.AddTexture(texture);
        break;
    }
    case "float"_hash:
    {
        if (id.empty())
        {
            id = "anonymous_texture_" + std::to_string(anonymous_texture_num++);
        }
        double value = texture_node.attribute("value").as_double(defalut_value);
        texture = new ConstantTexture(id, value);
        renderer.AddTexture(texture);
        break;
    }
    case "texture"_hash:
    {
        switch (Hash(texture_node.attribute("type").value()))
        {
        case "checkerboard"_hash:
        {
            Texture *color0 = ReadTexture(texture_node, {"color0"}, renderer, 0.4);
            Texture *color1 = ReadTexture(texture_node, {"color1"}, renderer, 0.2);
            auto to_uv_temp = ReadTransform4(texture_node.child("transform"));
            double u_offset = ReadDouble(texture_node, {"uoffset"}, 0.0),
                   v_offset = ReadDouble(texture_node, {"voffset"}, 0.0),
                   u_scale = ReadDouble(texture_node, {"uscale"}, 1.0),
                   v_scale = ReadDouble(texture_node, {"vscale"}, 1.0);
            to_uv_temp = glm::translate(dvec3{u_offset, v_offset, 0}) * to_uv_temp;
            to_uv_temp = glm::scale(dvec3{u_scale, v_scale, 1}) * to_uv_temp;

            dmat3 to_uv = dmat3(to_uv_temp);

            texture = new Checkerboard(id, color0, color1, to_uv);
            renderer.AddTexture(texture);
            break;
        }
        case "bitmap"_hash:
        {
            pugi::xml_node child_node;
            if (!GetChildByName(texture_node, {"filename"}, &child_node))
            {
                std::cerr << "[error] cannot find filename for bitmap texture";
                exit(1);
            }
            if (id.empty())
            {
                id = child_node.attribute("value").as_string();
            }
            texture = LoadImage(config_file_directory + child_node.attribute("value").as_string(), id);
            renderer.AddTexture(texture);
            break;
        }
        default:
        {
            std::cerr << "[error] unsupport texture type \"" << texture_node.attribute("type").value() << "\"";
            exit(1);
            break;
        }
        }
        break;
    }
    default:
    {
        std::cerr << "[error] unsupport texture type \"" << texture_node.attribute("type").value() << "\"";
        exit(1);
        break;
    }
    }
    return texture;
}

Bsdf *ReadBsdf(pugi::xml_node bsdf_node, Renderer &renderer)
{
    std::string id = bsdf_node.attribute("id").as_string();
    std::string type = bsdf_node.attribute("type").as_string();

    Texture *nump_map = nullptr;
    if (type == std::string("bumpmap"))
    {
        pugi::xml_node texture_node = bsdf_node.child("texture");
        if (bsdf_node.child("texture").attribute("type").as_string() == std::string("scale"))
        {
            texture_node = texture_node.child("texture");
        }
        nump_map = ReadTexture(texture_node, renderer, 1.0);
        bsdf_node = bsdf_node.child("bsdf");
        id = bsdf_node.attribute("id").as_string(id.c_str());
    }

    Texture *opacity = nullptr;
    if (type == std::string("mask"))
    {
        opacity = ReadTexture(bsdf_node, {"opacity"}, renderer, 1.0);
        bsdf_node = bsdf_node.child("bsdf");
        id = bsdf_node.attribute("id").as_string(id.c_str());
    }

    bool twosided = {type == "twosided"};
    if (bsdf_node.child("bsdf"))
    {
        bsdf_node = bsdf_node.child("bsdf");
        type = bsdf_node.attribute("type").value();
    }
    if (type == "null")
    {
        return nullptr;
    }

    if (id.empty())
    {
        id = "anonymous_bsdf_" + std::to_string(anonymous_bsdf_num++);
    }

    Bsdf *bsdf = nullptr;
    switch (Hash(type.c_str()))
    {
    case "diffuse"_hash:
    {
        Texture *reflectance = ReadTexture(bsdf_node, {"reflectance"}, renderer, 0.5);
        bsdf = new Diffuse(id, reflectance);
        break;
    }
    case "dielectric"_hash:
    {
        twosided = true;
        double int_ior = ReadDielectricIor(bsdf_node, {"int_ior", "intIOR"}, 1.5046);
        double ext_ior = ReadDielectricIor(bsdf_node, {"ext_ior", "extIOR"}, 1.000277);
        Texture *specular_reflectance = ReadTexture(bsdf_node, {"specularReflectance", "specular_reflectance"}, renderer, 1.0);
        Texture *specular_transmittance = ReadTexture(bsdf_node, {"specularTransmittance", "specular_transmittance"}, renderer, 1.0);
#ifdef ROUGH_SMOOTH
        Ndf *ndf = ReadNdf(bsdf_node, renderer, 0.001, true);
        bsdf = new RoughDielectric(id, int_ior, ext_ior, ndf, specular_reflectance, specular_transmittance);
#else
        bsdf = new Dielectric(id, int_ior, ext_ior, specular_reflectance, specular_transmittance);
#endif
        break;
    }
    case "thindielectric"_hash:
    {
        twosided = true;
        double int_ior = ReadDielectricIor(bsdf_node, {"int_ior", "intIOR"}, 1.5046);
        double ext_ior = ReadDielectricIor(bsdf_node, {"ext_ior", "extIOR"}, 1.000277);
        Texture *specular_reflectance = ReadTexture(bsdf_node, {"specularReflectance", "specular_reflectance"}, renderer, 1.0);
        Texture *specular_transmittance = ReadTexture(bsdf_node, {"specularTransmittance", "specular_transmittance"}, renderer, 1.0);
        bsdf = new ThinDielectric(id, int_ior, ext_ior, specular_reflectance, specular_transmittance);
        break;
    }
    case "roughdielectric"_hash:
    {
        twosided = true;
        Ndf *ndf = ReadNdf(bsdf_node, renderer, 0.1);
        double int_ior = ReadDielectricIor(bsdf_node, {"int_ior", "intIOR"}, 1.5046);
        double ext_ior = ReadDielectricIor(bsdf_node, {"ext_ior", "extIOR"}, 1.000277);
        Texture *specular_reflectance = ReadTexture(bsdf_node, {"specularReflectance", "specular_reflectance"}, renderer, 1.0);
        Texture *specular_transmittance = ReadTexture(bsdf_node, {"specularTransmittance", "specular_transmittance"}, renderer, 1.0);
        bsdf = new RoughDielectric(id, int_ior, ext_ior, ndf, specular_reflectance, specular_transmittance);
        break;
    }
    case "conductor"_hash:
    {
        dvec3 eta, k;
        ReadConductorIor(bsdf_node, &eta, &k);
        Texture *specular_reflectance = ReadTexture(bsdf_node, {"specularReflectance", "specular_reflectance"}, renderer, 1.0);
#ifdef ROUGH_SMOOTH
        Ndf *ndf = ReadNdf(bsdf_node, renderer, 0.001, true);
        bsdf = new RoughConductor(id, eta, k, ndf, specular_reflectance);
#else
        bsdf = new Conductor(id, eta, k, specular_reflectance);
#endif
        break;
    }
    case "roughconductor"_hash:
    {
        Ndf *ndf = ReadNdf(bsdf_node, renderer, 0.1);
        Texture *specular_reflectance = ReadTexture(bsdf_node, {"specularReflectance", "specular_reflectance"}, renderer, 1.0);
        dvec3 eta, k;
        ReadConductorIor(bsdf_node, &eta, &k);
        bsdf = new RoughConductor(id, eta, k, ndf, specular_reflectance);
        break;
    }
    case "clearcoatedconductor"_hash:
    {
        Ndf *ndf = ReadNdf(bsdf_node, renderer, 0.1);
        Texture *specular_reflectance = ReadTexture(bsdf_node, {"specularReflectance", "specular_reflectance"}, renderer, 1.0);
        dvec3 eta, k;
        ReadConductorIor(bsdf_node, &eta, &k);

        double coating_roughness = ReadDouble(bsdf_node, {"coatingRoughness", "coating_roughness"}, 0.01);
        Texture *alpha_u = ReadTexture(bsdf_node, {}, renderer, coating_roughness);
        Texture *alpha_v = ReadTexture(bsdf_node, {}, renderer, coating_roughness);
        Ndf *ndf_coat = new GgxNdf(alpha_u, alpha_v);

        double clear_coat = ReadDouble(bsdf_node, {"clearCoat", "clear_coat"}, 0.5);
        clear_coat = std::max<double>(std::min<double>(clear_coat, 1), 0);
        bsdf = new ClearCoatedConductor(id, eta, k, ndf, clear_coat, ndf_coat, specular_reflectance);
        break;
    }
    case "plastic"_hash:
    {
        double int_ior = ReadDielectricIor(bsdf_node, {"int_ior", "intIOR"}, 1.49);
        double ext_ior = ReadDielectricIor(bsdf_node, {"ext_ior", "extIOR"}, 1.000277);
        Texture *diffuse_reflectance = ReadTexture(bsdf_node, {"diffuseReflectance", "diffuse_reflectance"}, renderer, 0.5);
        Texture *specular_reflectance = ReadTexture(bsdf_node, {"specularReflectance", "specular_reflectance"}, renderer, 1.0);
        bool nonlinear = bsdf_node.child("boolean").attribute("value").as_bool(false);
#ifdef ROUGH_SMOOTH
        Ndf *ndf = ReadNdf(bsdf_node, renderer, 0.001, true);
        bsdf = new RoughPlastic(id, int_ior, ext_ior, ndf, diffuse_reflectance, specular_reflectance, nonlinear);
#else
        bsdf = new Plastic(id, int_ior, ext_ior, diffuse_reflectance, specular_reflectance, nonlinear);
#endif
        break;
    }
    case "roughplastic"_hash:
    {
        Ndf *ndf = ReadNdf(bsdf_node, renderer, 0.1);
        double int_ior = ReadDielectricIor(bsdf_node, {"int_ior", "intIOR"}, 1.49);
        double ext_ior = ReadDielectricIor(bsdf_node, {"ext_ior", "extIOR"}, 1.000277);
        Texture *diffuse_reflectance = ReadTexture(bsdf_node, {"diffuseReflectance", "diffuse_reflectance"}, renderer, 0.5);
        Texture *specular_reflectance = ReadTexture(bsdf_node, {"specularReflectance", "specular_reflectance"}, renderer, 1.0);
        bool nonlinear = bsdf_node.child("boolean").attribute("value").as_bool(false);
        bsdf = new RoughPlastic(id, int_ior, ext_ior, ndf, diffuse_reflectance, specular_reflectance, nonlinear);
        break;
    }
    default:
    {
        std::cerr << "[warning] unsupport bsdf \"" << type << "\", use default diffuse instead\n";
        Texture *reflectance = new ConstantTexture("anonymous_texture_" + std::to_string(anonymous_texture_num++), 0.5);
        renderer.AddTexture(reflectance);
        bsdf = new Diffuse(id, reflectance);
        break;
    }
    }
    bsdf->SetTwosided(twosided);
    bsdf->SetBumpMapping(nump_map);
    bsdf->SetOpacity(opacity);
    renderer.AddBsdf(bsdf);
    return bsdf;
}

Medium *ReadMedium(const pugi::xml_node &medium_node, Renderer &renderer)
{
    if (!medium_node)
    {
        return nullptr;
    }

    std::string type = medium_node.attribute("type").as_string();

    if (type == "heterogeneous")
    {
        std::cerr << "[warning] not support \"heterogeneous\" media, ignore it.\n";
        return nullptr;
    }

    std::string id = medium_node.attribute("id").as_string();
    if (id.empty())
    {
        id = "anonymous_medium_" + std::to_string(anonymous_medium_num++);
    }
    if (medium_mp.count(id))
    {
        return medium_mp[id];
    }

    double scale = ReadDouble(medium_node, {"scale"}, 1.0);

    dvec3 sigma_s, sigma_a;
    pugi::xml_node albedo_node;
    if (GetChildByName(medium_node, {"albedo"}, &albedo_node))
    {
        pugi::xml_node sigma_t_node;
        if (!GetChildByName(medium_node, {"sigma_t", "sigmaT"}, &sigma_t_node))
        {
            std::cerr << "[error] \"sigma_t\" and \"albedo\" must be provided at the same time.\n";
            exit(1);
        }
        dvec3 albedo = ReadVec3(albedo_node, dvec3(0.75));
        dvec3 sigma_t = ReadVec3(sigma_t_node);
        sigma_s = albedo * sigma_t;
        sigma_a = sigma_t - sigma_s;
    }
    pugi::xml_node sigma_a_node;
    if (GetChildByName(medium_node, {"sigmaA"}, &sigma_a_node))
    {
        pugi::xml_node sigma_s_node;
        if (!GetChildByName(medium_node, {"sigmaS"}, &sigma_s_node))
        {
            std::cerr << "[error] \"sigma_a\" and \"sigma_s\" must be provided at the same time.\n";
            exit(1);
        }
        sigma_a = ReadVec3(sigma_a_node);
        sigma_s = ReadVec3(sigma_s_node);
    }

    if (!albedo_node && !sigma_a_node)
    {
        std::string material = medium_node.child("string").attribute("value").as_string("skin1");
        Medium *medium = LookupHomogeneousMedium(id, scale, material);
        if (medium == nullptr)
        {
            std::cerr << "[error] unsupport medium \"" << material << "\"\n";
            exit(1);
        }
        return medium;
    }
    else
    {
        PhaseFunction *phase_function = ReadPhaseFunction(medium_node);
        Medium *medium = new HomogeneousMedium(id, sigma_a * scale, sigma_s * scale, phase_function);
        renderer.AddMedium(medium);
        medium_mp[id] = medium;
        return medium;
    }
}

PhaseFunction *ReadPhaseFunction(const pugi::xml_node &parent_node)
{
    if (!parent_node.child("phase"))
    {
        return new IsotropicPhaseFunction();
    }
    pugi::xml_node phase_fuction_node = parent_node.child("phase");

    std::string type = phase_fuction_node.attribute("type").value();
    switch (Hash(type.c_str()))
    {
    case "isotropic"_hash:
        return new IsotropicPhaseFunction();
        break;
    case "hg"_hash:
    {
        double g = ReadDouble(phase_fuction_node, {"g"}, 0);
        return new HenyeyGreensteinPhaseFunction(dvec3(g));
        break;
    }
    default:
        std::cerr << "[warning] unsupport phase function \"" << type << "\", use \"isotropic\" instead.\n";
        return new IsotropicPhaseFunction();
        break;
    }
}

Ndf *ReadNdf(const pugi::xml_node &parent_node, Renderer &renderer, double default_roughness, bool force_roughness)
{
    Texture *alpha_u = nullptr, *alpha_v = nullptr;
    if (force_roughness)
    {
        alpha_u = ReadTexture(parent_node, {}, renderer, default_roughness);
        alpha_v = ReadTexture(parent_node, {}, renderer, default_roughness);
    }
    else
    {
        alpha_u = ReadTexture(parent_node, {"alpha", "alpha_u", "alphaU"}, renderer, default_roughness);
        alpha_v = ReadTexture(parent_node, {"alpha", "alpha_v", "alphaV"}, renderer, default_roughness);
    }

    pugi::xml_node child_node;
    GetChildByName(parent_node, {"distribution"}, &child_node);
    std::string type = child_node.attribute("value").value();
    Ndf *ndf = nullptr;
    switch (Hash(type.c_str()))
    {
    case "beckmann"_hash:
        ndf = new BeckmannNdf(alpha_u, alpha_v);
        break;
    case "ggx"_hash:
    default:
        ndf = new GgxNdf(alpha_u, alpha_v);
        break;
    }
    return ndf;
}

double ReadDielectricIor(const pugi::xml_node &parent_node, const std::vector<std::string> &valid_names, double defalut_value)
{
    pugi::xml_node ior_node;
    GetChildByName(parent_node, valid_names, &ior_node);
    if (ior_node.name() == std::string("string"))
    {
        std::string material_name = ior_node.attribute("value").as_string();
        double ior = 0.0;
        if (!LookupDielectricIor(material_name, &ior))
        {
            std::cerr << "[error] unsupported  material :" << material_name << ", "
                      << "use default dielectric ior \"" << defalut_value << "\" instead." << std::endl;
            return defalut_value;
        }
        else
        {
            return ior;
        }
    }
    else
    {
        return ior_node.attribute("value").as_double(defalut_value);
    }
}

void ReadConductorIor(const pugi::xml_node &parent_node, dvec3 *eta, dvec3 *k)
{
    pugi::xml_node child_node;
    if (GetChildByName(parent_node, {"material"}, &child_node))
    {
        std::string material_name = child_node.attribute("value").as_string();
        if (!LookupConductorIor(material_name, eta, k))
        {
            std::cerr << "[error] unsupported bsdf :" << material_name << ", "
                      << "use default Conductor material \"Cu\" instead." << std::endl;
            LookupConductorIor("Cu", eta, k);
        }
    }
    else if (GetChildByName(parent_node, {"eta"}, &child_node))
    {
        *eta = ReadVec3(child_node);
        if (!GetChildByName(parent_node, {"k"}, &child_node))
        {
            std::cerr << "[error] cannot find \"k\" for Conductor bsdf \"" << parent_node.attribute("id").as_string() << "\"" << std::endl;
            exit(1);
        }
        *k = ReadVec3(child_node);
    }
    else
    {
        LookupConductorIor("Cu", eta, k);
    }
}

Shape *ReadShape(pugi::xml_node shape_node, Renderer &renderer)
{
    std::string id = shape_node.attribute("id").value();
    if (id.empty())
    {
        id = "anonymous_shape_" + std::to_string(anonymous_shape_num++);
    }
    std::string type = shape_node.attribute("type").value();
    dmat4 to_world = ReadTransform4(shape_node.child("transform"));

    bool flip_texcoords = false,
         flip_normals = false,
         face_normals = false;
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

    Shape *shape = nullptr;
    switch (Hash(type.c_str()))
    {
    case "rectangle"_hash:
        shape = new Rectangle(id, to_world, flip_normals);
        break;
    case "cube"_hash:
        shape = new Cube(id, to_world, flip_normals);
        break;
    case "disk"_hash:
        shape = new Disk(id, to_world, flip_normals);
        break;
    case "cylinder"_hash:
    {
        dvec3 p0 = ReadVec3(shape_node, {"p0"}, dvec3(0));
        dvec3 p1 = ReadVec3(shape_node, {"p1"}, dvec3{0, 0, 1});
        double radius = shape_node.child("float").attribute("value").as_double(1.0);
        shape = new Cylinder(id, p0, p1, radius, to_world, flip_normals);
        break;
    }
    case "sphere"_hash:
    {
        dvec3 center = ReadVec3(shape_node, {"center"}, dvec3(0));
        double radius = shape_node.child("float").attribute("value").as_double(1.0);
        shape = new Sphere(id, center, radius, to_world, flip_normals);
        break;
    }
    case "serialized"_hash:
    {
        int shape_index = shape_node.child("integer").attribute("value").as_int(0);
        std::string filename = shape_node.child("string").attribute("value").as_string();
        std::vector<Shape *> meshes = LoadObject(config_file_directory + filename, shape_index, face_normals, flip_normals,
                                                 flip_texcoords, id, to_world);
        shape = new Meshes(id, meshes, flip_normals);
        break;
    }
    case "obj"_hash:
        flip_texcoords = ReadBoolean(shape_node, {"flip_tex_coords", "flipTexCoords"}, true);
    case "gltf"_hash:
    case "ply"_hash:
    {
        std::string filename = shape_node.child("string").attribute("value").as_string();
        std::vector<Shape *> meshes = LoadObject(config_file_directory + filename, face_normals, flip_normals,
                                                 flip_texcoords, id, to_world);
        shape = new Meshes(id, meshes, flip_normals);
        break;
    }
    default:
        std::cerr << "[warning] unsupported shape type \"" << type << "\", ignore it.\n";
        return nullptr;
        break;
    }

    bool is_emitter = false;
    Bsdf *bsdf = nullptr;
    if (shape_node.child("emitter"))
    {
        pugi::xml_node emitter_node = shape_node.child("emitter");
        assert(emitter_node.attribute("type").value() == std::string("area") ||
               emitter_node.attribute("type").value() == std::string("constant"));
        if (!emitter_node.child("rgb"))
        {
            std::cerr << "[error] cannot find radiance for area light \"" << id << "\"";
            exit(1);
        }

        dvec3 radiance = ReadVec3(emitter_node, {"radiance"}, dvec3(1));
        if (shape_node.child("bsdf"))
        {
            bsdf = ReadBsdf(shape_node.child("bsdf"), renderer);
        }
        else
        {
            bsdf = new Diffuse(id, nullptr);
            renderer.AddBsdf(bsdf);
        }
        bsdf->SetRadiance(radiance);
        is_emitter = true;
    }
    else if (shape_node.child("bsdf"))
    {
        bsdf = ReadBsdf(shape_node.child("bsdf"), renderer);
    }
    else
    {
        bool found = false;
        for (auto ref_node : shape_node.children("ref"))
        {
            std::string bsdf_id = shape_node.child("ref").attribute("id").value();
            if (bsdf_mp.count(bsdf_id))
            {
                bsdf = bsdf_mp.at(bsdf_id);
                found = true;
                break;
            }
        }
        if (!found && !GetChildByName(shape_node, {"interior", "exterior"}))
        {
            std::cerr << "[error] cannot find bsdf for shape \"" << id << "\"";
            exit(1);
        }
    }
    if (is_emitter)
    {
        renderer.AddEmitter(new AreaLight(bsdf, shape));
    }
    shape->SetBsdf(bsdf);

    Medium *medium_int = nullptr;
    pugi::xml_node medium_int_node;
    if (GetChildByName(shape_node, {"interior"}, &medium_int_node))
    {
        medium_int = ReadMedium(medium_int_node, renderer);
    }
    Medium *medium_ext = nullptr;
    pugi::xml_node medium_ext_node;
    if (GetChildByName(shape_node, {"exterior"}, &medium_ext_node))
    {
        medium_ext = ReadMedium(medium_ext_node, renderer);
    }
    shape->SetMedium(medium_int, medium_ext);

    return shape;
}

dmat4 ReadTransform4(const pugi::xml_node &transform_node)
{
    auto result = dmat4(1);
    if (!transform_node)
    {
        return result;
    }

    for (pugi::xml_node node : transform_node.children())
    {
        const char *name = node.name();
        switch (Hash(name))
        {
        case "translate"_hash:
        {
            dvec3 translate = ReadVec3(node, dvec3(0));
            result = glm::translate(translate) * result;
            break;
        }
        case "rotate"_hash:
        {
            dvec3 axis = ReadVec3(node, dvec3(0));
            double angle = node.attribute("angle").as_double(0);
            result = glm::rotate(glm::radians(angle), axis) * result;
            break;
        }
        case "scale"_hash:
        {
            dvec3 scale = ReadVec3(node, dvec3(1));
            result = glm::scale(scale) * result;
            break;
        }
        case "matrix"_hash:
        {
            dmat4 matrix = ReadMatrix4(node);
            result = matrix * result;
            break;
        }
        case "lookat"_hash:
        {
            dvec3 origin = ReadVec3(node, dvec3{0, 0, 0}, "origin"),
                  target = ReadVec3(node, dvec3{1, 0, 0}, "target"),
                  up = ReadVec3(node, dvec3{0, 1, 0}, "up");
            result = glm::inverse(glm::lookAtLH(origin, target, up)) * result;
            break;
        }
        default:
            std::cerr << "[warning] unsupport transform type \"" << name << "\", ignore it.\n";
            break;
        }
    }
    return result;
}

dmat4 ReadMatrix4(const pugi::xml_node &matrix_node)
{
    if (!matrix_node.attribute("value"))
    {
        return dmat4(1);
    }

    auto result = dmat4(1);
    std::string str_buffer = matrix_node.attribute("value").as_string();
    const int space_count = static_cast<int>(std::count(str_buffer.begin(), str_buffer.end(), ' '));
    if (space_count == 8)
    {
        sscanf(str_buffer.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
               &result[0][0], &result[1][0], &result[2][0],
               &result[0][1], &result[1][1], &result[2][1],
               &result[0][2], &result[1][2], &result[2][2]);
    }
    else
    {
        sscanf(str_buffer.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
               &result[0][0], &result[1][0], &result[2][0], &result[3][0],
               &result[0][1], &result[1][1], &result[2][1], &result[3][1],
               &result[0][2], &result[1][2], &result[2][2], &result[3][2],
               &result[0][3], &result[1][3], &result[2][3], &result[3][3]);
    }

    return result;
}

dvec3 ReadVec3(const pugi::xml_node &vec3_node, dvec3 defalut_value, const char *value_name)
{
    if (vec3_node.attribute("value") || value_name != nullptr)
    {
        std::string str_buffer = vec3_node.attribute(value_name != nullptr ? value_name : "value").as_string();

        const int space_count = static_cast<int>(std::count(str_buffer.begin(), str_buffer.end(), ' '));
        if (space_count == 0)
        {
            defalut_value.x = vec3_node.attribute("value").as_double(defalut_value.x);
            return dvec3(defalut_value.x);
        }
        else if (space_count == 2)
        {
            auto result = dvec3(0);
            const int Comma_count = static_cast<int>(std::count(str_buffer.begin(), str_buffer.end(), ','));
            if (Comma_count == 0)
            {
                sscanf(str_buffer.c_str(), "%lf %lf %lf", &result[0], &result[1], &result[2]);
                return result;
            }
            else
            {
                sscanf(str_buffer.c_str(), "%lf, %lf, %lf", &result[0], &result[1], &result[2]);
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
        defalut_value.x = vec3_node.attribute("x").as_double(defalut_value.x);
        defalut_value.y = vec3_node.attribute("y").as_double(defalut_value.y);
        defalut_value.z = vec3_node.attribute("z").as_double(defalut_value.z);
        return defalut_value;
    }
}

dvec3 ReadVec3(const pugi::xml_node &parent_node, const std::vector<std::string> &valid_names, dvec3 defalut_value)
{
    pugi::xml_node vec3_node;
    GetChildByName(parent_node, valid_names, &vec3_node);
    if (!vec3_node)
    {
        return defalut_value;
    }
    return ReadVec3(vec3_node, defalut_value);
}

double ReadDouble(const pugi::xml_node &parent_node, const std::vector<std::string> &valid_names, double defalut_value)
{
    pugi::xml_node double_node;
    GetChildByName(parent_node, valid_names, &double_node);
    return double_node.attribute("value").as_double(defalut_value);
}

bool ReadBoolean(const pugi::xml_node &parent_node, const std::vector<std::string> &valid_names, bool defalut_value)
{
    pugi::xml_node boolean_node;
    GetChildByName(parent_node, valid_names, &boolean_node);
    return boolean_node.attribute("value").as_bool(defalut_value);
}

int ReadInt(const pugi::xml_node &parent_node, const std::vector<std::string> &valid_names, int defalut_value)
{
    pugi::xml_node int_node;
    GetChildByName(parent_node, valid_names, &int_node);
    return int_node.attribute("value").as_int(defalut_value);
}

bool GetChildByName(pugi::xml_node parent_node, const std::vector<std::string> &valid_names, pugi::xml_node *child_node)
{
    for (const std::string &name : valid_names)
    {
        for (pugi::xml_node node : parent_node.children())
        {
            if (node.attribute("name").value() == name)
            {
                if (child_node != nullptr)
                {
                    *child_node = node;
                }
                return true;
            }
        }
    }
    return false;
}

NAMESPACE_END(raytracer)