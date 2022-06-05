#include "../model_parser.h"

#include <omp.h>
#include <iostream>
#include <optional>
#include <tuple>

#include "../model_loader/obj_loader.h"
#include "../../utils/file_path.h"
#include "../../core/ior.h"

NAMESPACE_BEGIN(raytracer)

static std::vector<Bsdf *> ParseBsdf(const std::vector<tinyobj::material_t> &bsdfs, const std::string &mtl_path);

static std::optional<Vector3> GetVec3(const std::map<std::string, std::string> &params, const std::string &name);

static std::optional<Float> GetFloat(const std::map<std::string, std::string> &params, const std::string &name);

static std::optional<Float> GetIor(const std::map<std::string, std::string> &params, const std::string &name);

static std::optional<std::string> GetString(const std::map<std::string, std::string> &params, const std::string &name);

static std::optional<bool> GetBoolean(const std::map<std::string, std::string> &params, const std::string &name);

static std::optional<MicrofacetDistribType> GetDistrbType(const std::map<std::string, std::string> &params, const std::string &name);

static std::tuple<Vector3, Vector3> GetEtaK(const std::map<std::string, std::string> &params, const std::string &id);

void ModelParser::Parse(const std::string &obj_path, std::vector<Shape *> &shapes, std::vector<Bsdf *> &bsdfs)
{
    std::cout << "[info] begin read model info...\t\t\t\r";

    auto mtl_path = GetDirectory(obj_path);

    ObjLoader reader;
    tinyobj::ObjReaderConfig reader_config;
    if (!reader.ParseFromFile(obj_path, reader_config))
    {
        if (!reader.Error().empty())
        {
            std::cerr << "[error] TinyObjReader: " << reader.Error();
        }
        exit(1);
    }
    if (!reader.Warning().empty())
    {
        std::cout << "[warning] TinyObjReader: " << reader.Warning();
    }
    auto &attrib = reader.GetAttrib();
    auto &shapes_raw = reader.GetShapes();
    auto &bsdfs_raw = reader.GetMaterials();

    std::cout << "[info] read model info finished.    \r";
    std::cout << "[info] begin parse bsdf...      \r";

    auto bsdfs_new = ParseBsdf(bsdfs_raw, mtl_path);
    bsdfs.insert(bsdfs.end(), bsdfs_new.begin(), bsdfs_new.end());

    std::cout << "[info] parse bsdf finished.     \r";
    std::cout << "[info] begin parse geometry info... \r";

    for (int s = 0; s < shapes_raw.size(); s++)
    {
        auto m_now_id = shapes_raw[s].mesh.material_ids[0];
        if (m_now_id < 0)
            m_now_id = 0;
        bool textre_mapping_ = bsdfs_new[m_now_id]->TextureMapping() ||
                               bsdfs_new[m_now_id]->NormalPerturbing();

        std::vector<Shape *> meshes(shapes_raw[s].mesh.num_face_vertices.size());

#pragma omp parallel for
        for (int f = 0; f < shapes_raw[s].mesh.num_face_vertices.size(); f++)
        {
            auto index_offset = static_cast<size_t>(f * 3);
            std::vector<Vector3> vertices;
            std::vector<Vector3> normals;
            std::vector<Vector2> texcoords;
            for (int v = 0; v < 3; v++)
            {
                tinyobj::index_t idx = shapes_raw[s].mesh.indices[index_offset + v];

                Float vx = attrib.vertices[3 * idx.vertex_index + 0];
                Float vy = attrib.vertices[3 * idx.vertex_index + 1];
                Float vz = attrib.vertices[3 * idx.vertex_index + 2];
                vertices.emplace_back(Vector3(vx, vy, vz));

                if (!attrib.normals.empty())
                {
                    Float nx = attrib.normals[3 * idx.normal_index + 0];
                    Float ny = attrib.normals[3 * idx.normal_index + 1];
                    Float nz = attrib.normals[3 * idx.normal_index + 2];
                    normals.emplace_back(glm::normalize(Vector3(nx, ny, nz)));
                }

                if (textre_mapping_)
                {
                    Float tx = attrib.texcoords[2 * idx.texcoord_index + 0];
                    Float ty = attrib.texcoords[2 * idx.texcoord_index + 1];
                    texcoords.emplace_back(Vector2(tx, 1 - ty));
                }
            }
            meshes[f] = new Triangle(vertices, normals, texcoords, bsdfs_new[m_now_id], nullptr, nullptr, false);
        }
        shapes.push_back(new Meshes(meshes, bsdfs_new[m_now_id], nullptr, nullptr, false));
    }
    std::cout << "[info] load model succeed\t\t\t\r";
}

std::vector<Bsdf *> ParseBsdf(const std::vector<tinyobj::material_t> &bsdfs, const std::string &mtl_path)
{
    std::vector<Bsdf *> result;
    std::cout << "[info] begin parse bsdf...\t\t\t\r";
    for (size_t m = 0; m < bsdfs.size(); m++)
    {
        auto other_params = bsdfs[m].unknown_parameter;
        auto id = bsdfs[m].name;
        auto ke = Spectrum(bsdfs[m].emission[0], bsdfs[m].emission[1], bsdfs[m].emission[2]);
        if (auto Le = GetVec3(other_params, "Le"); Le.has_value())
            ke = Le.value();
        if (ke.r + ke.g + ke.b > 0)
        {
            result.push_back(new AreaLight(ke));
            continue;
        }

        auto kd = Spectrum(bsdfs[m].diffuse[0], bsdfs[m].diffuse[1], bsdfs[m].diffuse[2]);

        std::unique_ptr<Texture> diffuse_map = nullptr;
        if (auto diffuse_texname = bsdfs[m].diffuse_texname; !diffuse_texname.empty())
            diffuse_map.reset(new Bitmap(mtl_path + diffuse_texname, 2.2));

        std::unique_ptr<Texture> specular_map = nullptr;
        if (auto specular_texname = bsdfs[m].specular_texname; !specular_texname.empty())
            specular_map.reset(new Bitmap(mtl_path + specular_texname, 2.2));

        std::unique_ptr<Texture> bump_map = nullptr;
        if (auto bump_texname = bsdfs[m].bump_texname; !bump_texname.empty())
            bump_map.reset(new Bitmap(mtl_path + bump_texname, 1));

        std::unique_ptr<Texture> opacity_map = nullptr;
        if (auto alpha_texname = bsdfs[m].alpha_texname; !alpha_texname.empty())
            opacity_map.reset(new Bitmap(mtl_path + alpha_texname, 1));
        else
        {
            auto opacity = bsdfs[m].dissolve;
            if (opacity < 1)
                opacity_map.reset(new ConstantTexture(opacity));
        }

        auto ks = Spectrum(bsdfs[m].specular[0], bsdfs[m].specular[1], bsdfs[m].specular[2]);
        auto ns = bsdfs[m].shininess;

        auto flag_parsed = false;
        if (auto it = other_params.find("type"); it != other_params.end())
        {
            switch (Hash(it->second.c_str()))
            {
            case "dielectric"_hash:
            case "smoothdielectric"_hash:
            case "smoothDielectric"_hash:
            case "smooth_dielectric"_hash:
            case "smooth-dielectric"_hash:
            {
                auto ext_ior = GetIor(other_params, "extIOR").value_or(1.000277);
                auto int_ior = GetIor(other_params, "intIOR").value_or(1.5046);
                result.push_back(new Dielectric(int_ior, ext_ior, nullptr, nullptr));
                result.back()->SetTwosided(true);
                flag_parsed = true;
                break;
            }
            case "thindielectric"_hash:
            case "thinDielectric"_hash:
            case "thin_dielectric"_hash:
            case "thin-dielectric"_hash:
            {
                auto ext_ior = GetIor(other_params, "extIOR").value_or(1.000277);
                auto int_ior = GetIor(other_params, "intIOR").value_or(1.5046);
                result.push_back(new ThinDielectric(int_ior,
                                                    ext_ior,
                                                    nullptr,
                                                    nullptr));
                result.back()->SetTwosided(true);
                flag_parsed = true;
                break;
            }
            case "roughdielectric"_hash:
            case "roughDielectric"_hash:
            case "rough_dielectric"_hash:
            case "rough-dielectric"_hash:
            {
                auto distrib_type = GetDistrbType(other_params, "distribution").value_or(MicrofacetDistribType::kBeckmann);
                auto alpha = GetFloat(other_params, "alpha").value_or(0.1f);
                auto alpha_u = GetFloat(other_params, "alphaU").value_or(alpha);
                auto alpha_u_tex = std::make_unique<ConstantTexture>(Spectrum(alpha_u));
                auto alpha_v = GetFloat(other_params, "alphaV").value_or(alpha);
                auto alpha_v_tex = std::make_unique<ConstantTexture>(Spectrum(alpha_v));
                auto ext_ior = GetIor(other_params, "extIOR").value_or(1.000277);
                auto int_ior = GetIor(other_params, "intIOR").value_or(1.5046);
                result.push_back(new RoughDielectric(int_ior,
                                                     ext_ior,
                                                     nullptr,
                                                     nullptr,
                                                     distrib_type,
                                                     std::move(alpha_u_tex),
                                                     std::move(alpha_v_tex)));
                result.back()->SetTwosided(true);
                flag_parsed = true;
                break;
            }
            case "conductor"_hash:
            case "smoothconductor"_hash:
            case "smoothConductor"_hash:
            case "smooth_conductor"_hash:
            case "smooth-conductor"_hash:
            {
                auto [eta, k] = GetEtaK(other_params, id);
                result.push_back(new Conductor(eta,
                                               k,
                                               1.000277,
                                               nullptr));
                flag_parsed = true;
                break;
            }
            case "roughconductor"_hash:
            case "roughConductor"_hash:
            case "rough_conductor"_hash:
            case "rough-conductor"_hash:
            {
                auto distrib_type = GetDistrbType(other_params, "distribution").value_or(MicrofacetDistribType::kBeckmann);
                auto alpha = GetFloat(other_params, "alpha").value_or(0.1f);
                auto alpha_u = GetFloat(other_params, "alphaU").value_or(alpha);
                auto alpha_u_tex = std::make_unique<ConstantTexture>(Spectrum(alpha_u));
                auto alpha_v = GetFloat(other_params, "alphaV").value_or(alpha);
                auto alpha_v_tex = std::make_unique<ConstantTexture>(Spectrum(alpha_v));
                auto [eta, k] = GetEtaK(other_params, id);
                result.push_back(new RoughConductor(eta,
                                                    k,
                                                    1.000277,
                                                    nullptr,
                                                    distrib_type,
                                                    std::move(alpha_u_tex),
                                                    std::move(alpha_v_tex)));
                flag_parsed = true;
                break;
            }
            case "plastic"_hash:
            case "smoothPlastic"_hash:
            case "smoothplastic"_hash:
            case "smooth_plastic"_hash:
            case "smooth-plastic"_hash:
            {
                if (!diffuse_map)
                {
                    auto diffuse_reflectance = GetVec3(other_params, "diffuseReflectance").value_or(Vector3(0.5f));
                    diffuse_map.reset(new ConstantTexture(Spectrum(diffuse_reflectance)));
                }
                auto ext_ior = GetIor(other_params, "extIOR").value_or(1.000277);
                auto int_ior = GetIor(other_params, "intIOR").value_or(1.49);
                auto nolinear = GetBoolean(other_params, "nonlinear").value_or(false);
                result.push_back(new Plastic(int_ior,
                                             ext_ior,
                                             std::move(diffuse_map),
                                             nullptr,
                                             nolinear));
                flag_parsed = true;
                break;
            }
            case "roughplastic"_hash:
            case "roughPlastic"_hash:
            case "rough_plastic"_hash:
            case "rough-plastic"_hash:
            {
                auto distrib_type = GetDistrbType(other_params, "distribution").value_or(MicrofacetDistribType::kBeckmann);

                auto alpha = GetFloat(other_params, "alpha").value_or(0.1f);
                auto alpha_tex = std::make_unique<ConstantTexture>(Spectrum(alpha));

                auto ext_ior = GetIor(other_params, "extIOR").value_or(1.000277);
                auto int_ior = GetIor(other_params, "intIOR").value_or(1.49);

                if (!diffuse_map)
                {
                    auto diffuse_reflectance = GetVec3(other_params, "diffuseReflectance").value_or(Vector3(0.5f));
                    diffuse_map.reset(new ConstantTexture(Spectrum(diffuse_reflectance)));
                }

                auto nolinear = GetBoolean(other_params, "nonlinear").value_or(false);

                result.push_back(new RoughPlastic(int_ior,
                                                  ext_ior,
                                                  std::move(diffuse_map),
                                                  nullptr,
                                                  distrib_type,
                                                  std::move(alpha_tex),
                                                  nolinear));
                flag_parsed = true;
                break;
            }
            }
        }
        if (!flag_parsed)
        {
            if (!diffuse_map)
                diffuse_map.reset(new ConstantTexture(Spectrum(kd)));

            if (ks.r + ks.g + ks.b > 0)
            {
                if (!specular_map)
                    specular_map.reset(new ConstantTexture(Spectrum(ks)));
                result.push_back(new Glossy(std::move(diffuse_map), std::move(specular_map), ns));
            }
            else
                result.push_back(new Diffuse(std::move(diffuse_map)));
        }
        result.back()->SetBumpMapping(std::move(bump_map));
        result.back()->SetOpacity(std::move(opacity_map));
    }
    std::cout << "[info] parse bsdf finished\t\t\t\r";
    return result;
}

std::optional<Vector3> GetVec3(const std::map<std::string, std::string> &params, const std::string &name)
{
    auto it = params.find(name);
    if (it == params.end())
        return std::nullopt;
    Vector3 ret(0);
    sscanf(it->second.c_str(), "%lf %lf %lf", &ret[0], &ret[1], &ret[2]);
    return ret;
}

std::optional<Float> GetFloat(const std::map<std::string, std::string> &params, const std::string &name)
{
    auto it = params.find(name);
    if (it == params.end())
        return std::nullopt;
    else
        return std::stof(it->second);
}

std::optional<Float> GetIor(const std::map<std::string, std::string> &params, const std::string &name)
{
    auto it = params.find(name);
    if (it == params.end())
        return std::nullopt;
    else
    {
        Float ior = 0;
        if (LookupDielectricIor(it->second, ior))
            return ior;
        else
            return std::stof(it->second);
    }
}

std::tuple<Spectrum, Spectrum> GetEtaK(const std::map<std::string, std::string> &params, const std::string &id)
{
    auto material_name = GetString(params, "material").value_or("");
    if (material_name.empty())
    {
        auto eta = GetVec3(params, "eta");
        auto k = GetVec3(params, "k");
        if (!eta.has_value() || !k.has_value())
        {
            if (eta.has_value())
            {
                std::cerr << "[error] cannot find paramter \"k\" for material \"" << id << "\"" << std::endl;
                exit(1);
            }
            else if (k.has_value())
            {
                std::cerr << "[error] cannot find paramter \"eta\" for material \"" << id << "\"" << std::endl;
                exit(1);
            }
            else
                return {Spectrum(0), Spectrum(1)};
        }
        else
            return {eta.value(), k.value()};
    }
    else
    {
        auto eta = Spectrum(0), k = Spectrum(1);
        if (LookupConductorIor(material_name, eta, k))
        {
            return {eta, k};
        }
        else
        {
            std::cerr << "[error] unsupported material name \"" << material_name << "\" for material \"" << id << "\"" << std::endl;
            exit(1);
        }
    }
}

std::optional<std::string> GetString(const std::map<std::string, std::string> &params, const std::string &name)
{
    auto it = params.find(name);
    if (it == params.end())
        return std::nullopt;
    else
        return it->second;
}

std::optional<bool> GetBoolean(const std::map<std::string, std::string> &params, const std::string &name)
{
    auto it = params.find(name);
    if (it == params.end())
        return std::nullopt;
    else if (it->second == "true")
        return true;
    else
        return false;
}

std::optional<MicrofacetDistribType> GetDistrbType(const std::map<std::string, std::string> &params, const std::string &name)
{
    auto it = params.find(name);
    if (it == params.end())
        return std::nullopt;
    switch (Hash(it->second.c_str()))
    {
    case "beckmann"_hash:
        return MicrofacetDistribType::kBeckmann;
        break;
    case "ggx"_hash:
        return MicrofacetDistribType::kGgx;
        break;
    default:
        std::cout << "[warning] unkown microfacet distribution: " << it->second << ", use Beckmann instead.";
        return MicrofacetDistribType::kBeckmann;
        break;
    }
}

NAMESPACE_END(raytracer)