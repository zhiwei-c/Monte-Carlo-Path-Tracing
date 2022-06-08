#include "config_parser.h"

#include "../../bsdfs/ior.h"

Float GetIor(rapidxml::xml_node<> *node_parent, std::string ior_type, std::string default_bsdf_name);

MicrofacetDistribType GetDistrbType(const std::string &name);

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

    auto bsdf_info = BsdfInfo();
    switch (Hash(bsdf_type.c_str()))
    {
    case "diffuse"_hash:
        bsdf_info = ParseDiffuse(node_bsdf);
        break;
    case "dielectric"_hash:
        twosided = true;
        bsdf_info = ParseDielectric(node_bsdf, false);
        break;
    case "roughdielectric"_hash:
        twosided = true;
        bsdf_info = ParseRoughDielectric(node_bsdf);
        break;
    case "thindielectric"_hash:
        twosided = true;
        bsdf_info = ParseDielectric(node_bsdf, true);
        break;
    case "conductor"_hash:
        bsdf_info = ParseConductor(node_bsdf);
        break;
    case "roughconductor"_hash:
        bsdf_info = ParseRoughConductor(node_bsdf);
        break;
    case "plastic"_hash:
        bsdf_info = ParsePlastic(node_bsdf);
        break;
    case "roughplastic"_hash:
        bsdf_info = ParseRoughPlastic(node_bsdf);
        break;
    default:
        std::cerr << "[warning] " << GetTreeName(node_bsdf) << std::endl
                  << "\tconduct as diffuse" << std::endl;
        bsdf_info = ParseDiffuse(node_bsdf);
    };

    bsdf_info.twosided = twosided;
    bsdf_info.bump_map_idx = bump_map_idx;
    bsdf_info.opacity_idx = opacity_idx;
    renderer->AddBsdfInfo(bsdf_info);
    m_id_to_m_idx[id] = bsdf_cnt++;
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
                      << "\tcannot find existed bsdf with id: " << ref_id << std::endl;
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

BsdfInfo ParseDiffuse(rapidxml::xml_node<> *node_diffuse)
{
    auto reflectance_idx = static_cast<uint>(-1);
    auto reflectance_info = ParseTextureOrOther(node_diffuse, "reflectance");
    if (reflectance_info)
    {
        renderer->AddTextureInfo(reflectance_info);
        reflectance_idx = texture_cnt++;
    }

    auto diffuse_info = BsdfInfo();
    diffuse_info.type = kDiffuse;
    diffuse_info.diffuse_reflectance_idx = reflectance_idx;
    return diffuse_info;
}

BsdfInfo ParseDielectric(rapidxml::xml_node<> *node_dielectric, bool thin)
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

    auto bsdf_info = BsdfInfo();
    bsdf_info.type = thin ? kThinDielectric : kDielectric;
    bsdf_info.eta = vec3(int_ior / ext_ior);
    bsdf_info.specular_reflectance_idx = specular_reflectance_idx;
    bsdf_info.specular_transmittance_idx = specular_transmittance_idx;
    return bsdf_info;
}

BsdfInfo ParseRoughDielectric(rapidxml::xml_node<> *node_rough_dielectric)
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
    auto bsdf_info = BsdfInfo();
    bsdf_info.type = kRoughDielectric;
    bsdf_info.eta = vec3(int_ior / ext_ior);
    bsdf_info.specular_reflectance_idx = specular_reflectance_idx;
    bsdf_info.specular_transmittance_idx = specular_transmittance_idx;
    bsdf_info.distri = GetDistrbType(distri);
    bsdf_info.alpha_u_idx = alpha_u_idx;
    bsdf_info.alpha_v_idx = alpha_v_idx;
    return bsdf_info;
}

BsdfInfo ParseConductor(rapidxml::xml_node<> *node_conductor)
{
    auto eta = vec3(0);
    auto k = vec3(1);
    auto ext_eta = GetIor(node_conductor, "extEta", "air");

    auto specular_reflectance_idx = static_cast<uint>(-1);
    auto specular_reflectance = ParseTextureOrOther(node_conductor, "specularReflectance");
    if (specular_reflectance)
    {
        renderer->AddTextureInfo(specular_reflectance);
        specular_reflectance_idx = texture_cnt++;
    }

    bool mirror = false;
    auto node_bsdf_name = GetChild(node_conductor, "material");
    if (node_bsdf_name)
    {
        auto bsdf_name = GetAttri(node_bsdf_name, "value").value();
        if (bsdf_name == "none")
            mirror = true;
        else if (!LookupConductorIor(bsdf_name, eta, k))
        {
            std::cerr << "[error] " << GetTreeName(node_bsdf_name) << std::endl
                      << " unsupported bsdf :" << bsdf_name << ", "
                      << "use default Conductor bsdf instead." << std::endl;
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
    auto bsdf_info = BsdfInfo();
    bsdf_info.type = kConductor;
    bsdf_info.mirror = mirror;
    bsdf_info.eta = eta / ext_eta;
    bsdf_info.k = k / ext_eta;
    bsdf_info.specular_reflectance_idx = specular_reflectance_idx;
    return bsdf_info;
}

BsdfInfo ParseRoughConductor(rapidxml::xml_node<> *node_rough_conductor)
{

    auto eta = vec3(0);
    auto k = vec3(1);
    auto ext_eta = GetIor(node_rough_conductor, "extEta", "air");
    auto node_material_name = GetChild(node_rough_conductor, "material");

    auto specular_reflectance_idx = static_cast<uint>(-1);
    auto specular_reflectance = ParseTextureOrOther(node_rough_conductor, "specularReflectance");
    if (specular_reflectance)
    {
        renderer->AddTextureInfo(specular_reflectance);
        specular_reflectance_idx = texture_cnt++;
    }

    bool mirror = false;
    if (node_material_name)
    {
        auto bsdf_name = GetAttri(node_material_name, "value").value();
        if (bsdf_name == "none")
            mirror = true;
        else if (!LookupConductorIor(bsdf_name, eta, k))
        {
            std::cerr << "[error] " << GetTreeName(node_material_name) << std::endl
                      << " unsupported bsdf :" << bsdf_name << ", "
                      << "use default Conductor bsdf instead." << std::endl;
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

    auto bsdf_info = BsdfInfo();
    bsdf_info.type = kRoughConductor;
    bsdf_info.mirror = mirror;
    bsdf_info.eta = eta / ext_eta;
    bsdf_info.k = k / ext_eta;
    bsdf_info.specular_reflectance_idx = specular_reflectance_idx;
    bsdf_info.distri = GetDistrbType(distri);
    bsdf_info.alpha_u_idx = alpha_u_idx;
    bsdf_info.alpha_v_idx = alpha_v_idx;
    return bsdf_info;
}

BsdfInfo ParsePlastic(rapidxml::xml_node<> *node_plastic)
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

    auto bsdf_info = BsdfInfo();
    bsdf_info.type = kPlastic;
    bsdf_info.eta = vec3(int_ior / ext_ior);
    bsdf_info.diffuse_reflectance_idx = diffuse_reflectance_idx;
    bsdf_info.specular_reflectance_idx = specular_reflectance_idx;
    bsdf_info.nonlinear = nonlinear;
    return bsdf_info;
}

BsdfInfo ParseRoughPlastic(rapidxml::xml_node<> *node_rough_plastic)
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

    auto bsdf_info = BsdfInfo();
    bsdf_info.type = kRoughPlastic;
    bsdf_info.eta = vec3(int_ior / ext_ior);
    bsdf_info.diffuse_reflectance_idx = diffuse_reflectance_idx;
    bsdf_info.specular_reflectance_idx = specular_reflectance_idx;
    bsdf_info.distri = GetDistrbType(distri);
    bsdf_info.alpha_u_idx = alpha_u_idx;
    bsdf_info.alpha_v_idx = alpha_v_idx;
    bsdf_info.nonlinear = nonlinear;
    return bsdf_info;
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

Float GetIor(rapidxml::xml_node<> *node_parent, std::string ior_type, std::string default_bsdf_name)
{
    Float ior = 0;
    auto node_ior = GetChild(node_parent, ior_type);
    if (!node_ior)
        LookupDielectricIor(default_bsdf_name, ior);
    else if (strcmp(node_ior->name(), "float") == 0)
        return std::stof(GetAttri(node_ior, "value").value());
    else
    {
        auto int_ior_name = GetAttri(node_ior, "value").value();
        if (!LookupDielectricIor(int_ior_name, ior))
        {
            std::cerr << "[error] " << GetTreeName(node_ior) << std::endl
                      << "\tunsupported ior bsdf " << int_ior_name << std::endl;
            exit(1);
        }
    }
    return ior;
}
