#include "xml_parser.h"

#include "../../core/ior.h"

NAMESPACE_BEGIN(raytracer)

void XmlParser::ParseMedium(rapidxml::xml_node<> *node_medium, Renderer *renderer, const std::string &id_default)
{
    auto medium_type = GetAttri(node_medium, "type").value();

    auto id = id_default.empty() ? GetAttri(node_medium, "id").value() : id_default;

    if (medium_type != "homogeneous")
    {
        std::cerr << "[error] " << GetTreeName(node_medium) << std::endl
                  << "\tunsupport medium type \"" << medium_type << " \", use \"homogeneous\" instead" << std::endl;
    }

    Medium *medium = nullptr;

    auto node_material = GetChild(node_medium, "material");
    if (node_material)
    {
        auto material_name = GetAttri(node_material, "value").value();
        medium = LookupHomogeneousMedium(material_name);
        if (!medium)
        {
            std::cerr << "[error] " << GetTreeName(node_material) << std::endl
                      << " unsupported material :\"" << material_name << "\", "
                      << "use default Conductor material instead." << std::endl;
            medium = LookupHomogeneousMedium("Skimmilk");
        }
    }
    else if (node_medium->first_node())
    {
        Spectrum sigma_s = GetSpectrum(GetChild(node_medium, "sigmaS", false));
        Spectrum sigma_a = GetSpectrum(GetChild(node_medium, "sigmaA", false));

        std::unique_ptr<PhaseFunction> phase_function = ParsePhaseFunction(node_medium);
        medium = new HomogeneousMedium(sigma_a, sigma_s, std::move(phase_function));
    }
    else
        medium = LookupHomogeneousMedium("Skimmilk");

    id_to_medium_[id] = medium;
    renderer->AddMedium(medium);
}

std::unique_ptr<PhaseFunction> XmlParser::ParsePhaseFunction(rapidxml::xml_node<> *&node_medium)
{
    auto node_phase = node_medium->first_node("phase");
    if (!node_phase)
        return std::make_unique<IsotropicPhase>();

    auto phase_type = GetAttri(node_phase, "type").value();
    switch (Hash(phase_type.c_str()))
    {
    case "isotropic"_hash:
        return std::make_unique<IsotropicPhase>();
        break;
    case "hg"_hash:
    {
        auto g = Spectrum(0);
        auto node_g = GetChild(node_medium, "g");
        if (node_g)
        {
            if (strcmp(node_g->name(), "float"))

                g = Spectrum(std::stof(GetAttri(node_g, "value").value()));

            else
                g = GetSpectrum(node_g);
        }

        for (int i = 0; i < 3; i++)
        {
            if (g[i] < -1 || g[i] > 1)
            {
                std::cerr << "[error] " << GetTreeName(node_phase) << std::endl
                          << "\terror g for hg phase function" << std::endl;
                exit(1);
            }
        }
        return std::make_unique<HenyeyGreensteinPhase>(g);
        break;
    }
    default:
        std::cerr << "[error] " << GetTreeName(node_phase) << std::endl
                  << "\tunsupport phase function \"" << phase_type << " \", use \"isotropic\" instead" << std::endl;
        return std::make_unique<IsotropicPhase>();
        break;
    }
}

void XmlParser::ParseBsdf(rapidxml::xml_node<> *node_bsdf, Renderer *renderer, const std::string &id_default)
{
    auto bsdf_type = GetAttri(node_bsdf, "type").value();
    auto bump_map = ParseBumpMapping(node_bsdf, bsdf_type);
    auto id = id_default.empty() ? GetAttri(node_bsdf, "id").value() : id_default;
    auto opacity_map = ParseOpacity(node_bsdf, bsdf_type);
    auto twsided = false;
    if (bsdf_type == "twosided")
    {
        twsided = true;
        node_bsdf = node_bsdf->first_node("bsdf");
        if (!node_bsdf)
        {
            std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
                      << "\tnot enough bsdf information" << std::endl;
            exit(1);
        }
        bsdf_type = GetAttri(node_bsdf, "type").value();
    }
    if (ParseCoating(id, node_bsdf, bsdf_type))
        return;
    Bsdf *bsdf = nullptr;
    switch (Hash(bsdf_type.c_str()))
    {
    case "diffuse"_hash:
        bsdf = ParseDiffuse(node_bsdf);
        break;
    case "roughdiffuse"_hash:
        bsdf = ParseRoughDiffuse(node_bsdf);
        break;
    case "dielectric"_hash:
        twsided = true;
        bsdf = ParseDielectric(node_bsdf);
        break;
    case "roughdielectric"_hash:
        twsided = true;
        bsdf = ParseRoughDielectric(node_bsdf);
        break;
    case "thindielectric"_hash:
        twsided = true;
        bsdf = ParseDielectric(node_bsdf, true);
        break;
    case "conductor"_hash:
        bsdf = ParseConductor(node_bsdf);
        break;
    case "roughconductor"_hash:
        bsdf = ParseRoughConductor(node_bsdf);
        break;
    case "plastic"_hash:
        bsdf = ParsePlastic(node_bsdf);
        break;
    case "roughplastic"_hash:
        bsdf = ParseRoughPlastic(node_bsdf);
        break;
    default:
        std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
                  << "\tcannot handle bsdf type " << bsdf_type << std::endl;
        exit(1);
    }

    bsdf->SetTwosided(twsided);
    bsdf->SetBumpMapping(std::move(bump_map));
    bsdf->SetOpacity(std::move(opacity_map));
    id_to_bsdf_[id] = bsdf;
    renderer->AddBsdf(bsdf);
}

std::unique_ptr<Texture> XmlParser::ParseBumpMapping(rapidxml::xml_node<> *&node_bsdf, std::string &bsdf_type)
{
    if (bsdf_type != "bumpmap")
        return nullptr;

    auto node_bump = node_bsdf->first_node("texture");
    auto bump_map = ParseTexture(node_bump);

    node_bsdf = node_bsdf->first_node("bsdf");
    if (!node_bsdf)
    {
        std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
                  << "\tnot enough bsdf information" << std::endl;
        exit(1);
    }
    bsdf_type = GetAttri(node_bsdf, "type").value();

    return bump_map;
}

std::unique_ptr<Texture> XmlParser::ParseOpacity(rapidxml::xml_node<> *&node_bsdf, std::string &bsdf_type)
{
    if (bsdf_type != "mask")
        return nullptr;

    auto node_opacity = GetChild(node_bsdf, "opacity", false);
    auto opacity_map = ParseTextureOrOther(node_opacity, "opacity");
    if (!opacity_map)
    {
        std::cerr << "[error] " << GetTreeName(node_opacity) << std::endl
                  << "\tnot enough opacity information" << std::endl;
        exit(1);
    }

    if (opacity_map->Constant() &&
        (!FloatEqual(opacity_map->Color(Vector2(0)).r, opacity_map->Color(Vector2(0)).g) ||
         !FloatEqual(opacity_map->Color(Vector2(0)).r, opacity_map->Color(Vector2(0)).b) ||
         !FloatEqual(opacity_map->Color(Vector2(0)).g, opacity_map->Color(Vector2(0)).b)))
    {
        std::cerr << "[error] " << GetTreeName(node_opacity) << std::endl
                  << "\tnot support different opacity for different color channel" << std::endl;
        exit(1);
    }

    node_bsdf = node_bsdf->first_node("bsdf");
    if (!node_bsdf)
    {
        std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
                  << "\tnot enough bsdf information" << std::endl;
        exit(1);
    }
    bsdf_type = GetAttri(node_bsdf, "type").value();

    return opacity_map;
}

bool XmlParser::ParseCoating(const std::string &id, rapidxml::xml_node<> *&node_bsdf, std::string &bsdf_type)
{
    if (bsdf_type != "coating" &&
        bsdf_type != "roughcoating")
        return false;
    std::cout << "[warning] not support coating bsdf, ignore it." << std::endl;
    if (auto node_ref = node_bsdf->first_node("ref"); node_ref)
    {
        if (node_ref->next_sibling("ref"))
        {
            std::cerr << "[error] " << GetTreeName(node_ref) << std::endl
                      << "\tfind multiple ref" << std::endl;
            exit(1);
        }
        auto ref_id = GetAttri(node_ref, "id").value();
        if (!id_to_bsdf_.count(ref_id))
        {
            std::cerr << "[error] " << GetTreeName(node_ref) << std::endl
                      << "\tcannot find existed bsdf with id: " << ref_id << std::endl;
            exit(1);
        }
        else
        {
            id_to_bsdf_[id] = id_to_bsdf_[ref_id];
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

Bsdf *XmlParser::ParseDiffuse(rapidxml::xml_node<> *node_diffuse)
{
    auto reflectance = ParseTextureOrOther(node_diffuse, "reflectance");
    if (!reflectance)
        reflectance.reset(new ConstantTexture(Spectrum(0.5)));

    return new Diffuse(std::move(reflectance));
}

Bsdf *XmlParser::ParseRoughDiffuse(rapidxml::xml_node<> *node_rough_diffuse)
{
    auto reflectance = ParseTextureOrOther(node_rough_diffuse, "reflectance");
    if (!reflectance)
        reflectance.reset(new ConstantTexture(Spectrum(0.5)));
    auto alpha = ParseTextureOrOther(node_rough_diffuse, "alpha");
    if (!alpha)
        alpha.reset(new ConstantTexture(Spectrum(0.2)));

    auto use_fast_approx = GetBoolean(node_rough_diffuse, "useFastApprox").value_or(false);
    return new RoughDiffuse(std::move(reflectance), std::move(alpha), use_fast_approx);
}

Bsdf *XmlParser::ParseDielectric(rapidxml::xml_node<> *node_dielectric, bool thin)
{
    auto int_ior = GetIor(node_dielectric, "intIOR", "bk7");
    auto ext_ior = GetIor(node_dielectric, "extIOR", "air");
    auto specular_reflectance = ParseTextureOrOther(node_dielectric, "specularReflectance");
    auto specular_transmittance = ParseTextureOrOther(node_dielectric, "specularTransmittance");
    if (thin)
        return new ThinDielectric(int_ior,
                                  ext_ior,
                                  std::move(specular_reflectance),
                                  std::move(specular_transmittance));
    else
        return new Dielectric(int_ior,
                              ext_ior,
                              std::move(specular_reflectance),
                              std::move(specular_transmittance));
}

Bsdf *XmlParser::ParseRoughDielectric(rapidxml::xml_node<> *node_rough_dielectric)
{
    auto int_ior = GetIor(node_rough_dielectric, "intIOR", "bk7");
    auto ext_ior = GetIor(node_rough_dielectric, "extIOR", "air");

    auto specular_reflectance = ParseTextureOrOther(node_rough_dielectric, "specularReflectance");
    auto specular_transmittance = ParseTextureOrOther(node_rough_dielectric, "specularTransmittance");

    auto distri = GetString(node_rough_dielectric, "distribution").value_or("beckmann");

    auto alpha_u = ParseTextureOrOther(node_rough_dielectric, "alpha");
    if (!alpha_u)
        alpha_u = ParseTextureOrOther(node_rough_dielectric, "alphaU");
    if (!alpha_u)
        alpha_u.reset(new ConstantTexture(Spectrum(0.1)));
    auto alpha_v = ParseTextureOrOther(node_rough_dielectric, "alphaV");

    return new RoughDielectric(int_ior,
                               ext_ior,
                               std::move(specular_reflectance),
                               std::move(specular_transmittance),
                               GetDistrbType(distri),
                               std::move(alpha_u),
                               std::move(alpha_v));
}

Bsdf *XmlParser::ParseConductor(rapidxml::xml_node<> *node_conductor)
{
    auto eta = Spectrum(0);
    auto k = Spectrum(1);
    auto ext_eta = GetIor(node_conductor, "extEta", "air");
    auto node_bsdf = GetChild(node_conductor, "bsdf");
    if (node_bsdf)
    {
        auto bsdf_name = GetAttri(node_bsdf, "value").value();
        if (!LookupConductorIor(bsdf_name, eta, k))
        {
            std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
                      << " unsupported bsdf :" << bsdf_name << ", "
                      << "use default Conductor bsdf instead." << std::endl;
            exit(1);
        }
    }
    else if (node_conductor->first_node())
    {
        auto node_eta = GetChild(node_conductor, "eta", false);
        eta = GetSpectrum(node_eta);
        auto node_k = GetChild(node_conductor, "k", false);
        k = GetSpectrum(node_k);
    }

    auto specular_reflectance = ParseTextureOrOther(node_conductor, "specularReflectance");

    return new Conductor(eta,
                         k,
                         ext_eta,
                         std::move(specular_reflectance));
}

Bsdf *XmlParser::ParseRoughConductor(rapidxml::xml_node<> *node_rough_conductor)
{
    auto eta = Spectrum(0);
    auto k = Spectrum(1);
    auto ext_eta = GetIor(node_rough_conductor, "extEta", "air");
    auto node_bsdf = GetChild(node_rough_conductor, "bsdf");
    if (node_bsdf)
    {
        auto bsdf_name = GetAttri(node_bsdf, "value").value();
        if (!LookupConductorIor(bsdf_name, eta, k))
        {
            std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
                      << "unsupported bsdf :" << bsdf_name << ", "
                      << "use default Conductor bsdf instead." << std::endl;
            exit(1);
        }
    }
    else if (node_rough_conductor->first_node())
    {
        auto node_eta = GetChild(node_rough_conductor, "eta", false);
        eta = GetSpectrum(node_eta);
        auto node_k = GetChild(node_rough_conductor, "k", false);
        k = GetSpectrum(node_k);
    }

    auto specular_reflectance = ParseTextureOrOther(node_rough_conductor, "specularReflectance");

    auto distri = GetString(node_rough_conductor, "distribution").value_or("beckmann");
    auto alpha_u = ParseTextureOrOther(node_rough_conductor, "alpha");
    if (!alpha_u)
        alpha_u = ParseTextureOrOther(node_rough_conductor, "alphaU");
    if (!alpha_u)
        alpha_u.reset(new ConstantTexture(Spectrum(0.1)));
    auto alpha_v = ParseTextureOrOther(node_rough_conductor, "alphaV");

    return new RoughConductor(eta,
                              k,
                              ext_eta,
                              std::move(specular_reflectance),
                              GetDistrbType(distri),
                              std::move(alpha_u),
                              std::move(alpha_v));
}

Bsdf *XmlParser::ParsePlastic(rapidxml::xml_node<> *node_plastic)
{
    auto int_ior = GetIor(node_plastic, "intIOR", "polypropylene");
    auto ext_ior = GetIor(node_plastic, "extIOR", "air");
    auto specular_reflectance = ParseTextureOrOther(node_plastic, "specularReflectance");
    auto diffuse_reflectance = ParseTextureOrOther(node_plastic, "diffuseReflectance");
    if (!diffuse_reflectance)
        diffuse_reflectance.reset(new ConstantTexture(Spectrum(0.5)));
    bool nonlinear = GetBoolean(node_plastic, "nonlinear").value_or(false);

    return new Plastic(int_ior,
                       ext_ior,
                       std::move(diffuse_reflectance),
                       std::move(specular_reflectance),
                       nonlinear);
}

Bsdf *XmlParser::ParseRoughPlastic(rapidxml::xml_node<> *node_rough_plastic)
{
    auto int_ior = GetIor(node_rough_plastic, "intIOR", "polypropylene");
    auto ext_ior = GetIor(node_rough_plastic, "extIOR", "air");

    auto specular_reflectance = ParseTextureOrOther(node_rough_plastic, "specularReflectance");
    auto diffuse_reflectance = ParseTextureOrOther(node_rough_plastic, "diffuseReflectance");
    if (!diffuse_reflectance)
        diffuse_reflectance.reset(new ConstantTexture(0.5));

    auto distri = GetString(node_rough_plastic, "distribution").value_or("beckmann");
    auto alpha = ParseTextureOrOther(node_rough_plastic, "alpha");
    if (!alpha)
        alpha.reset(new ConstantTexture(Spectrum(0.1)));

    auto nonlinear = GetBoolean(node_rough_plastic, "nonlinear").value_or(false);

    return new RoughPlastic(int_ior,
                            ext_ior,
                            std::move(diffuse_reflectance),
                            std::move(specular_reflectance),
                            GetDistrbType(distri),
                            std::move(alpha),
                            nonlinear);
}

Float XmlParser::GetIor(rapidxml::xml_node<> *node_parent, std::string ior_type, std::string default_bsdf_name)
{
    Float ior = 0;
    auto node_ior = GetChild(node_parent, ior_type);
    if (!node_ior)
        LookupDielectricIor(default_bsdf_name, ior);
    else if (strcmp(node_ior->name(), "float") == 0)
        ior = std::stof(GetAttri(node_ior, "value").value());
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

MicrofacetDistribType XmlParser::GetDistrbType(const std::string &name)
{
    switch (Hash(name.c_str()))
    {
    case "beckmann"_hash:
        return MicrofacetDistribType::kBeckmann;
        break;
    case "ggx"_hash:
        return MicrofacetDistribType::kGgx;
        break;
    default:
        std::cout << "[warning] unkown microfacet distribution: " << name << ", use Beckmann instead.";
        return MicrofacetDistribType::kBeckmann;
        break;
    }
}

NAMESPACE_END(raytracer)