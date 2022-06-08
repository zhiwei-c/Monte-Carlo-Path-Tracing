#include "config_parser.h"

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
