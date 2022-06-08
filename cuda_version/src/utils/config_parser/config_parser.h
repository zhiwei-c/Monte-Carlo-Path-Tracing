#include <map>
#include <vector>
#include <string>
#include <optional>
#include <iostream>

#include "rapidxml/rapidxml_utils.hpp"
#include "glm/gtx/matrix_query.hpp"

#include "../file_path.h"
#include "../../renderer.h"

extern Renderer *renderer;
extern uint bsdf_cnt;
extern uint texture_cnt;
extern std::string xml_directory;
extern std::map<std::string, uint> m_id_to_m_idx;

Renderer *ParseRenderConfig(const std::string &config_path);

void ParseBsdf(rapidxml::xml_node<> *node_bsdf, const std::string *id_default = nullptr);
void ParseShape(rapidxml::xml_node<> *node_shape);
void ParseIntegrator(rapidxml::xml_node<> *node_integrator);
CameraInfo ParseCamera(rapidxml::xml_node<> *node_sensor);
void ParseEnvmap(rapidxml::xml_node<> *node_envmap, const CameraInfo &camera_info);

bool ParseCoating(const std::string &id, rapidxml::xml_node<> *&node_bsdf, std::string &bsdf_type);
BsdfInfo ParseDiffuse(rapidxml::xml_node<> *node_diffuse);
BsdfInfo ParseDielectric(rapidxml::xml_node<> *node_dielectric, bool thin);
BsdfInfo ParseRoughDielectric(rapidxml::xml_node<> *node_rough_dielectric);
BsdfInfo ParseConductor(rapidxml::xml_node<> *node_conductor);
BsdfInfo ParseRoughConductor(rapidxml::xml_node<> *node_rough_conductor);
BsdfInfo ParsePlastic(rapidxml::xml_node<> *node_plastic);
BsdfInfo ParseRoughPlastic(rapidxml::xml_node<> *node_rough_plastic);

TextureInfo *ParseTextureOrOther(rapidxml::xml_node<> *node_parent, const std::string &name);
TextureInfo *ParseTexture(rapidxml::xml_node<> *node_texture);

vec3 GetVec3(rapidxml::xml_node<> *node_vec3);
gmat4 *GetToWorld(rapidxml::xml_node<> *node_parent);
std::optional<bool> GetBoolean(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);
std::optional<int> GetInt(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);
std::optional<Float> GetFloat(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);
std::optional<vec3> GetPoint(rapidxml::xml_node<> *node_parent, const std::string &name, bool not_exist_ok = true);
std::optional<std::string> GetString(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);

std::string GetTreeName(rapidxml::xml_node<> *node);
std::optional<std::string> GetAttri(rapidxml::xml_node<> *node, std::string key, bool not_exist_ok = false);
rapidxml::xml_node<> *GetChild(rapidxml::xml_node<> *node, std::string name, bool not_exist_ok = true);
