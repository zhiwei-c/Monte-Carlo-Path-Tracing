#pragma once

#include <tuple>
#include <utility>
#include <optional>

#include "nlohmann/json.hpp"

#include "../../renderer.h"

NAMESPACE_BEGIN(raytracer)

Renderer *ParseJsonCfg(const std::string &file_path);

///\brief 从 json 格式数据内全局光照模型
static Integrator *InitIntegrator(nlohmann::json &data);

///\brief 从json格式数据内解析相机信息
static Camera *InitCamera(nlohmann::json &data);

///\brief 从json格式数据内解析生成图像信息
static Film InitFilm(nlohmann::json &data);

///\brief 从 json 格式数据内解析天空盒信息
static Envmap *InitEnvmap(const std::string &dir_path, nlohmann::json &data);

///\brief 从json格式数据内解析出一个类型为 Vector3 的变量
static std::optional<Vector3> GetVec3(const nlohmann::json &data, const std::string &name, bool not_exist_ok = true);

///\brief 从json格式数据内解析出一个类型为 string 的变量
static std::optional<std::string> GetString(const nlohmann::json &data, const std::string &name, bool not_exist_ok = true);

///\brief 从 json 格式数据内解析出一个类型为 int 的变量
static std::optional<int> GetInt(const nlohmann::json &data, const std::string &name, bool not_exist_ok = true);

///\brief 从json格式数据内解析出一个类型为 Float 的变量
static std::optional<Float> GetFloat(const nlohmann::json &data, const std::string &name, bool not_exist_ok = true);

NAMESPACE_END(raytracer)