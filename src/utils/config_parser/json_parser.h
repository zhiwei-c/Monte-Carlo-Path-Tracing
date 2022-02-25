#pragma once

#include <tuple>
#include <utility>
#include <optional>

#include "nlohmann/json.hpp"
#include "../global.h"
#include "../../rendering/camera.h"
#include "../../modeling/envmap.h"
#include "../../modeling/scene.h"
#include "../../rendering/integrator.h"

NAMESPACE_BEGIN(simple_renderer)

std::tuple<Scene *, Camera *, Integrator *> ParseJsonCfg(const std::string &file_path);

//\brief 从 json 格式数据内解析待绘制的模型信息
//
//\param data 待解析的 json 数据
//
//\return 获取到的待绘制模型信息
static Scene *InitScene(const std::string &dir_path, nlohmann::json &data);

//\brief 从 json 格式数据内解析绘制方程积分求解方法的信息
//
//\param data 待解析的 json 数据
//
//\return 获取到的绘制方程积分求解方法
static Integrator* InitIntegrator(nlohmann::json &data);

//\brief 从json格式数据内解析相机信息
//
//\param data 待解析的 json 数据
//
//\return 获取到的相机信息
static Camera *InitCamera(nlohmann::json &data);

//\brief 从json格式数据内解析相机信息
//
//\param data 待解析的 json 数据
//
//\return 获取到的相片信息
static Film InitFilm(nlohmann::json &data);

//\brief 从 json 格式数据内解析天空盒信息
//
//\param data 待解析的 json 数据
//
//\return 获取到的天空盒信息
static Envmap *InitEnvmap(const std::string &dir_path, nlohmann::json &data);

//\brief 从json格式数据内解析出一个类型为 Vector3 的变量
//
//\param data 待解析的 json 数据
//
//\param name 目标变量名
//
//\return 获取到的目标变量值
static std::optional<Vector3> GetVec3(const nlohmann::json &data, const std::string &name, bool not_exist_ok = true);

//\brief 从json格式数据内解析出一个类型为 string 的变量
//
//\param data 待解析的 json 数据
//\param name 目标变量名
//\param not_exist_ok 相应名称的变量可以不一定存在
//
//\return 获取到的目标变量值
static std::optional<std::string> GetString(const nlohmann::json &data, const std::string &name, bool not_exist_ok = true);

//\brief 从 json 格式数据内解析出一个类型为 int 的变量
//
//\param data 待解析的 json 数据
//\param name 目标变量名
//\param not_exist_ok 相应名称的变量可以不一定存在
//
//\return 获取到的目标变量值
static std::optional<int> GetInt(const nlohmann::json &data, const std::string &name, bool not_exist_ok = true);

//\brief 从json格式数据内解析出一个类型为 Float 的变量
//
//\param data 待解析的 json 数据
//\param name 目标变量名
//\param not_exist_ok 相应名称的变量可以不一定存在
//
//\return 获取到的目标变量值
static std::optional<Float> GetFloat(const nlohmann::json &data, const std::string &name, bool not_exist_ok = true);

NAMESPACE_END(simple_renderer)