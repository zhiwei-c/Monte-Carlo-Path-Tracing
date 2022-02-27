#pragma once

#include "../material/bsdfs/materials.h"
#include "../utils/file_path.h"
#include "model_loader/obj_parser.h"
#include "shapes/shapes.h"
#include "envmap.h"

NAMESPACE_BEGIN(simple_renderer)

//场景类
class Scene
{
public:
	/**
	 * \brief 空的场景。场景类包含了物体、材质、环境光映射
	 */
	Scene() : envmap_(nullptr) {}

	/**
	 * \brief 场景类，包含了物体、材质、环境光映射
	 * \param shapes 场景中包含的物体
	 * \param materials 场景中包含物体的材质
	 * \param materials 场景中包含的环境光映射
	 */
	Scene(std::vector<Shape *> shapes, std::vector<Material *> materials, Envmap *envmap)
		: shapes_(shapes), materials_(materials), envmap_(envmap) {}

	~Scene()
	{
		for (auto &shape : shapes_)
		{
			if (shape)
			{
				delete shape;
				shape = nullptr;
			}
		}
		for (auto &material : materials_)
		{
			if (material)
			{
				delete material;
				material = nullptr;
			}
		}

		if (envmap_)
		{
			delete envmap_;
			envmap_ = nullptr;
		}
	}

	/**
	 * \brief 使用TinyObjLoader加载物体模型
	 * \param obj_path 待加载的obj文件的路径
	 */
	void AddSceneObj(const std::string &obj_path)
	{
		ObjParser::Parse(obj_path, shapes_, materials_);
	}

	///\return 场景中包含的物体
	std::vector<Shape *> shapes() const { return shapes_; }

	///\brief 设置场景的环境光映射
	void setEnvmap(Envmap *envmap) { envmap_ = envmap; }

	///\return 场景中包含的环境光映射
	Envmap *envmap() const { return envmap_; }

private:
	std::vector<Shape *> shapes_;		//场景包含的物体
	std::vector<Material *> materials_; //场景包含的材质
	Envmap *envmap_;					//环境光映射
};

NAMESPACE_END(simple_renderer)