#pragma once

#include "core/emitter.h"
#include "core/bsdf.h"
#include "core/shape.h"
#include "core/camera.h"

#include "utils/file_path.h"
#include "utils/model_parser.h"

NAMESPACE_BEGIN(raytracer)

///\brief 渲染器类
class Renderer
{
public:
	///\brief 渲染器
	Renderer() : camera_(nullptr), integrator_(nullptr), envmap_(nullptr) {}

	~Renderer()
	{
		if (camera_)
		{
			delete camera_;
			camera_ = nullptr;
		}

		if (integrator_)
		{
			delete integrator_;
			integrator_ = nullptr;
		}

		if (envmap_)
		{
			delete envmap_;
			envmap_ = nullptr;
		}

		for (auto &shape : shapes_)
		{
			if (shape)
			{
				delete shape;
				shape = nullptr;
			}
		}

		for (auto &bsdf : bsdfs_)
		{
			if (bsdf)
			{
				delete bsdf;
				bsdf = nullptr;
			}
		}
	}

	///\brief 添加物体
	void AddShape(Shape *shape)
	{
		shapes_.push_back(shape);
	}

	///\brief 添加材质
	void AddBsdf(Bsdf *bsdf)
	{
		bsdfs_.push_back(bsdf);
	}

	///\brief 设置物体和材质
	void AddShapesBsdfs(const std::string &obj_path)
	{
		ModelParser::Parse(obj_path, shapes_, bsdfs_);
	}

	///\brief 设置天空盒
	void SetEnvmap(Envmap *envmap) { envmap_ = envmap; }

	///\brief 照相机
	void SetCamera(Camera *camera) { camera_ = camera; }

	///\brief 设置全局光照模型
	void SetIntegrator(Integrator *integrator) { integrator_ = integrator; }

	///\brief 生成图像
	void Render(std::string output_filename)
	{
		if (output_filename.empty())
		{
			std::cout << "[warning] enmpty output file name, use default: \"result." << camera_->Format() << "\"" << std::endl;
			output_filename = ChangeSuffix("result.png", camera_->Format());
		}

		integrator_->InitIntegrator(shapes_, envmap_);
		auto frame = camera_->Shoot(integrator_);
		frame->Save(output_filename);
	}

private:
	Camera *camera_;					//照相机
	Integrator *integrator_;			//全局光照模型
	Envmap *envmap_;					//环境光映射
	std::vector<Shape *> shapes_;		//场景包含的物体
	std::vector<Bsdf *> bsdfs_; //场景包含的材质
};

NAMESPACE_END(raytracer)