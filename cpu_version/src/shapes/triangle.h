#pragma once

#include "../core/shape_base.h"

NAMESPACE_BEGIN(raytracer)

//三角形面片类
class Triangle : public Shape
{
public:
	/**
	 * \brief 三角面片
	 * \param vertices 顶点
	 * \param normals 顶点法向量
	 * \param texcoords 顶点纹理坐标
	 * \param tangents 切线
	 * \param bitangents 副切线
	 * \param bsdf 材质
	 * \param flip_normals 法线方向是否翻转
	 */
	Triangle(const std::vector<Vector3> &vertices, const std::vector<Vector3> &normals, const std::vector<Vector2> &texcoords,
			 const std::vector<Vector3> &tangents, const std::vector<Vector3> &bitangents, Bsdf *bsdf, Medium *medium,
			 bool flip_normals);

	/**
	 * \brief 三角面片
	 * \param vertices 顶点
	 * \param normals 顶点法向量
	 * \param texcoords 顶点纹理坐标
	 * \param bsdf 材质
	 * \param flip_normals 法线方向是否翻转
	 */
	Triangle(const std::vector<Vector3> &vertices, const std::vector<Vector3> &normals, const std::vector<Vector2> &texcoords,
			 Bsdf *bsdf, Medium *medium, bool flip_normals);

	void Intersect(const Ray &ray, Intersection &its) const override;

	Intersection SampleP() const override;

private:
	void Setup(const std::vector<Vector3> &vertices, const std::vector<Vector3> &normals);

	Vector3 v0v1_;					  //三角形的一条边
	Vector3 v0v2_;					  //三角形的一条边
	std::vector<Vector3> vertices_;	  //面片包含的点
	std::vector<Vector2> texcoords_;  //纹理坐标 (width,height)
	std::vector<Vector3> normals_;	  //面片包含点对应的法向量
	std::vector<Vector3> tangents_;	  //切线
	std::vector<Vector3> bitangents_; //副切线
};

NAMESPACE_END(raytracer)