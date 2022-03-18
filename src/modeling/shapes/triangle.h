#pragma once

#include "../shape.h"

NAMESPACE_BEGIN(simple_renderer)

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
	 * \param material 材质
	 * \param flip_normals 法线方向是否翻转
	 */
	Triangle(const std::vector<Vector3> &vertices,
			 const std::vector<Vector3> &normals,
			 const std::vector<Vector2> &texcoords,
			 const std::vector<Vector3> &tangents,
			 const std::vector<Vector3> &bitangents,
			 Material *material,
			 bool flip_normals);

	/**
	 * \brief 三角面片
	 * \param vertices 顶点
	 * \param normals 顶点法向量
	 * \param texcoords 顶点纹理坐标
	 * \param material 材质
	 * \param flip_normals 法线方向是否翻转
	 */
	Triangle(const std::vector<Vector3> &vertices,
			 const std::vector<Vector3> &normals,
			 const std::vector<Vector2> &texcoords,
			 Material *material,
			 bool flip_normals);

	Intersection Intersect(const Ray &ray) const override;

	Intersection SampleP() const override;

private:
	Vector3 v0v1_;					  //三角形的一条边
	Vector3 v0v2_;					  //三角形的一条边
	std::vector<Vector3> vertices_;	  //面片包含的点
	std::vector<Vector2> texcoords_;  //纹理坐标 (width,height)
	std::vector<Vector3> normals_;	  //面片包含点对应的法向量
	std::vector<Vector3> tangents_;	  //切线
	std::vector<Vector3> bitangents_; //副切线

	void Setup(const std::vector<Vector3> &vertices, const std::vector<Vector3> &normals);
};

NAMESPACE_END(simple_renderer)