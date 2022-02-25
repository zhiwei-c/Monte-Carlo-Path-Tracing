#pragma once

#include <memory>

#include "../shape.h"
#include "../../utils/accelerator/bvh_accel.h"
#include "triangle.h"

NAMESPACE_BEGIN(simple_renderer)

static constexpr float RectanglePositions[][3] = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};

static constexpr float RectangleNormals[][3] = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};

static constexpr float RectangleTexcoords[][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

static constexpr uint32_t RectangleTriangles[][3] = {{0, 1, 2}, {2, 3, 0}};

class Rectangle : public Shape
{
public:
    /**
     * \brief 标准正方形。在局部坐标下表示为：-1 < x < 1，-1 < y < 1，z = 0
     * \param material 材质
     * \param to_world 从局部坐标系到世界坐标系的变换矩阵
     * \param flip_normals 法线方向是否翻转
     */
    Rectangle(Material *material, std::unique_ptr<Mat4> to_world, bool flip_normals)
        : Shape(ShapeType::kRectangle, material, flip_normals)
    {
        Mat4 to_world_p, to_world_n;
        if (to_world != nullptr)
        {
            to_world_p = Mat4(*to_world);
            to_world_n = Mat4(glm::inverse(glm::transpose(*to_world)));
        }

        Vector3 vector;
        Vector2 vec;
        for (auto i = 0; i < 2; ++i)
        {
            std::vector<unsigned int> indices = {
                RectangleTriangles[i][0],
                RectangleTriangles[i][1],
                RectangleTriangles[i][2],
            };

            std::vector<Vector3> vertices;
            std::vector<Vector3> normals;
            std::vector<Vector2> texcoords;

            for (int v = 0; v < 3; v++)
            {
                // positions
                vector[0] = RectanglePositions[indices[v]][0];
                vector[1] = RectanglePositions[indices[v]][1];
                vector[2] = RectanglePositions[indices[v]][2];
                if (to_world)
                    vector = TransfromPt(to_world_p, vector);
                vertices.push_back(vector);

                // normals
                vector[0] = RectangleNormals[indices[v]][0];
                vector[1] = RectangleNormals[indices[v]][1];
                vector[2] = RectangleNormals[indices[v]][2];
                if (to_world)
                    vector = TransfromDir(to_world_n, vector);
                normals.push_back(vector);

                // texture coordinates
                vec[0] = RectangleTexcoords[indices[v]][0];
                vec[1] = RectangleTexcoords[indices[v]][1];
                texcoords.push_back(vec);
            }

            meshes_.push_back(new Triangle(vertices, normals, texcoords, material, flip_normals));
        }
        bvh_ = std::make_unique<BvhAccel>(meshes_);
        area_ = bvh_->area();
        aabb_ = bvh_->aabb();
        
        for (auto &mesh : meshes_)
        {
            mesh->SetParent(this);
        }
    }

    ~Rectangle()
    {
        for (auto &mesh : meshes_)
        {
            if (mesh)
            {
                delete ((Triangle *)mesh);
                mesh = nullptr;
            }
        }
        meshes_.clear();
    }

    Intersection Intersect(const Ray &ray) const override
    {
        return this->bvh_->Intersect(ray);
    }

    std::pair<Intersection, Float> SampleP() const override
    {
        return this->bvh_->Sample();
    }

private:
    std::unique_ptr<BvhAccel> bvh_; //层次包围盒
    std::vector<Shape *> meshes_;   //包含的三角面片
};

NAMESPACE_END(simple_renderer)