#pragma once

#include <memory>

#include "../shape.h"
#include "../../utils/accelerator/bvh_accel.h"
#include "triangle.h"

NAMESPACE_BEGIN(simple_renderer)

static constexpr float CubePositions[][3] = {{1, -1, -1}, {1, -1, 1}, {-1, -1, 1}, {-1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, 1, 1}, {1, 1, 1}, {1, -1, -1}, {1, 1, -1}, {1, 1, 1}, {1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}, {-1, -1, 1}, {-1, -1, 1}, {-1, 1, 1}, {-1, 1, -1}, {-1, -1, -1}, {1, 1, -1}, {1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}};

static constexpr float CubeNormals[][3] = {{0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}};

static constexpr float CubeTexcoords[][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}};

static constexpr uint32_t CubeTriangles[][3] = {{0, 1, 2}, {3, 0, 2}, {4, 5, 6}, {7, 4, 6}, {8, 9, 10}, {11, 8, 10}, {12, 13, 14}, {15, 12, 14}, {16, 17, 18}, {19, 16, 18}, {20, 21, 22}, {23, 20, 22}};

class Cube : public Shape
{
public:
    /**
     * \brief 标准立方体。在局部坐标下表示为：-1 < x < 1，-1 < y < 1，-1 < z < 1
     * \param material 材质
     * \param to_world 从局部坐标系到世界坐标系的变换矩阵
     * \param flip_normals 法线方向是否翻转
     */
    Cube(Material *material,
         std::unique_ptr<Mat4> to_world,
         bool flip_normals)
        : Shape(ShapeType::kCube, material, flip_normals)
    {
        Mat4 to_world_p, to_world_n;
        if (to_world != nullptr)
        {
            to_world_p = Mat4(*to_world);
            to_world_n = Mat4(glm::inverse(glm::transpose(*to_world)));
        }

        Vector3 vector;
        Vector2 vec;
        for (auto i = 0; i < 12; ++i)
        {
            std::vector<unsigned int> indices = {
                CubeTriangles[i][0],
                CubeTriangles[i][1],
                CubeTriangles[i][2],
            };

            std::vector<Vector3> vertices;
            std::vector<Vector3> normals;
            std::vector<Vector2> texcoords;

            for (int v = 0; v < 3; v++)
            {
                // positions
                vector[0] = CubePositions[indices[v]][0];
                vector[1] = CubePositions[indices[v]][1];
                vector[2] = CubePositions[indices[v]][2];
                if (to_world)
                    vector = TransfromPt(to_world_p, vector);
                vertices.push_back(vector);

                // normals
                vector[0] = CubeNormals[indices[v]][0];
                vector[1] = CubeNormals[indices[v]][1];
                vector[2] = CubeNormals[indices[v]][2];
                if (to_world)
                    vector = TransfromDir(to_world_n, vector);
                normals.push_back(glm::normalize(vector));

                // texture coordinates
                vec[0] = CubeTexcoords[indices[v]][0];
                vec[1] = CubeTexcoords[indices[v]][1];
                texcoords.push_back(vec);
            }

            meshes_.push_back(new Triangle(vertices, normals, texcoords, material, flip_normals));
        }
        bvh_ = std::make_unique<BvhAccel>(meshes_);
        aabb_ = bvh_->aabb();
        area_ = bvh_->area();
        pdf_area_ = 1 / area_;
        for (auto &mesh : meshes_)
        {
            mesh->setPdfArea(this->pdf_area_);
        }
    }

    ~Cube()
    {
        for (auto &mesh : meshes_)
        {
            if (mesh)
            {
                delete mesh;
                mesh = nullptr;
            }
        }
    }

    // Intersection Intersect(const Ray &ray) const override
    // {
    //     return this->bvh_->Intersect(ray);
    // }
    
    void Intersect(const Ray &ray, Intersection& its) const override
    {
        this->bvh_->Intersect(ray, its);
    }

    Intersection SampleP() const override
    {
        return this->bvh_->Sample();
    }

private:
    std::unique_ptr<BvhAccel> bvh_; //层次包围盒
    std::vector<Shape *> meshes_;   //包含的三角面片
};

NAMESPACE_END(simple_renderer)