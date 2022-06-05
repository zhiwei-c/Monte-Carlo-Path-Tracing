#include "cube.h"

NAMESPACE_BEGIN(raytracer)

constexpr float CubePositions[][3] = {{1, -1, -1}, {1, -1, 1}, {-1, -1, 1}, {-1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, 1, 1}, {1, 1, 1}, {1, -1, -1}, {1, 1, -1}, {1, 1, 1}, {1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}, {-1, -1, 1}, {-1, -1, 1}, {-1, 1, 1}, {-1, 1, -1}, {-1, -1, -1}, {1, 1, -1}, {1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}};

constexpr float CubeNormals[][3] = {{0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}};

constexpr float CubeTexcoords[][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}};

constexpr uint32_t CubeTriangles[][3] = {{0, 1, 2}, {3, 0, 2}, {4, 5, 6}, {7, 4, 6}, {8, 9, 10}, {11, 8, 10}, {12, 13, 14}, {15, 12, 14}, {16, 17, 18}, {19, 16, 18}, {20, 21, 22}, {23, 20, 22}};

Cube::Cube(Bsdf *bsdf, Medium *medium, std::unique_ptr<Mat4> to_world, bool flip_normals)
    : Shape(ShapeType::kCube, bsdf, medium, flip_normals)
{
    auto to_world_p = Mat4(1),
         to_world_n = Mat4(1);
    if (to_world != nullptr)
    {
        to_world_p = Mat4(*to_world);
        to_world_n = Mat4(glm::inverse(glm::transpose(*to_world)));
    }

    auto vector = Vector3(0);
    auto vec = Vector2(0);
    auto indices = std::vector<unsigned int>();
    auto vertices = std::vector<Vector3>();
    auto normals = std::vector<Vector3>();
    auto texcoords = std::vector<Vector2>();
    for (auto i = 0; i < 12; ++i)
    {
        indices = {CubeTriangles[i][0],
                   CubeTriangles[i][1],
                   CubeTriangles[i][2]};

        vertices.clear();
        normals.clear();
        texcoords.clear();

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

        meshes_.push_back(new Triangle(vertices, normals, texcoords, bsdf, medium, flip_normals));
    }
    bvh_ = std::make_unique<BvhAccel>(meshes_);
    aabb_ = bvh_->aabb();
    area_ = bvh_->area();
    pdf_area_ = 1.0 / area_;
    for (auto &mesh : meshes_)
        mesh->SetPdfArea(this->pdf_area_);
}

NAMESPACE_END(raytracer)