#include "cube.h"

NAMESPACE_BEGIN(simple_renderer)

constexpr float CubePositions[][3] = {{1, -1, -1}, {1, -1, 1}, {-1, -1, 1}, {-1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, 1, 1}, {1, 1, 1}, {1, -1, -1}, {1, 1, -1}, {1, 1, 1}, {1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}, {-1, -1, 1}, {-1, -1, 1}, {-1, 1, 1}, {-1, 1, -1}, {-1, -1, -1}, {1, 1, -1}, {1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}};

constexpr float CubeNormals[][3] = {{0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}};

constexpr float CubeTexcoords[][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}};

constexpr uint32_t CubeTriangles[][3] = {{0, 1, 2}, {3, 0, 2}, {4, 5, 6}, {7, 4, 6}, {8, 9, 10}, {11, 8, 10}, {12, 13, 14}, {15, 12, 14}, {16, 17, 18}, {19, 16, 18}, {20, 21, 22}, {23, 20, 22}};

Cube::Cube(Material *material,
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
    pdf_area_ = 1.0 / area_;
    for (auto &mesh : meshes_)
        mesh->SetPdfArea(this->pdf_area_);
    
}

Cube::~Cube()
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

NAMESPACE_END(simple_renderer)