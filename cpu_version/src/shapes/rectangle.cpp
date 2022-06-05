#include "rectangle.h"

NAMESPACE_BEGIN(raytracer)

constexpr float RectanglePositions[][3] = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};

constexpr float RectangleNormals[][3] = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};

constexpr float RectangleTexcoords[][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

constexpr uint32_t RectangleTriangles[][3] = {{0, 1, 2}, {2, 3, 0}};

Rectangle::Rectangle(Bsdf *bsdf, Medium *int_medium, Medium *ext_medium, std::unique_ptr<Mat4> to_world, bool flip_normals)
    : Shape(ShapeType::kRectangle, bsdf, int_medium, ext_medium, flip_normals)
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
    for (auto i = 0; i < 2; ++i)
    {
        indices = {RectangleTriangles[i][0],
                   RectangleTriangles[i][1],
                   RectangleTriangles[i][2]};

        vertices.clear();
        normals.clear();
        texcoords.clear();

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
        meshes_.push_back(new Triangle(vertices, normals, texcoords, bsdf, int_medium, ext_medium, flip_normals));
    }
    bvh_ = std::make_unique<BvhAccel>(meshes_);
    aabb_ = bvh_->aabb();
    area_ = bvh_->area();
    pdf_area_ = 1 / area_;
    for (auto &mesh : meshes_)
        mesh->SetPdfArea(this->pdf_area_);
}

NAMESPACE_END(raytracer)