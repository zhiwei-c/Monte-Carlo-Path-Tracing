#include "cube.hpp"

#include "triangle.hpp"
#include "../math/coordinate.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"

static constexpr double kPositions[][3] = {{1, -1, -1}, {1, -1, 1}, {-1, -1, 1}, {-1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, 1, 1}, {1, 1, 1}, {1, -1, -1}, {1, 1, -1}, {1, 1, 1}, {1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}, {-1, -1, 1}, {-1, -1, 1}, {-1, 1, 1}, {-1, 1, -1}, {-1, -1, -1}, {1, 1, -1}, {1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}};
static constexpr double kNormals[][3] = {{0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}};
static constexpr double kTexcoords[][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}};
static constexpr unsigned int kIndices[][3] = {{0, 1, 2}, {3, 0, 2}, {4, 5, 6}, {7, 4, 6}, {8, 9, 10}, {11, 8, 10}, {12, 13, 14}, {15, 12, 14}, {16, 17, 18}, {19, 16, 18}, {20, 21, 22}, {23, 20, 22}};

NAMESPACE_BEGIN(raytracer)

Cube::Cube(const std::string &id, const dmat4 &to_world, bool flip_normals)
    : Shape(id, ShapeType::kCube, flip_normals)
{
    std::array<dvec3, 48> vertex_buffer;
    dmat4 normal_to_world = glm::inverse(glm::transpose(to_world));
    for (int i = 0; i < 24; ++i)
    {
        vertex_buffer[i] = TransfromPoint(to_world, {kPositions[i][0], kPositions[i][1], kPositions[i][2]});
        vertex_buffer[i + 24] = TransfromVec(normal_to_world, {kNormals[i][0], kNormals[i][1], kNormals[i][2]});
    }

    area_ = 0;
    aabb_ = AABB();
    for (int i = 0; i < 12; ++i)
    {
        std::array<unsigned int, 3> indices = {kIndices[i][0], kIndices[i][1], kIndices[i][2]};
        auto positions = std::vector<dvec3>(3);
        auto normals = std::vector<dvec3>(3);
        auto texcoords = std::vector<dvec2>(3);

        for (int v : {0, 1, 2})
        {
            positions[v] = vertex_buffer[indices[v]];
            normals[v] = vertex_buffer[indices[v] + 24];
            texcoords[v] = {kTexcoords[indices[v]][0], kTexcoords[indices[v]][1]};
        }

        dvec2 delta_uv_1 = texcoords[1] - texcoords[0],
              delta_uv_2 = texcoords[2] - texcoords[0];
        double r = 1.0 / (delta_uv_2.x * delta_uv_1.y - delta_uv_1.x * delta_uv_2.y);
        dvec3 v0v1 = positions[1] - positions[0],
              v0v2 = positions[2] - positions[0],
              tangent = glm::normalize(dvec3(delta_uv_1.y * v0v2 - delta_uv_2.y * v0v1) * r),
              bitangent = glm::normalize(dvec3(delta_uv_2.x * v0v1 - delta_uv_1.x * v0v2) * r);
        auto tangents = std::vector<dvec3>(3, tangent),
             bitangents = std::vector<dvec3>(3, bitangent);
        meshes_[i] = new Triangle(id, positions, normals, tangents, bitangents, texcoords, flip_normals);
        area_ += meshes_[i]->area();
        aabb_ += meshes_[i]->aabb();
    }

    pdf_area_ = 1.0 / area_;
    for (Shape *mesh : meshes_)
    {
        mesh->SetPdfArea(pdf_area_);
    }
}

Cube::~Cube()
{
    for (Shape *&mesh : meshes_)
    {
        delete mesh;
        mesh = nullptr;
    }
}

void Cube::Intersect(const Ray &ray, Sampler *sampler, Intersection *its) const
{
    if (aabb_.Intersect(ray))
    {
        for (Shape *mesh : meshes_)
        {
            mesh->Intersect(ray, sampler, its);
        }
    }
}

Intersection Cube::SamplePoint(Sampler *sampler) const
{
    double p = sampler->Next1D() * area_;
    for (Shape *mesh : meshes_)
    {
        if (p < mesh->area())
        {
            return mesh->SamplePoint(sampler);
        }
        p -= mesh->area();
    }
    return meshes_.back()->SamplePoint(sampler);
}

void Cube::SetBsdf(Bsdf *bsdf)
{
    Shape::SetBsdf(bsdf);
    for (Shape *&mesh : meshes_)
    {
        mesh->SetBsdf(bsdf);
    }
}

void Cube::SetMedium(Medium *medium_int, Medium *medium_ext)
{
    Shape::SetMedium(medium_int, medium_ext);
    for (Shape *&mesh : meshes_)
    {
        mesh->SetMedium(medium_int, medium_ext);
    }
}

NAMESPACE_END(raytracer)