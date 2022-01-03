#include "triangle.h"

NAMESPACE_BEGIN(simple_renderer)

Triangle::Triangle(const std::vector<Vector3> &vertices,
                   const std::vector<Vector3> &normals,
                   const std::vector<Vector2> &texcoords,
                   Material *material,
                   bool flip_normals)
    : Shape(ShapeType::kTriangle, material, flip_normals),
      vertices_(vertices),
      texcoords_(texcoords)
{
    Setup(vertices, normals);
    if (material->NormalPerturbing())
    {
        auto delta_uv_1 = texcoords[1] - texcoords[0];
        auto delta_uv_2 = texcoords[2] - texcoords[0];
        auto r = 1 / (delta_uv_2.x * delta_uv_1.y - delta_uv_1.x * delta_uv_2.y);
        auto tangent = glm::normalize(Vector3(delta_uv_1.y * v0v2_ - delta_uv_2.y * v0v1_) * r);
        tangents_ = std::vector<Vector3>(3, tangent);
        auto bitangent = glm::normalize(Vector3(delta_uv_2.x * v0v1_ - delta_uv_1.x * v0v2_) * r);
        bitangents_ = std::vector<Vector3>(3, bitangent);
    }
}

Triangle::Triangle(const std::vector<Vector3> &vertices,
                   const std::vector<Vector3> &normals,
                   const std::vector<Vector2> &texcoords,
                   const std::vector<Vector3> &tangents,
                   const std::vector<Vector3> &bitangents,
                   Material *material,
                   bool flip_normals)
    : Shape(ShapeType::kTriangle, material, flip_normals),
      vertices_(vertices),
      texcoords_(texcoords),
      tangents_(tangents),
      bitangents_(bitangents)
{
    Setup(vertices, normals);
}

void Triangle::Setup(const std::vector<Vector3> &vertices, const std::vector<Vector3> &normals)
{
    v0v1_ = vertices[1] - vertices[0];
    v0v2_ = vertices[2] - vertices[0];

    if (normals.empty())
    {
        auto fn = glm::normalize(glm::cross(v0v1_, v0v2_));
        normals_ = std::vector<Vector3>(3, fn);
    }
    else
    {
        normals_ = normals;
    }

    area_ = glm::length(glm::cross(v0v1_, v0v2_)) * 0.5;

    aabb_ = AABB();
    for (int i = 0; i < vertices.size(); i++)
    {
        aabb_ += vertices[i];
    }
}

Intersection Triangle::Intersect(const Ray &ray) const
{

    auto P = glm::cross(ray.dir(), this->v0v2_);
    auto det = glm::dot(this->v0v1_, P);

    if (material_->twosided()) //丢弃与三角面片平行的光线
    {
        if (std::fabs(det) < kEpsilon)
            return Intersection();
    }
    else
    {
        if (flip_normals_)
        {
            if (det > -kEpsilon)
                return Intersection();
        }
        else
        {
            if (det < kEpsilon)
                return Intersection();
        }
    }

    auto det_inv = 1 / det;
    auto T = ray.origin() - this->vertices_[0];

    auto u = glm::dot(T, P) * det_inv;
    if (u < kEpsilon || u > 1 - kEpsilon)
        return Intersection();

    auto Q = glm::cross(T, this->v0v1_);
    auto v = glm::dot(ray.dir(), Q) * det_inv;
    if (v < kEpsilon || (u + v) > 1 - kEpsilon)
        return Intersection();

    auto t = glm::dot(this->v0v2_, Q) * det_inv;
    if (t < kEpsilon)
        return Intersection();

    auto alpha = static_cast<Float>(1 - u - v),
         beta = static_cast<Float>(u),
         gamma = static_cast<Float>(v);

    auto normal = alpha * this->normals_[0] + beta * this->normals_[1] + gamma * this->normals_[2];
    auto texcoord = Vector2(-1);
    if (this->material_->NormalPerturbing() ||
        this->material_->OpacityMapping() ||
        this->material_->TextureMapping())
    {
        texcoord = alpha * this->texcoords_[0] + beta * this->texcoords_[1] + gamma * this->texcoords_[2];
        if (this->material_->Transparent(texcoord))
            return Intersection();
        if (this->material_->NormalPerturbing())
        {
            auto tangent = glm::normalize(alpha * this->tangents_[0] +
                                          beta * this->tangents_[1] +
                                          gamma * this->tangents_[2]);
            auto bitangent = glm::normalize(alpha * this->bitangents_[0] +
                                            beta * this->bitangents_[1] +
                                            gamma * this->bitangents_[2]);
            normal = this->material_->PerturbNormal(normal, tangent, bitangent, texcoord);
        }
    }

    auto pos = alpha * this->vertices_[0] + beta * this->vertices_[1] + gamma * this->vertices_[2];
    auto inside = false;
    if (det < 0)
    {
        normal = -normal;
        inside = !inside;
    }
    if (flip_normals_)
    {
        normal = -normal;
        inside = !inside;
    }

    return Intersection(pos, normal, texcoord, inside, t, this->material_);
}

std::pair<Intersection, Float> Triangle::SampleP(const Vector3 &pos_pre) const
{
    auto u = UniformFloat();
    auto v = UniformFloat();

    auto alpha = static_cast<Float>(1 - u - v),
         beta = static_cast<Float>(u),
         gamma = static_cast<Float>(v);

    auto pos = alpha * this->vertices_[0] + beta * this->vertices_[1] + gamma * this->vertices_[2];
    auto normal = alpha * this->normals_[0] + beta * this->normals_[1] + gamma * this->normals_[2];
    auto distance = glm::length(pos - pos_pre);

    return {Intersection(pos, normal, Vector2(-1), false, distance, this->material_), 1 / this->area_};
}

NAMESPACE_END(simple_renderer)