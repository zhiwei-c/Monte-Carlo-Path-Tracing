#include "triangle.h"

NAMESPACE_BEGIN(raytracer)

Triangle::Triangle(const std::vector<Vector3> &vertices,
                   const std::vector<Vector3> &normals,
                   const std::vector<Vector2> &texcoords,
                   Bsdf *bsdf,
                   bool flip_normals)
    : Shape(ShapeType::kTriangle, bsdf, flip_normals),
      vertices_(vertices),
      texcoords_(texcoords)
{
    Setup(vertices, normals);
    if (bsdf->NormalPerturbing())
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
                   Bsdf *bsdf,
                   bool flip_normals)
    : Shape(ShapeType::kTriangle, bsdf, flip_normals),
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
    pdf_area_ = 1 / area_;

    aabb_ = AABB();
    for (const auto &v : vertices)
    {
        aabb_ += v;
    }
}

void Triangle::Intersect(const Ray &ray, Intersection &its) const
{
    Vector3 P = glm::cross(ray.dir(), v0v2_);
    Float det = glm::dot(v0v1_, P);
    if (std::abs(det) < kEpsilon)
        return; //丢弃与三角面片平行的光线
    Vector3 T = ray.origin() - vertices_[0],
            Q = glm::cross(T, v0v1_);
    Float det_inv = 1.0 / det,
          u = glm::dot(T, P) * det_inv;
    if (u < kEpsilon || u > 1 - kEpsilon)
        return;
    Float v = glm::dot(ray.dir(), Q) * det_inv;
    if (v < kEpsilon || (u + v) > 1 - kEpsilon)
        return;
    Float distance = glm::dot(v0v2_, Q) * det_inv;
    if (distance < kEpsilon || distance > its.distance())
        return;
    if (!bsdf_->Twosided() &&
        (flip_normals_ && det > -kEpsilon ||
         !flip_normals_ && det < kEpsilon))
    { //丢弃与单面材质的三角面片交于背面的光线
        its = Intersection(distance);
        return;
    }
    Float alpha = 1.0 - u - v, beta = u, gamma = v;
    Vector3 normal = alpha * normals_[0] + beta * normals_[1] + gamma * normals_[2];
    auto texcoord = Vector2(-1);
    if (bsdf_->TextureMapping())
    {
        texcoord = alpha * texcoords_[0] + beta * texcoords_[1] + gamma * texcoords_[2];
        if (bsdf_->Transparent(texcoord))
            return;

        if (bsdf_->NormalPerturbing())
        {
            Vector3 tangent = glm::normalize(alpha * tangents_[0] + beta * tangents_[1] + gamma * tangents_[2]),
                    bitangent = glm::normalize(alpha * bitangents_[0] + beta * bitangents_[1] + gamma * bitangents_[2]);
            normal = bsdf_->PerturbNormal(normal, tangent, bitangent, texcoord);
        }
    }
    Vector3 pos = alpha * vertices_[0] + beta * vertices_[1] + gamma * vertices_[2];
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
    its = Intersection(pos, normal, texcoord, inside, distance, bsdf_, pdf_area_);
}

Intersection Triangle::SampleP() const
{
    Float u = UniformFloat(), v = UniformFloat();
    Float alpha = 1.0 - u - v, beta = u, gamma = v;
    Vector3 pos = alpha * vertices_[0] + beta * vertices_[1] + gamma * vertices_[2],
            normal = alpha * normals_[0] + beta * normals_[1] + gamma * normals_[2];
    return Intersection(pos, normal, Vector2(-1), false, INFINITY, bsdf_, pdf_area_);
}

NAMESPACE_END(raytracer)