#pragma once

#include "../core/shape_base.h"

__device__ void Mesh::InitTriangle(Vertex *v,
                                   Material **material,
                                   Float area,
                                   Mesh *pre,
                                   Mesh *next)
{
    for (int k = 0; k < 3; k++)
    {
        v_[k] = v[k];
    }
    material_ = material;

    if (v[0].normal.length() < kEpsilon)
    {
        auto fn = myvec::normalize(myvec::cross(v[1].position - v[0].position, v[2].position - v[0].position));
        for (int k = 0; k < 3; k++)
        {
            v_[k].normal = fn;
        }
    }
    area_ = area;
    pre_ = pre;
    next_ = next;
}

__device__ void Mesh::Intersect(const Ray &ray, const vec2 &sample, Intersection &its) const
{
    vec3 v0v2 = v_[2].position - v_[0].position,
         v0v1 = v_[1].position - v_[0].position,
         P = myvec::cross(ray.dir(), v0v2);
    Float det = myvec::dot(v0v1, P);

    if (abs(det) < kEpsilon)
        return;

    Float det_inv = 1.0 / det;
    vec3 T = ray.origin() - v_[0].position;
    vec3 Q = myvec::cross(T, v0v1);

    Float u = myvec::dot(T, P) * det_inv;
    if (u < kEpsilon || u > 1 - kEpsilon)
        return;

    Float v = myvec::dot(ray.dir(), Q) * det_inv;
    if (v < kEpsilon || (u + v) > 1 - kEpsilon)
        return;

    Float distance = myvec::dot(v0v2, Q) * det_inv;
    if (distance < kEpsilon || distance > its.distance())
        return;

    if (!(*material_)->twosided())
    {
        if (flip_normals_)
        {
            if (det > kEpsilon)
            {
                its = Intersection(distance);
                return;
            }
        }
        else
        {
            if (det < -kEpsilon)
            {
                its = Intersection(distance);
                return;
            }
        }
    }

    Float alpha = 1.0 - u - v, beta = u, gamma = v;
    vec3 normal = alpha * v_[0].normal + beta * v_[1].normal + gamma * v_[2].normal;
    auto texcoord = vec2(0);
    if ((*material_)->TextureMapping())
    {
        texcoord = alpha * v_[0].texcoord + beta * v_[1].texcoord + gamma * v_[2].texcoord;
        if ((*material_)->BumpMapping())
        {
            vec3 tangent = myvec::normalize(alpha * v_[0].tangent +
                                            beta * v_[1].tangent +
                                            gamma * v_[2].tangent);
            vec3 bitangent = myvec::normalize(alpha * v_[0].bitangent +
                                              beta * v_[1].bitangent +
                                              gamma * v_[2].bitangent);
            normal = (*material_)->PerturbNormal(normal, tangent, bitangent, texcoord);
        }
    }
    if ((*material_)->Transparent(texcoord, sample))
        return;
    vec3 pos = alpha * v_[0].position + beta * v_[1].position + gamma * v_[2].position;
    int inside = -1;
    if (det < 0)
    {
        normal = -normal;
        inside = -inside;
    }
    if (flip_normals_)
    {
        normal = -normal;
        inside = -inside;
    }
    its = Intersection(pos, normal, texcoord, inside, distance, material_, pdf_area_);
}

__device__ void Mesh::SampleP(Intersection &its, const vec3 &sample) const
{

    auto alpha = static_cast<Float>(1 - sample.y - sample.z),
         beta = static_cast<Float>(sample.y),
         gamma = static_cast<Float>(sample.z);

    auto pos = alpha * v_[0].position + beta * v_[1].position + gamma * v_[2].position;
    auto normal = alpha * v_[0].normal + beta * v_[1].normal + gamma * v_[2].normal;

    its = Intersection(pos, normal, vec2(-1), false, INFINITY, material_, pdf_area_);
}