#pragma once

#include "shape_base.h"

__device__ void Mesh::InitTriangle(Vertex *v,
                                   Material *material,
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
    auto v0v2 = v_[2].position - v_[0].position;
    auto v0v1 = v_[1].position - v_[0].position;
    auto P = myvec::cross(ray.dir(), v0v2);
    auto det = myvec::dot(v0v1, P);

    if (abs(det) < kEpsilon)
        return;

    auto det_inv = 1 / det;
    auto T = ray.origin() - v_[0].position;

    auto Q = myvec::cross(T, v0v1);

    auto u = myvec::dot(T, P) * det_inv;
    if (u < kEpsilon || u > 1 - kEpsilon)
        return;

    auto v = myvec::dot(ray.dir(), Q) * det_inv;
    if (v < kEpsilon || (u + v) > 1 - kEpsilon)
        return;

    auto distance = myvec::dot(v0v2, Q) * det_inv;
    if (distance < kEpsilon || distance > its.distance())
        return;

    if (!material_->twosided())
    {
        if (flip_normals_)
        {
            if (det > -kEpsilon)
            {
                its = Intersection(distance);
                return;
            }
        }
        else
        {
            if (det < kEpsilon)
            {
                its = Intersection(distance);
                return;
            }
        }
    }

    auto alpha = static_cast<Float>(1 - u - v),
         beta = static_cast<Float>(u),
         gamma = static_cast<Float>(v);

    auto texcoord = alpha * v_[0].texcoord +
                    beta * v_[1].texcoord +
                    gamma * v_[2].texcoord;
    if (material_->Transparent(texcoord, sample))
        return;

    auto normal = alpha * v_[0].normal +
                  beta * v_[1].normal +
                  gamma * v_[2].normal;
    if (material_->BumpMapping())
    {
        auto tangent = myvec::normalize(alpha * v_[0].tangent +
                                        beta * v_[1].tangent +
                                        gamma * v_[2].tangent);
        auto bitangent = myvec::normalize(alpha * v_[0].bitangent +
                                          beta * v_[1].bitangent +
                                          gamma * v_[2].bitangent);
        normal = material_->PerturbNormal(normal, tangent, bitangent, texcoord);
    }

    auto pos = alpha * v_[0].position +
               beta * v_[1].position +
               gamma * v_[2].position;

    bool inside = false;

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