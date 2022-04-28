#pragma once

#include "../../modeling/shape.h"

static constexpr uint theta_steps = 20;
static constexpr uint phi_steps = theta_steps * 2;
static constexpr Float d_theta = kPi / (theta_steps - 1);
static constexpr Float d_phi = (2 * kPi) / (phi_steps - 1);

std::vector<Float> PrecomputeCosine()
{
    auto result = std::vector<Float>(phi_steps);
    for (uint32_t i = 0; i < phi_steps; ++i)
        result[i] = std::cos(i * d_phi);
    return result;
}
static const std::vector<Float> cos_phi = PrecomputeCosine();

std::vector<Float> PrecomputeSine()
{
    auto result = std::vector<Float>(phi_steps);
    for (uint32_t i = 0; i < phi_steps; ++i)
        result[i] = std::sin(i * d_phi);
    return result;
}
static const std::vector<Float> sin_phi = PrecomputeSine();

inline void LoadSphere(ShapeInfo *shape_info, bool bump_mapping, std::vector<Vertex> &vertex_list, std::vector<uvec3> &idx_list)
{
    gmat4 *to_world_pos = nullptr, *to_world_norm = nullptr;
    auto to_world = shape_info->to_world;
    if (to_world != nullptr)
    {
        to_world_pos = new gmat4(*to_world);
        to_world_norm = new gmat4(glm::inverse(glm::transpose(*to_world)));
    }

    auto vector_tmp = gvec3(0);
    auto vector = gvec3(0);

    auto old_v_num = vertex_list.size();
    auto vertex_num = theta_steps * phi_steps;
    vertex_list.resize(old_v_num + vertex_num);
    uint vertex_idx = 0;
    for (uint i_theta = 0; i_theta < theta_steps; i_theta++)
    {
        auto sin_theta = std::sin(i_theta * d_theta);
        auto cos_theta = std::cos(i_theta * d_theta);
        for (uint i_phi = 0; i_phi < phi_steps; i_phi++)
        {
            vector_tmp = gvec3(sin_theta * cos_phi[i_phi],
                               sin_theta * sin_phi[i_phi],
                               cos_theta);

            vector = gvec3(shape_info->center.x + vector_tmp.x * shape_info->radius,
                           shape_info->center.y + vector_tmp.y * shape_info->radius,
                           shape_info->center.z + vector_tmp.z * shape_info->radius);
            if (to_world_pos)
                vector = TransfromPt(*to_world_pos, vector);
            vertex_list[old_v_num + vertex_idx].position = vector;

            vector = glm::normalize(vector_tmp);
            if (to_world_norm)
                vector = TransfromDir(*to_world_norm, vector);
            vertex_list[old_v_num + vertex_idx].normal = vector;

            vertex_list[old_v_num + vertex_idx].texcoord = vec2(i_phi * d_phi * kPiInv * 0.5, i_theta * d_theta * kPiInv);

            vertex_idx++;
        }
    }

    if (to_world)
    {
        delete to_world_pos;
        delete to_world_norm;
    }

    auto old_i_num = idx_list.size();
    auto mesh_num = 2 * (phi_steps - 1) * (theta_steps - 1);
    idx_list.resize(old_i_num + mesh_num);
    uint mesh_idx = 0;
    for (uint i_theta = 1; i_theta < theta_steps; i_theta++)
    {

        for (uint i_phi = 0; i_phi < phi_steps - 1; i_phi++)
        {
            auto next_i_phi = i_phi + 1;
            auto idx0 = old_v_num + phi_steps * i_theta + i_phi;
            auto idx1 = old_v_num + phi_steps * i_theta + next_i_phi;
            auto idx2 = old_v_num + phi_steps * (i_theta - 1) + i_phi;
            auto idx3 = old_v_num + phi_steps * (i_theta - 1) + next_i_phi;

            idx_list[old_i_num + mesh_idx][0] = idx0;
            idx_list[old_i_num + mesh_idx][1] = idx2;
            idx_list[old_i_num + mesh_idx][2] = idx1;
            mesh_idx++;

            idx_list[old_i_num + mesh_idx][0] = idx1;
            idx_list[old_i_num + mesh_idx][1] = idx2;
            idx_list[old_i_num + mesh_idx][2] = idx3;
            mesh_idx++;
        }
    }

    if (!bump_mapping)
        return;

    for (uint i = 0; i < mesh_num; i++)
    {
        auto v = std::vector<Vertex>(3);
        for (int j = 0; j < 3; j++)
        {
            v[j] = vertex_list[idx_list[old_i_num + i][j]];
        }

        auto v0v1 = v[1].position - v[0].position;
        auto v0v2 = v[2].position - v[0].position;
        auto delta_uv_1 = v[1].texcoord - v[0].texcoord;
        auto delta_uv_2 = v[2].texcoord - v[0].texcoord;
        auto r = 1.0 / (delta_uv_2.x * delta_uv_1.y - delta_uv_1.x * delta_uv_2.y);
        auto tangent = myvec::normalize(vec3(delta_uv_1.y * v0v2 - delta_uv_2.y * v0v1) * r);
        auto bitangent = myvec::normalize(vec3(delta_uv_2.x * v0v1 - delta_uv_1.x * v0v2) * r);
        for (int j = 0; j < 3; j++)
        {
            vertex_list[idx_list[old_i_num + i][j]].tangent = tangent;
            vertex_list[idx_list[old_i_num + i][j]].bitangent = bitangent;
        }
    }
}