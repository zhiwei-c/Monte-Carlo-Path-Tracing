#pragma once

#include "../../modeling/shape.h"

inline void LoadDisk(ShapeInfo *shape_info, bool bump_mapping, std::vector<Vertex> &vertex_list, std::vector<uvec3> &idx_list)
{
    auto to_world_pos = static_cast<gmat4 *>(nullptr),
         to_world_norm = static_cast<gmat4 *>(nullptr);
    auto to_world = shape_info->to_world;
    if (to_world != nullptr)
    {
        to_world_pos = new gmat4(*to_world);
        to_world_norm = new gmat4(glm::inverse(glm::transpose(*to_world)));
    }

    auto center = gvec3(shape_info->center.x, shape_info->center.y, shape_info->center.z);
    auto normal = gvec3(0, 0, 1);
    if (to_world_norm)
        normal = TransfromDir(*to_world_norm, normal);
    normal = normal;

    const uint phi_steps = 40;
    const Float d_phi = (2.0 * kPi) / (phi_steps - 1);

    auto vector = gvec3(0);
    auto old_v_num = vertex_list.size();
    auto vertex_num = 2 * phi_steps;
    vertex_list.resize(old_v_num + vertex_num);
    for (uint i = 0; i < phi_steps; i++)
    {
        auto phi = i * d_phi;

        vector = center;
        if (to_world_pos)
            vector = TransfromPt(*to_world_pos, vector);
        vertex_list[old_v_num + i].position = vector;
        vertex_list[old_v_num + i].normal = normal;
        vertex_list[old_v_num + i].texcoord = vec2(0, phi * kPiInv * 0.5);

        vector = gvec3(std::cos(phi), std::sin(phi), 0);
        if (to_world_pos)
            vector = TransfromPt(*to_world_pos, vector);
        vertex_list[old_v_num + i + phi_steps].position = vector;
        vertex_list[old_v_num + i + phi_steps].normal = normal;
        vertex_list[old_v_num + i + phi_steps].texcoord = vec2(1.0, phi * kPiInv * 0.5);
    }

    if (to_world)
    {
        delete to_world_pos;
        delete to_world_norm;
    }

    auto old_i_num = idx_list.size();
    auto mesh_num = phi_steps - 1;
    idx_list.resize(old_i_num + mesh_num);
    for (uint i = 0; i < mesh_num; i++)
    {
        idx_list[old_i_num + i][0] = old_v_num + i;
        idx_list[old_i_num + i][1] = old_v_num + i + phi_steps;
        idx_list[old_i_num + i][2] = old_v_num + i + phi_steps + 1;
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