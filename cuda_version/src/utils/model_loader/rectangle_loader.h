#pragma once

#include "../../modeling/shape.h"

static constexpr float RectanglePositions[][3] = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};

static constexpr float RectangleNormals[][3] = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};

static constexpr float RectangleTexcoords[][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

static constexpr uint32_t RectangleTriangles[][3] = {{0, 1, 2}, {2, 3, 0}};

inline void LoadRectangle(ShapeInfo *shape_info, bool bump_mapping, std::vector<Vertex> &vertex_list, std::vector<uvec3> &idx_list)
{
    auto to_world_pos = static_cast<gmat4 *>(nullptr),
         to_world_norm = static_cast<gmat4 *>(nullptr);
    auto to_world = shape_info->to_world;
    if (to_world != nullptr)
    {
        to_world_pos = new gmat4(*to_world);
        to_world_norm = new gmat4(glm::inverse(glm::transpose(*to_world)));
    }

    auto old_v_num = vertex_list.size();
    vertex_list.resize(old_v_num + 4);
    auto vector = gvec3(1);
    auto vec = gvec2(0);
    for (int i = 0; i < 4; i++)
    {
        vector.x = RectanglePositions[i][0];
        vector.y = RectanglePositions[i][1];
        vector.z = RectanglePositions[i][2];
        if (to_world_pos)
            vector = TransfromPt(*to_world_pos, vector);

        vertex_list[old_v_num + i].position = vector;

        vector.x = RectangleNormals[i][0];
        vector.y = RectangleNormals[i][1];
        vector.z = RectangleNormals[i][2];
        if (to_world_norm)
            vector = TransfromDir(*to_world_norm, vector);
        vertex_list[old_v_num + i].normal = glm::normalize(vector);

        vec.x = RectangleTexcoords[i][0];
        vec.y = RectangleTexcoords[i][1];
        vertex_list[old_v_num + i].texcoord = vec;
    }

    if (to_world)
    {
        delete to_world_pos;
        delete to_world_norm;
    }

    auto old_i_num = idx_list.size();
    idx_list.resize(old_i_num + 2);
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
            idx_list[old_i_num + i][j] = old_v_num + RectangleTriangles[i][j];
    }

    if (!bump_mapping)
        return;

    for (int i = 0; i < 2; i++)
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