#include "../model_loader.h"

constexpr float CubePositions[][3] = {{1, -1, -1}, {1, -1, 1}, {-1, -1, 1}, {-1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, 1, 1}, {1, 1, 1}, {1, -1, -1}, {1, 1, -1}, {1, 1, 1}, {1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}, {-1, -1, 1}, {-1, -1, 1}, {-1, 1, 1}, {-1, 1, -1}, {-1, -1, -1}, {1, 1, -1}, {1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}};

constexpr float CubeNormals[][3] = {{0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}};

constexpr float CubeTexcoords[][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}};

constexpr uint32_t CubeTriangles[][3] = {{0, 1, 2}, {3, 0, 2}, {4, 5, 6}, {7, 4, 6}, {8, 9, 10}, {11, 8, 10}, {12, 13, 14}, {15, 12, 14}, {16, 17, 18}, {19, 16, 18}, {20, 21, 22}, {23, 20, 22}};

void LoadCube(ShapeInfo *shape_info, bool bump_mapping, std::vector<Vertex> &vertex_list, std::vector<uvec3> &idx_list)
{
    gmat4 *to_world_pos = nullptr,
          *to_world_norm = nullptr;
    auto to_world = shape_info->to_world;
    if (to_world != nullptr)
    {
        to_world_pos = new gmat4(*to_world);
        to_world_norm = new gmat4(glm::inverse(glm::transpose(*to_world)));
    }

    auto old_v_num = vertex_list.size();
    vertex_list.resize(old_v_num + 24);
    auto vector = gvec3(1);
    auto vec = gvec2(0);
    for (int i = 0; i < 24; i++)
    {
        vector.x = CubePositions[i][0];
        vector.y = CubePositions[i][1];
        vector.z = CubePositions[i][2];
        if (to_world_pos)
            vector = TransfromPt(*to_world_pos, vector);

        vertex_list[old_v_num + i].position = vector;

        vector.x = CubeNormals[i][0];
        vector.y = CubeNormals[i][1];
        vector.z = CubeNormals[i][2];
        if (to_world_norm)
            vector = TransfromDir(*to_world_norm, vector);
        vertex_list[old_v_num + i].normal = glm::normalize(vector);

        vec.x = CubeTexcoords[i][0];
        vec.y = CubeTexcoords[i][1];
        vertex_list[old_v_num + i].texcoord = vec;
    }

    if (to_world)
    {
        delete to_world_pos;
        delete to_world_norm;
    }

    auto old_i_num = idx_list.size();
    idx_list.resize(old_i_num + 12);
    for (int i = 0; i < 12; i++)
    {
        for (int j = 0; j < 3; j++)
            idx_list[old_i_num + i][j] = old_v_num + CubeTriangles[i][j];
    }

    if (!bump_mapping)
        return;

    for (int i = 0; i < 12; i++)
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