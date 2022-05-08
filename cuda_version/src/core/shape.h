#pragma once

#include "../shapes/triangle.h"

__global__ void CreateMeshes(uint max_x,
                             uint max_y,
                             uint mesh_num,
                             Vertex *v_buffer,
                             uvec3 *i_buffer,
                             uint *m_idx,
                             Material *materials,
                             AABB *mesh_aabbs,
                             Float *mesh_areas,
                             Mesh *mesh_list)
{
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;

    auto idx = j * max_x + i;
    if (idx >= mesh_num)
        return;

    Vertex v[3];
    mesh_aabbs[idx] = AABB();
    for (int k = 0; k < 3; k++)
    {
        v[k] = v_buffer[i_buffer[idx][k]];
        mesh_aabbs[idx] += v[k].position;
    }

    auto pre = static_cast<Mesh *>(nullptr);
    if (idx > 0)
        pre = mesh_list + idx - 1;

    auto next = static_cast<Mesh *>(nullptr);
    if (idx + 1 < mesh_num)
        next = mesh_list + idx + 1;

    mesh_areas[idx] = myvec::length(myvec::cross(v[1].position - v[0].position, v[2].position - v[0].position)) * static_cast<Float>(0.5);
    mesh_list[idx].InitTriangle(v,
                                materials + m_idx[idx],
                                mesh_areas[idx],
                                pre,
                                next);
}

__global__ void SetMeshesOtherInfo(uint mesh_idx_begin,
                                   uint mesh_num,
                                   uint shape_idx,
                                   bool flip_normals,
                                   Float shape_area,
                                   Mesh *mesh_list_)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (uint i = 0; i < mesh_num; i++)
        {
            mesh_list_[mesh_idx_begin + i].SetOtherInfo(i, shape_idx, flip_normals, static_cast<Float>(1) / shape_area);
        }
    }
}
