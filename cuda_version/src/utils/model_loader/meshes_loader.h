#pragma once

#include <iostream>
#include <vector>
#include <thread>

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

#include "../../modeling/shape.h"

inline void VertexBufferThread(aiMesh *mesh,
                               uint begin,
                               uint end,
                               gmat4 *to_world_pos,
                               gmat4 *to_world_norm,
                               bool face_normals,
                               uint old_v_num,
                               std::vector<Vertex> &vertex_list)
{
    auto vector = gvec3(0); // we declare a placeholder vector since assimp uses its own vector class that doesn't directly convert to glm's vec3 class so we transfer the data to this placeholder glm::vec3 first.
    auto vec = gvec2(0);
    for (uint i = begin; i < end; i++)
    {
        // positions
        vector.x = mesh->mVertices[i].x;
        vector.y = mesh->mVertices[i].y;
        vector.z = mesh->mVertices[i].z;
        if (to_world_pos)
            vector = TransfromPt(*to_world_pos, vector);

        vertex_list[old_v_num + i].position = vector;
        // normals
        if (!face_normals)
        {
            vector.x = mesh->mNormals[i].x;
            vector.y = mesh->mNormals[i].y;
            vector.z = mesh->mNormals[i].z;
            if (to_world_norm)
                vector = TransfromDir(*to_world_norm, vector);
            vertex_list[old_v_num + i].normal = glm::normalize(vector);
        }
        // texture coordinates
        if (mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
        {
            // a vertex can contain up to 8 different texture coordinates. We thus make the assumption that we won't
            // use models where a vertex can have multiple texture coordinates so we always take the first set (0).
            vec.x = mesh->mTextureCoords[0][i].x;
            vec.y = mesh->mTextureCoords[0][i].y;
            vertex_list[old_v_num + i].texcoord = vec;
            // tangent
            if (mesh->mTangents)
            {
                vector.x = mesh->mTangents[i].x;
                vector.y = mesh->mTangents[i].y;
                vector.z = mesh->mTangents[i].z;
                vertex_list[old_v_num + i].tangent = vector;
            }
            // bitangent
            if (mesh->mBitangents)
            {
                vector.x = mesh->mBitangents[i].x;
                vector.y = mesh->mBitangents[i].y;
                vector.z = mesh->mBitangents[i].z;
                vertex_list[old_v_num + i].bitangent = vector;
            }
        }
    }
}

inline void IndexBufferThread(aiMesh *mesh,
                              uint begin,
                              uint end,
                              uint old_v_num,
                              uint old_i_num,
                              std::vector<uvec3> &idx_list)
{
    for (uint i = begin; i < end; i++)
    {
        aiFace face = mesh->mFaces[i];
        for (int j = 0; j < 3; j++)
            idx_list[old_i_num + i][j] = old_v_num + face.mIndices[j];
    }
}

inline void ProcessMesh(const aiScene *scene,
                        aiMesh *mesh,
                        gmat4 *to_world_pos,
                        gmat4 *to_world_norm,
                        bool face_normals,
                        std::vector<Vertex> &vertex_list,
                        std::vector<uvec3> &idx_list)
{
    auto num_thread = std::thread::hardware_concurrency();
    auto wokers = std::vector<std::thread>();
    //
    auto blcok_num = mesh->mNumVertices < num_thread ? 1 : num_thread;
    auto block_length = mesh->mNumVertices / blcok_num;
    auto old_v_num = vertex_list.size();
    vertex_list.resize(old_v_num + mesh->mNumVertices);
    for (uint i = 0; i < blcok_num; i++)
    {
        auto begin = i * block_length;
        auto end = (i == blcok_num - 1) ? mesh->mNumVertices : ((i + 1) * block_length);
        wokers.push_back(std::thread(VertexBufferThread,
                                     mesh, begin, end,
                                     to_world_pos, to_world_norm, face_normals,
                                     old_v_num, std::ref(vertex_list)));
    }
    //
    blcok_num = mesh->mNumFaces < num_thread ? 1 : num_thread;
    block_length = mesh->mNumFaces / blcok_num;
    auto local_i_buffer = std::vector<uvec3>(mesh->mNumFaces);
    auto old_i_num = idx_list.size();
    idx_list.resize(old_i_num + mesh->mNumFaces);
    for (uint i = 0; i < blcok_num; i++)
    {
        auto begin = i * block_length;
        auto end = (i == blcok_num - 1) ? mesh->mNumFaces : ((i + 1) * block_length);
        wokers.push_back(std::thread(IndexBufferThread,
                                     mesh, begin, end,
                                     old_v_num, old_i_num,
                                     std::ref(idx_list)));
    }
    for (int i = 0; i < wokers.size(); i++)
    {
        wokers[i].join();
    }
}

inline void ProcessNode(const aiScene *scene,
                        aiNode *node,
                        gmat4 *to_world_pos,
                        gmat4 *to_world_norm,
                        bool face_normals,
                        std::vector<Vertex> &vertex_list,
                        std::vector<uvec3> &idx_list)
{
    for (uint i = 0; i < node->mNumMeshes; i++)
    {
        auto mesh = scene->mMeshes[node->mMeshes[i]];
        ProcessMesh(scene, mesh, to_world_pos, to_world_norm, face_normals, vertex_list, idx_list);
    }
    for (uint i = 0; i < node->mNumChildren; i++)
    {
        ProcessNode(scene, node->mChildren[i], to_world_pos, to_world_norm, face_normals, vertex_list, idx_list);
    }
}

inline void LoadMeshes(ShapeInfo *shape_info, bool bump_mapping, std::vector<Vertex> &vertex_list, std::vector<uvec3> &idx_list)
{
    std::cout << "[info] begin load " << shape_info->filename << "\r";

    auto to_world_pos = static_cast<gmat4 *>(nullptr),
         to_world_norm = static_cast<gmat4 *>(nullptr);
    auto to_world = shape_info->to_world;
    if (to_world != nullptr)
    {
        to_world_pos = new gmat4(*to_world);
        to_world_norm = new gmat4(glm::inverse(glm::transpose(*to_world)));
    }

    auto importer = Assimp::Importer();
    auto option = aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_GenUVCoords;
    if (shape_info->flip_tex_coords)
        option = option | aiProcess_FlipUVs;
    if (bump_mapping)
        option = option | aiProcess_CalcTangentSpace;
    auto scene = importer.ReadFile(shape_info->filename, option);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
    {
        std::cerr << "[error] ASSIMP:: " << importer.GetErrorString() << std::endl;
        exit(1);
    }

    ProcessNode(scene, scene->mRootNode, to_world_pos, to_world_norm, shape_info->face_normals, vertex_list, idx_list);

    if (to_world)
    {
        delete to_world_pos;
        delete to_world_norm;
    }
    std::cout << "[info] load " << shape_info->filename << " finished\n";
}
