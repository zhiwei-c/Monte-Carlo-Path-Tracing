#include "object_loader.hpp"

#include <thread>
#include <iostream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "../math/coordinate.hpp"
#include "../shapes/triangle.hpp"

NAMESPACE_BEGIN(raytracer)

void ProcessMesh(aiMesh *mesh, unsigned int begin, unsigned int end, bool face_normals, bool flip_normals,
                 const std::string &id, const dmat4 &to_world, std::vector<Shape *> &shapes);
std::vector<Shape *> ProcessNode(const aiScene *scene, aiNode *node, bool face_normals, bool flip_normals,
                                 const std::string &id, const dmat4 &to_world);

std::vector<Shape *> LoadObject(const std::string &filename, bool face_normals, bool flip_normals, bool flip_tex_coords,
                                const std::string &id, const dmat4 &to_world)
{
    int assimp_option = aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_GenUVCoords |
                        aiProcess_CalcTangentSpace;
    if (flip_tex_coords)
    {
        assimp_option = assimp_option | aiProcess_FlipUVs;
    }
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(filename, assimp_option);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        std::cerr << "[error] ASSIMP:: " << importer.GetErrorString() << std::endl;
        exit(1);
    }

    std::cerr << "[info] read file \"" << filename << "\"\n";

    return ProcessNode(scene, scene->mRootNode, face_normals, flip_normals, id, to_world);
}

std::vector<Shape *> ProcessNode(const aiScene *scene, aiNode *node, bool face_normals, bool flip_normals,
                                 const std::string &id, const dmat4 &to_world)
{
    int thread_num = std::max(static_cast<int>(std::thread::hardware_concurrency()) - 1, 1);

    std::vector<Shape *> shapes;
    for (unsigned int i = 0; i < node->mNumMeshes; ++i)
    {
        aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
        unsigned int face_nums = mesh->mNumFaces;

        auto local_shapes = std::vector<Shape *>(face_nums);
        if (face_nums < static_cast<unsigned int>(thread_num))
        {
            ProcessMesh(mesh, 0, face_nums, face_normals, flip_normals, id, to_world, local_shapes);
        }
        else
        {
            unsigned int block_length = face_nums / thread_num;
            std::vector<std::thread> workers;
            for (int j = 0; j < thread_num; ++j)
            {
                unsigned int begin = j * block_length,
                             end = (j == thread_num - 1) ? face_nums : ((j + 1) * block_length);
                workers.push_back(std::thread{ProcessMesh, mesh, begin, end, face_normals, flip_normals,
                                              id, to_world, std::ref(local_shapes)});
            }
            for (size_t i = 0; i < workers.size(); ++i)
            {
                workers[i].join();
            }
        }

        shapes.insert(shapes.end(), local_shapes.begin(), local_shapes.end());
    }

    for (unsigned int i = 0; i < node->mNumChildren; ++i)
    {
        std::vector<Shape *> local_shapes = ProcessNode(scene, node->mChildren[i], face_normals, flip_normals,
                                                        id, to_world);
        shapes.insert(shapes.end(), local_shapes.begin(), local_shapes.end());
    }

    return shapes;
}

void ProcessMesh(aiMesh *mesh, unsigned int begin, unsigned int end, bool face_normals, bool flip_normals,
                 const std::string &id, const dmat4 &to_world, std::vector<Shape *> &shapes)
{
    dmat4 normal_to_world = glm::inverse(glm::transpose(to_world));
    for (unsigned int face_index = begin; face_index < end; face_index++)
    {
        aiFace face = mesh->mFaces[face_index];
        unsigned int indices[3] = {face.mIndices[0], face.mIndices[1], face.mIndices[2]};

        auto positions = std::vector<dvec3>(3);
        for (int v = 0; v < 3; ++v)
        {
            positions[v] = {mesh->mVertices[indices[v]].x, mesh->mVertices[indices[v]].y, mesh->mVertices[indices[v]].z};
            positions[v] = TransfromPoint(to_world, positions[v]);
        }

        auto normals = std::vector<dvec3>(3);
        if (face_normals)
        {
            const dvec3 v0v1 = positions[1] - positions[0],
                        v0v2 = positions[2] - positions[0];
            dvec3 normal = glm::normalize(glm::cross(v0v1, v0v2));
            normal = TransfromVec(normal_to_world, normal);
            normals = std::vector<dvec3>(3, normal);
        }
        else
        {
            for (int v = 0; v < 3; ++v)
            {
                normals[v] = {mesh->mNormals[indices[v]].x, mesh->mNormals[indices[v]].y, mesh->mNormals[indices[v]].z};
                normals[v] = TransfromVec(normal_to_world, normals[v]);
            }
        }

        auto texcoords = std::vector<dvec2>(3, {0, 0});
        if (mesh->mTextureCoords[0])
        {
            for (int v = 0; v < 3; ++v)
            {
                texcoords[v] = {mesh->mTextureCoords[0][indices[v]].x, mesh->mTextureCoords[0][indices[v]].y};
            }
        }

        auto tangents = std::vector<dvec3>(3);
        for (int v = 0; v < 3; ++v)
        {
            if (mesh->mTangents)
            {
                tangents[v] = {mesh->mTangents[indices[v]].x, mesh->mTangents[indices[v]].y, mesh->mTangents[indices[v]].z};
                tangents[v] = TransfromVec(normal_to_world, tangents[v]);
            }
            else if (std::abs(glm::dot(normals[v], dvec3{0, 1, 0})) != 1.0)
            {
                tangents[v] = glm::normalize(glm::cross(dvec3{0, 1, 0}, normals[v]));
            }
            else
            {
                tangents[v] = {1, 0, 0};
            }
        }

        auto bitangents = std::vector<dvec3>(3);
        for (int v = 0; v < 3; ++v)
        {
            if (mesh->mBitangents)
            {
                bitangents[v] = {mesh->mBitangents[indices[v]].x, mesh->mBitangents[indices[v]].y, mesh->mBitangents[indices[v]].z};
                bitangents[v] = TransfromVec(normal_to_world, bitangents[v]);
            }
            else
            {
                bitangents[v] = glm::normalize(glm::cross(normals[v], tangents[v]));
            }
        }

        shapes[face_index] = new Triangle(id + "_" + std::to_string(face_index), positions, normals, tangents, bitangents,
                                          texcoords, flip_normals);
    }
}

NAMESPACE_END(raytracer)