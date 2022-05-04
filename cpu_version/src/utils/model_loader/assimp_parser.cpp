#include "../model_parser.h"

#include <iostream>

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

NAMESPACE_BEGIN(simple_renderer)

static void ProcessNode(aiNode *node,
                        const aiScene *scene,
                        Mat4 *to_world_p,
                        Mat4 *to_world_n,
                        std::vector<Shape *> &shapes,
                        Material *material,
                        bool flip_normals,
                        bool face_normals);

static void ProcessMesh(aiMesh *mesh,
                        Mat4 *to_world_p,
                        Mat4 *to_world_n,
                        std::vector<Shape *> &shapes,
                        Material *material,
                        bool flip_normals,
                        bool face_normals);

Meshes *ModelParser::Parse(std::string filename,
                         Material *material,
                         std::unique_ptr<Mat4> to_world,
                         bool flip_normals,
                         bool face_normals,
                         bool flip_tex_coords)
{
    std::cout << "[info] begin load " << filename << "\r";

    Mat4 *to_world_p = nullptr, *to_world_n = nullptr;
    if (to_world != nullptr)
    {
        to_world_p = new Mat4(*to_world);
        to_world_n = new Mat4(glm::inverse(glm::transpose(*to_world)));
    }

    Assimp::Importer importer;
    auto option = aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_GenUVCoords;
    if (flip_tex_coords)
        option = option | aiProcess_FlipUVs;
    if (material->NormalPerturbing())
        option = option | aiProcess_CalcTangentSpace;

    auto scene = importer.ReadFile(filename, option);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
    {
        std::cerr << "[error] ASSIMP:: " << importer.GetErrorString() << std::endl;
        exit(1);
    }

    std::vector<Shape *> triangles;
    ProcessNode(scene->mRootNode, scene, to_world_p, to_world_n, triangles, material, flip_normals, face_normals);

    if (to_world)
    {
        delete to_world_p;
        delete to_world_n;
    }
    std::cout << "[info] load " << filename << " finished\n";
    return new Meshes(triangles, material, flip_normals);
}

void ProcessNode(aiNode *node,
                 const aiScene *scene,
                 Mat4 *to_world_p,
                 Mat4 *to_world_n,
                 std::vector<Shape *> &shapes,
                 Material *material,
                 bool flip_normals,
                 bool face_normals)
{
    for (unsigned int i = 0; i < node->mNumMeshes; i++)
    {
        aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
        ProcessMesh(mesh, to_world_p, to_world_n, shapes, material, flip_normals, face_normals);
    }
    for (unsigned int i = 0; i < node->mNumChildren; i++)
    {
        ProcessNode(node->mChildren[i], scene, to_world_p, to_world_n, shapes, material, flip_normals, face_normals);
    }
}

void ProcessMesh(aiMesh *mesh,
                 Mat4 *to_world_p,
                 Mat4 *to_world_n,
                 std::vector<Shape *> &shapes,
                 Material *material,
                 bool flip_normals,
                 bool face_normals)
{
    Vector3 vector;
    Vector2 vec;

    for (unsigned int i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];

        std::vector<unsigned int> indices = {face.mIndices[0], face.mIndices[1], face.mIndices[2]};

        std::vector<Vector3> vertices;
        std::vector<Vector3> normals;
        std::vector<Vector2> texcoords;
        std::vector<Vector3> tangents;
        std::vector<Vector3> bitangents;

        for (int v = 0; v < 3; v++)
        {
            // positions
            vector.x = mesh->mVertices[indices[v]].x;
            vector.y = mesh->mVertices[indices[v]].y;
            vector.z = mesh->mVertices[indices[v]].z;
            if (to_world_p)
                vector = TransfromPt(*to_world_p, vector);
            vertices.push_back(vector);
            // normals
            if (!face_normals)
            {
                vector.x = mesh->mNormals[indices[v]].x;
                vector.y = mesh->mNormals[indices[v]].y;
                vector.z = mesh->mNormals[indices[v]].z;
                if (to_world_n)
                    vector = TransfromDir(*to_world_n, vector);
                normals.push_back(glm::normalize(vector));
            }
            // texture coordinates
            if (mesh->mTextureCoords[0])
            {
                vec.x = mesh->mTextureCoords[0][indices[v]].x;
                vec.y = mesh->mTextureCoords[0][indices[v]].y;
                texcoords.push_back(vec);
                // tangent
                if (mesh->mTangents)
                {
                    vector.x = mesh->mTangents[i].x;
                    vector.y = mesh->mTangents[i].y;
                    vector.z = mesh->mTangents[i].z;
                    tangents.push_back(vector);
                }
                // bitangent
                if (mesh->mBitangents)
                {
                    vector.x = mesh->mBitangents[i].x;
                    vector.y = mesh->mBitangents[i].y;
                    vector.z = mesh->mBitangents[i].z;
                    bitangents.push_back(vector);
                }
            }
        }
        shapes.push_back(new Triangle(vertices, normals, texcoords, tangents, bitangents, material, flip_normals));
    }
}

NAMESPACE_END(simple_renderer)