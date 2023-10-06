#pragma once

#include <vector>
#include <string>

#include <assimp/scene.h>

#include "../tensor/tensor.cuh"
#include "../geometry/primitive.cuh"

class ModelLoader
{
public:
    std::vector<Primitive> Load(const std::string &filename, const Mat4 &to_world, bool flip_texcoords,
                                bool face_normals, uint64_t id_bsdf);
    std::vector<Primitive> Load(const std::string &filename, int index_shape, const Mat4 &to_world,
                                bool flip_texcoords, bool face_normals, uint64_t id_bsdf);

private:
    static std::vector<Primitive> ProcessNode(const aiScene *scene, aiNode *node, bool face_normals,
                                              const Mat4 &to_world, uint64_t id_bsdf);
    static void ProcessMesh(aiMesh *mesh, uint64_t begin, uint64_t end, bool face_normals,
                            const Mat4 &to_world, uint64_t id_bsdf,
                            std::vector<Primitive> &primitive_info_buffer);

    void ParseSerialized(const std::string &filename, int index_shape, std::vector<Vec3> *positions,
                         std::vector<Vec3> *normals, std::vector<Vec2> *texcoords,
                         std::vector<Uvec3> *indices);
};