#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <pugixml.hpp>

#include "misc.cuh"
#include "../textures/texture.cuh"
#include "../bsdfs/bsdfs.cuh"
#include "../geometry/primitive.cuh"
#include "../geometry/instance.cuh"
#include "../renderer/camera.cuh"
#include "../emitters/emitters.cuh"


struct SceneInfo
{
    Camera camera;
    Sun *sun = nullptr;
    EnvMap *env_map = nullptr;
    std::vector<float> pixel_buffer;
    std::vector<Texture::Info> texture_info_buffer;
    std::vector<Bsdf::Info> bsdf_info_buffer;
    std::vector<Emitter::Info> emitter_info_buffer;
    std::vector<Primitive::Info> primitive_info_buffer;
    std::vector<Instance::Info> instance_info_buffer;
};

class ConfigParser
{
public:
    SceneInfo LoadConfig(bool is_realtime, const std::string &filename);

private:
    SceneInfo LoadDefault();

    void ReadCamera(bool is_realtime, pugi::xml_node sensor_node);
    uint32_t ReadTexture(const pugi::xml_node &texture_node, const float scale,
                         const float defalut_value);
    uint32_t ReadBsdf(pugi::xml_node bsdf_node, std::string id, uint32_t id_opacity,
                      uint32_t id_bumpmap, bool twosided);
    void ReadShape(pugi::xml_node shape_node);
    void ReadEmitter(pugi::xml_node emitter_node);

    void CreateSphere(const Vec3 &center, const float radius, const Mat4 &to_world,
                      const uint32_t id_bsdf);
    void CreateDisk(const Mat4 &to_world, const uint32_t id_bsdf);
    void CreateRectangle(const Mat4 &to_world, const uint32_t id_bsdf);
    void CreateCube(const Mat4 &to_world, uint32_t id_bsdf);
    void CreateMeshes(const std::string &filename, const int index_shape, const Mat4 &to_world,
                      const bool flip_texcoords, const bool face_normals, const uint32_t id_bsdf);

    uint32_t ReadTexture(const pugi::xml_node &parent_node,
                         const std::vector<std::string> &valid_names, const float defalut_value);
    uint32_t ReadBitmap(const std::string &filename, const std::string &id, float gamma, float scale,
                        int *width_max);

    float ReadDielectricIor(const pugi::xml_node &parent_node,
                            const std::vector<std::string> &valid_names, float defalut_value);
    void ReadConductorIor(const pugi::xml_node &parent_node, Vec3 *eta, Vec3 *k);

    std::string config_file_directory_;
    std::unordered_map<std::string, std::string> default_mp_;
    std::unordered_map<std::string, uint32_t> id_texture_mp_;
    std::unordered_map<std::string, uint32_t> id_bsdf_mp_;
    SceneInfo info_;
};