#pragma once

#include <string>
#include <vector>

#include <tiny_obj_loader.h>
#include "../global.h"

NAMESPACE_BEGIN(raytracer)

class ObjLoader : public tinyobj::ObjReader
{
public:
    ObjLoader() : tinyobj::ObjReader(), valid_(false) {}

    bool ParseFromFile(const std::string &filename, const tinyobj::ObjReaderConfig &config = tinyobj::ObjReaderConfig());

    ///
    /// .obj was loaded or parsed correctly.
    ///
    bool Valid() const { return valid_; }

    const tinyobj::attrib_t &GetAttrib() const { return attrib_; }

    const std::vector<tinyobj::shape_t> &GetShapes() const { return shapes_; }

    const std::vector<tinyobj::material_t> &GetMaterials() const { return materials_; }

    ///
    /// Warning message(may be filled after `Load` or `Parse`)
    ///
    const std::string &Warning() const { return warning_; }

    ///
    /// Error message(filled when `Load` or `Parse` failed)
    ///
    const std::string &Error() const { return error_; }

private:
    bool valid_;

    tinyobj::attrib_t attrib_;
    std::vector<tinyobj::shape_t> shapes_;
    std::vector<tinyobj::material_t> materials_;

    std::string warning_;
    std::string error_;
};

NAMESPACE_END(raytracer)