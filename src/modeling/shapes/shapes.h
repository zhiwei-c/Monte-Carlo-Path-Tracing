#pragma once

#include <iostream>

#include "triangle.h"
#include "meshes.h"
#include "cube.h"
#include "rectangle.h"
#include "sphere.h"
#include "disk.h"

NAMESPACE_BEGIN(simple_renderer)

inline void DeleteShapePointer(Shape *&shape)
{
    if (!shape)
        return;
    switch (shape->type())
    {
    case ShapeType::kSphere:
        delete ((Sphere *)shape);
        break;
    case ShapeType::kDisk:
        delete ((Disk *)shape);
        break;
    case ShapeType::kMeshes:
        delete ((Meshes *)shape);
        break;
    case ShapeType::kTriangle:
        delete ((Triangle *)shape);
        break;
    case ShapeType::kCube:
        delete ((Cube *)shape);
        break;
    case ShapeType::kRectangle:
        delete ((Rectangle *)shape);
        break;
    default:
        std::cerr << "unknown shape type" << std::endl;
        exit(1);
    }
    shape = nullptr;
}

NAMESPACE_END(simple_renderer)