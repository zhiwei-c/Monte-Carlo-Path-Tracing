#pragma once

#include "bitmap.h"
#include "checkerboard.h"
#include "grid_texture.h"

#include <iostream>

NAMESPACE_BEGIN(simple_renderer)

inline void DeleteTexturePointer(Texture *&texture)
{
    if (!texture)
        return;
    switch (texture->type())
    {
    case TextureType::kBitmap:
        delete ((Bitmap *)texture);
        break;
    case TextureType::kCheckerboard:
        delete ((Checkerboard *)texture);
        break;
    case TextureType::kGridTexture:
        delete ((GridTexture *)texture);
        break;
    default:
        std::cerr << "unknown texture type" << std::endl;
        exit(1);
    }
    texture = nullptr;
}

NAMESPACE_END(simple_renderer)