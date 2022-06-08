#pragma once

#include "../core/shape.h"

void LoadCube(ShapeInfo *shape_info, bool bump_mapping, std::vector<Vertex> &vertex_list, std::vector<uvec3> &idx_list);
void LoadDisk(ShapeInfo *shape_info, bool bump_mapping, std::vector<Vertex> &vertex_list, std::vector<uvec3> &idx_list);
void LoadMeshes(ShapeInfo *shape_info, bool bump_mapping, std::vector<Vertex> &vertex_list, std::vector<uvec3> &idx_list);
void LoadRectangle(ShapeInfo *shape_info, bool bump_mapping, std::vector<Vertex> &vertex_list, std::vector<uvec3> &idx_list);
void LoadSphere(ShapeInfo *shape_info, bool bump_mapping, std::vector<Vertex> &vertex_list, std::vector<uvec3> &idx_list);