#pragma once

#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//相机
struct Camera
{
    int spp;      //每个像素的样本容量
    int width;    //水平像素数
    int height;   //水平像素数
    double fov_x; //水平视野
    double fov_y; //垂直视野
    dvec3 eye;    //相机位置
    dvec3 front;  //观察方向
    dvec3 up;     //观察坐标系的竖直向上方向
    dvec3 right;  //观察坐标系的正右方向

    Camera()
        : spp(4),
          width(768),
          height(576),
          fov_x(39.5977527),
          fov_y(29.6983145),
          eye(dvec3(0)),
          front({0, 0, 1}),
          up({0, 1, 0}),
          right({-1, 0, 0})
    {
    }
};

NAMESPACE_END(raytracer)
