#pragma once

#include <algorithm>
#include <omp.h>

#include "../utils/timer.h"
#include "integrator.h"

NAMESPACE_BEGIN(raytracer)

//相片
struct Film
{
    int height;         //高度
    int width;          //宽度
    Float gamma;        //伽马值
    std::string format; //格式（可选：png，jpg，hdr，exr）
};

//相机类
class Camera
{
public:
    ///\brief 相机
    ///\param film 相片（生成图像）信息
    ///\param eye_pos 相机位置
    ///\param look_at 观察的目标点
    ///\param up 观察坐标系的竖直向上方向
    ///\param fov_height 垂直方向的视角（弧度制）
    ///\param spp 每个像素点采样的次数
    Camera(Film film, Vector3 eye_pos, Vector3 look_at, Vector3 up, Float fov_height, int spp)
        : film_(film), eye_pos_(eye_pos), fov_height_(fov_height), spp_(spp)
    {
        Float fov_width = fov_height_ * film_.width / film_.height;
        look_dir_ = glm::normalize(look_at - eye_pos_);
        Vector3 right_dir_ = glm::normalize(glm::cross(look_dir_, up));
        up_ = glm::normalize(glm::cross(right_dir_, look_dir_));
        view_dx = right_dir_ * static_cast<Float>(glm::tan(glm::radians(0.5 * fov_width)));
        view_dy = up_ * static_cast<Float>(glm::tan(glm::radians(0.5 * fov_height_)));

        spp_x_ = static_cast<int>(std::sqrt(spp));
        spp_y_ = static_cast<int>(spp / spp_x_);
        spp_r_ = spp - spp_x_ * spp_y_;
        dxx_ = static_cast<Float>(1) / spp_x_;
        dyy_ = static_cast<Float>(1) / spp_y_;
    }

    ///\brief 在一个像素对应的区域内生成一束光线的方向
    ///\param i 像素在 width 方向的相对位置
    ///\param j 像素在 height 方向的相对位置
    ///\return 生成的一束光线方向
    std::vector<Vector3> GetDirections(int i, int j) const
    {
        auto dirs = std::vector<Vector3>();
        for (int k_x = 0; k_x < spp_x_; k_x++)
        {
            for (int k_y = 0; k_y < spp_y_; k_y++)
            {
                Float dx = (k_x + UniformFloat()) * dxx_,
                      dy = (k_y + UniformFloat()) * dyy_,
                      x = 2.0 * (i + dx) / film_.width - 1.0,
                      y = 1.0 - 2.0 * (j + dy) / film_.height;
                dirs.push_back(glm::normalize(look_dir_ + x * view_dx + y * view_dy));
            }
        }
        for (int i = 0; i < spp_r_; i++)
        {
            Float x = 2.0 * (i + UniformFloat()) / film_.width - 1.0,
                  y = 1.0 - 2.0 * (j + UniformFloat()) / film_.height;
            dirs.push_back(glm::normalize(look_dir_ + x * view_dx + y * view_dy));
        }
        return dirs;
    }

    ///\brief 根据相机信息生成一张给定场景的图像
    ///\param integrator 全局光照模型
    Bitmap *Shoot(Integrator *integrator)
    {
        Timer timer;
        std::cout << "[info] Begin render......\t\t\t\r";

        // std::vector<Vector3> look_dirs = GetDirections(119, 37);
        // for (auto look_dir : look_dirs)
        //     integrator->Shade(eye_pos_, look_dir);

        auto frame = new Bitmap(film_.width, film_.height, 3, film_.gamma);

        auto pixels = std::vector<std::pair<int, int>>();
        for (int j = 0; j < film_.height; j++)
        {
            for (int i = 0; i < film_.width; i++)
                pixels.push_back({i, j});
        }
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(pixels.begin(), pixels.end(), g);

        int count = 0;
        Float total_inv = 1.0 / (film_.height * film_.width),
              spp_inv = 1.0 / spp_;
#pragma omp parallel for shared(count)
        for (int line = 0; line < film_.height; line++)
        {
            auto beginning = pixels.begin() + line * film_.width;
            auto ending = pixels.begin() + (line + 1) * film_.width;
            auto pixels_now = std::vector<std::pair<int, int>>(beginning, ending);
            for (auto pixel : pixels_now)
            {
                auto color = Spectrum(0);
                auto [i, j] = pixel;
                std::vector<Vector3> look_dirs = GetDirections(i, j);
                for (auto look_dir : look_dirs)
                    color += integrator->Shade(eye_pos_, look_dir) * spp_inv;
                frame->SetColor(i, j, color);
#pragma omp critical
                {
                    count++;
                    Float progress = count * total_inv;
                    timer.PrintProgress(progress);
                }
            }
        }
        timer.PrintTimePassed();
        std::cout << std::endl;
        return frame;
    }

    std::string Format() const { return film_.format; }

private:
    Vector3 eye_pos_;  //相机位置
    Vector3 look_dir_; //观察方向
    Vector3 up_;       //观察坐标系的竖直向上方向
    Float fov_height_; //垂直方向的视角（弧度制）
    Film film_;        //相片（生成图像）信息
    int spp_;          //对每个像素进行采样的次数

    int spp_x_;
    int spp_y_;
    int spp_r_;
    Float dxx_;
    Float dyy_;
    Vector3 view_dx; //当光线于视锥体底的投影点在相机坐标系向x方向移动单位长度时，在世界坐标系下移动的长度
    Vector3 view_dy; //当光线于视锥体底的投影点在相机坐标系向x方向移动单位长度时，在世界坐标系下移动的长度
};

NAMESPACE_END(raytracer)