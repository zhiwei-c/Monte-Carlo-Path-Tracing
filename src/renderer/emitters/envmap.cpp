#include "csrt/renderer/emitters/envmap.hpp"

#include "csrt/renderer/emitters/emitter.hpp"

namespace
{
using namespace csrt;

QUALIFIER_D_H float LinearRgbToLuminance(const Vec3 &rgb)
{
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

} // namespace

namespace csrt
{


void CreateEnvMapCdfPdf(const int width, const int height,
                        const Texture &radiance, std::vector<float> *cdf_cols,
                        std::vector<float> *cdf_rows,
                        std::vector<float> *weight_rows, float *normalization)
{
    const float width_inv = 1.0f / width;
    const float height_inv = 1.0f / height;

    *cdf_rows = std::vector<float>(height + 1);
    *weight_rows = std::vector<float>(height);
    *cdf_cols = std::vector<float>((width + 1) * height);

    float sum_row = 0.0f;
    (*cdf_rows)[0] = 0;
    for (int y = 0; y < height; ++y)
    {
        float sum_col = 0.0f;
        (*cdf_cols)[0 + 0] = 0;
        for (int x = 0; x < width; ++x)
        {
            const Vec3 rgb = radiance.GetColor({x * width_inv, y * height_inv});
            sum_col += LinearRgbToLuminance(rgb);
            (*cdf_cols)[y * (width + 1) + (x + 1)] = sum_col;
        }

        (*cdf_cols)[y * (width + 1) + width] = 1.0f;
        const float normalization_col = 1.0f / sum_col;
        for (int x = 1; x < width; ++x)
            (*cdf_cols)[y * (width + 1) + width - x] *= normalization_col;

        const float weight = sinf((y + 0.5f) * kPi / height);
        (*weight_rows)[y] = weight;
        sum_row += sum_col * weight;
        (*cdf_rows)[y + 1] = sum_row;
    }

    (*cdf_rows)[height] = 1.0f;
    const float normalization_row = 1.0f / sum_row;
    for (int y = 1; y < height; ++y)
        (*cdf_rows)[height - y] *= normalization_row;

    if (!std::isfinite(sum_row))
    {
        throw MyException("The environment map contains an invalid floating "
                             "point value (nan/inf).");
    }

    *normalization = 1.0 / (sum_row * (k2Pi * width_inv) * (kPi * height_inv));
}

QUALIFIER_D_H void SampleEnvMap(const EnvMapData &data, const Vec3 &origin,
                                const float xi_0, const float xi_1,
                                EmitterSampleRec *rec)
{
    uint32_t row = BinarySearch(data.height + 1, data.cdf_rows, xi_0) - 1;
    float *const &cdf_col = data.cdf_cols + row * (data.width + 1);
    uint32_t col = BinarySearch(data.width + 1, cdf_col, xi_1) - 1;

    Vec3 vec_local = SphericalToCartesian(row * kPi / data.height,
                                          col * k2Pi / data.width, 1),
         vec = TransformVector(data.to_world, vec_local);

    *rec = {
        true,      // valid
        false,     // harsh
        kMaxFloat, // distance
        vec        // wi
    };
}

QUALIFIER_D_H Vec3 EvaluateEnvMap(const EnvMapData &data,
                                  const EmitterSampleRec *rec)
{
    const Vec3 dir = TransformVector(data.to_local, rec->wi);
    float phi = 0, theta = 0;
    CartesianToSpherical(-dir, &theta, &phi, nullptr);
    const Vec2 texcoord = {phi * k1Div2Pi, theta * k1DivPi};
    return data.radiance->GetColor(texcoord);
}

QUALIFIER_D_H Vec3 EvaluateEnvMap(const EnvMapData &data, const Vec3 &look_dir)
{
    const Vec3 dir = TransformVector(data.to_local, look_dir);
    float phi = 0, theta = 0;
    CartesianToSpherical(dir, &theta, &phi, nullptr);
    const Vec2 texcoord = {phi * k1Div2Pi, theta * k1DivPi};
    return data.radiance->GetColor(texcoord);
}

QUALIFIER_D_H float PdfEnvMap(const EnvMapData &data, const Vec3 &look_dir)
{
    const Vec3 dir = TransformVector(data.to_local, look_dir);
    float phi = 0, theta = 0;
    CartesianToSpherical(dir, &theta, &phi, nullptr);
    const Vec2 texcoord = {phi * k1Div2Pi, theta * k1DivPi};
    const Vec3 color = data.radiance->GetColor(texcoord);

    const float row =
        fminf(fmaxf(texcoord.u * data.height, 0), data.height - 1);
    const int row_int = static_cast<int>(row);
    const float t = row - row_int;
    if (t == 0)
    {
        return LinearRgbToLuminance(color) * data.weight_rows[row_int] *
               data.normalization / fmaxf(fabs(sinf(theta)), 1e-4f);
    }
    else
    {
        return LinearRgbToLuminance(color) *
               Lerp(data.weight_rows[row_int], data.weight_rows[row_int + 1],
                    t) *
               data.normalization / fmaxf(fabs(sinf(theta)), 1e-4f);
    }
}

} // namespace csrt
