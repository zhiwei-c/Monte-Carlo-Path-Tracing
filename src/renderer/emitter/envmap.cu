#include "csrt/renderer/emitter.cuh"

namespace
{
using namespace csrt;

QUALIFIER_D_H float LinearRgbToLuminance(const Vec3 &rgb)
{
    return 0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
}

} // namespace

namespace csrt
{

QUALIFIER_D_H void Emitter::InitEnvMap(const int width, const int height,
                                       const float normalization, float *data)
{
    data_.envmap.width = width;
    data_.envmap.height = height;
    data_.envmap.normalization = normalization;
    data_.envmap.cdf_cols = data;
    data_.envmap.cdf_rows = data + height + 1;
    data_.envmap.weight_rows = data + (height + 1) + height;
}

void Emitter::CreateEnvMapCdfPdf(const int width, const int height,
                                 const Texture &radiance,
                                 std::vector<float> *cdf_cols,
                                 std::vector<float> *cdf_rows,
                                 std::vector<float> *weight_rows,
                                 float *normalization)
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

    if (!isfinite(sum_row))
    {
        throw std::exception("The environment map contains an invalid floating "
                             "point value (nan/inf).");
    }

    *normalization = 1.0 / (sum_row * (k2Pi * width_inv) * (kPi * height_inv));
}

QUALIFIER_D_H Emitter::SampleRec Emitter::SampleEnvMap(const Vec3 &origin,
                                                       const float xi_0,
                                                       const float xi_1) const
{
    uint32_t row =
        BinarySearch(data_.envmap.height + 1, data_.envmap.cdf_rows, xi_0) - 1;
    float *const &cdf_col =
        data_.envmap.cdf_cols + row * (data_.envmap.width + 1);
    uint32_t col = BinarySearch(data_.envmap.width + 1, cdf_col, xi_1) - 1;

    Vec3 vec_local = SphericalToCartesian(row * kPi / data_.envmap.height,
                                          col * k2Pi / data_.envmap.width, 1),
         vec = TransformVector(data_.envmap.to_world, vec_local);

    return {true, false, kMaxFloat, vec};
}

QUALIFIER_D_H Vec3 Emitter::EvaluateEnvMap(const SampleRec &rec) const
{
    const Vec3 dir = TransformVector(data_.envmap.to_local, rec.wi);
    float phi = 0, theta = 0;
    CartesianToSpherical(-dir, &theta, &phi, nullptr);
    const Vec2 texcoord = {phi * k1Div2Pi, theta * k1DivPi};
    return data_.envmap.radiance->GetColor(texcoord);
}

QUALIFIER_D_H float Emitter::PdfEnvMap(const Vec3 &look_dir) const
{

    const Vec3 dir = TransformVector(data_.envmap.to_local, look_dir);
    float phi = 0, theta = 0;
    CartesianToSpherical(dir, &theta, &phi, nullptr);
    const Vec2 texcoord = {phi * k1Div2Pi, theta * k1DivPi};
    const Vec3 color = data_.envmap.radiance->GetColor(texcoord);

    const float row = fminf(fmaxf(texcoord.u * data_.envmap.height, 0),
                            data_.envmap.height - 1);
    const int row_int = static_cast<int>(row);
    const float t = row - row_int;
    if (t == 0)
    {
        return LinearRgbToLuminance(color) * data_.envmap.weight_rows[row_int] *
               data_.envmap.normalization / fmaxf(fabs(sinf(theta)), 1e-4f);
    }
    else
    {
        return LinearRgbToLuminance(color) *
               Lerp(data_.envmap.weight_rows[row_int],
                    data_.envmap.weight_rows[row_int + 1], t) *
               data_.envmap.normalization / fmaxf(fabs(sinf(theta)), 1e-4f);
    }
}

QUALIFIER_D_H Vec3 Emitter::EvaluateEnvMap(const Vec3 &look_dir) const
{
    const Vec3 dir = TransformVector(data_.envmap.to_local, look_dir);
    float phi = 0, theta = 0;
    CartesianToSpherical(dir, &theta, &phi, nullptr);
    const Vec2 texcoord = {phi * k1Div2Pi, theta * k1DivPi};
    return data_.envmap.radiance->GetColor(texcoord);
}

} // namespace csrt
