#include "csrt/renderer/integrators/integrator.hpp"

namespace csrt
{

QUALIFIER_D_H Vec3 Integrator::Shade(const Vec3 &eye, const Vec3 &look_dir,
                                     uint32_t *seed) const
{
    switch (data_.info.type)
    {
    case IntegratorType::kPath:
        return ShadePath(&data_, eye, look_dir, seed);
        break;
    case IntegratorType::kVolPath:
        return ShadeVolPath(&data_, eye, look_dir, seed);
        break;
    }
    return {};
}

} // namespace csrt