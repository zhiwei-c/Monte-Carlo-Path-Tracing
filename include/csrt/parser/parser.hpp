#ifndef CSRT__PARSER__PARSER_HPP
#define CSRT__PARSER__PARSER_HPP

#include <string>

#include "../renderer/renderer.hpp"

namespace csrt
{

RendererConfig LoadConfig(const std::string &filename);

} // namespace csrt

#endif