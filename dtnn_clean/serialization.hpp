#pragma once

#if _WIN32 || _WIN64
#ifndef NOMINMAX
#define NOMINMAX
#undef min
#undef max
#endif
//#include <WinSock2.h>
#endif

#include <cereal/archives/json.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include "arrayfire_serialization.hpp"