#pragma once

#if _WIN32 || _WIN64
#define NOMINMAX
#include <WinSock2.h>
#endif

#include <arrayfire.h>

#include <cereal/archives/json.hpp>
#include <cereal/types/memory.hpp>

#include "wb.hpp"
#include "feed.hpp"
#include "techniques/fully_connected.hpp"
#include "techniques/logistic.hpp"
#include "techniques/squared_loss.hpp"
#include "techniques/sgd.hpp"

#include "serialization/arrayfire.hpp"
#include "serialization/wb.hpp"
#include "serialization/fully_connected.hpp"
#include "serialization/logistic.hpp"
#include "serialization/squared_loss.hpp"
