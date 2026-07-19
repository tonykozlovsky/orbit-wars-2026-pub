#pragma once

#include "common.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

SmallPlanetIdSet orbit_wars_comet_planet_ids_from_python(
    py::iterable comet_planet_ids_obj);

CometPathByPlanetId orbit_wars_comet_paths_from_python(
    py::dict comet_path_by_planet_id_obj);
