#include "kaggle_integration.h"

SmallPlanetIdSet orbit_wars_comet_planet_ids_from_python(
    py::iterable comet_planet_ids_obj) {
  SmallPlanetIdSet out;
  for (auto item : comet_planet_ids_obj) {
    out.insert(py::cast<int32_t>(item));
  }
  return out;
}

CometPathByPlanetId orbit_wars_comet_paths_from_python(
    py::dict comet_path_by_planet_id_obj) {
  CometPathByPlanetId out;
  for (auto item : comet_path_by_planet_id_obj) {
    const int32_t pid = py::cast<int32_t>(item.first);
    const py::sequence triple = py::reinterpret_borrow<py::sequence>(item.second);
    TORCH_CHECK_DISABLED(py::len(triple) == 3,
                "comet_path_by_planet_id[pid] must be [path_index, path_xy, ships]");
    const int32_t path_index = py::cast<int32_t>(triple[0]);
    const py::sequence path_xy_obj = py::reinterpret_borrow<py::sequence>(triple[1]);
    CometPathInfo cp;
    cp.path_index = static_cast<int32_t>(path_index);
    cp.ships = py::cast<double>(triple[2]);
    for (auto pt_obj : path_xy_obj) {
      const py::sequence pt = py::reinterpret_borrow<py::sequence>(pt_obj);
      TORCH_CHECK_DISABLED(py::len(pt) == 2, "comet path point must be [x,y]");
      const double x = py::cast<double>(pt[0]);
      const double y = py::cast<double>(pt[1]);
      cp.path_xy.push_back(std::make_pair(x, y));
    }
    out.insert(pid, cp);
  }
  return out;
}
