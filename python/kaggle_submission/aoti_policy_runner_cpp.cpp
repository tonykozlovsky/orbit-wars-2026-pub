#include <torch/extension.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

#include <pybind11/stl.h>

#include <string>
#include <vector>

namespace py = pybind11;

class AOTIPolicyRunnerCpu final {
 public:
  explicit AOTIPolicyRunnerCpu(const std::string& model_so_path)
      : runner_(model_so_path, 1) {}

  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs) {
    std::vector<at::Tensor> outputs;
    {
      py::gil_scoped_release no_gil;
      outputs = runner_.run(inputs);
    }
    return outputs;
  }

 private:
  torch::inductor::AOTIModelContainerRunnerCpu runner_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<AOTIPolicyRunnerCpu>(m, "AOTIPolicyRunnerCpu")
      .def(py::init<const std::string&>(), py::arg("model_so_path"))
      .def("run", &AOTIPolicyRunnerCpu::run, py::arg("inputs"));
}
