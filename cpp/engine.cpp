// C++ backtest engine for the Calendar rebalancing strategy.
// Placeholder for the build setup; the backtest functions are added in the next step.
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_engine, m) {
    m.doc() = "C++ backtest engine for the Calendar rebalancing strategy";
    m.def("version", []() { return "0.1.0"; }, "Engine version, used to check the build.");
}
