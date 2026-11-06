# Builds the C++ backtest engine as a Python extension module:
# rebalancing_frontrun/_engine (compiled by scikit-build-core during `uv sync`).
cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_engine cpp/engine.cpp)
target_compile_features(_engine PRIVATE cxx_std_17)

install(TARGETS _engine DESTINATION rebalancing_frontrun)
