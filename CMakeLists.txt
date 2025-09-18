cmake_minimum_required(VERSION 3.24)
project(sons LANGUAGES CXX)

list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/icu4c")

find_package(PkgConfig REQUIRED)
pkg_check_modules(ICU REQUIRED IMPORTED_TARGET icu-uc icu-i18n icu-io)
find_package(benchmark REQUIRED)
find_package(Boost REQUIRED)
# find_package(OpenMP REQUIRED)
set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I$$(brew --prefix libomp)/include")
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I$$(brew --prefix libomp)/include")
set(OpenMP_C_LIB_NAMES "omp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "$$(brew --prefix libomp)/lib/libomp.dylib")
find_package(OpenMP REQUIRED)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(bpe csrc/bpe.cpp)
add_executable(test csrc/test.cpp)
target_link_libraries(test PRIVATE PkgConfig::ICU)
target_link_libraries(bpe PRIVATE benchmark::benchmark PkgConfig::ICU Boost::boost)

target_compile_options(bpe PRIVATE $<$<CONFIG:Release>:-march=native -O3 -ffast-math -fno-exceptions -flto>)
