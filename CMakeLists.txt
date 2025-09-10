cmake_minimum_required(VERSION 3.24)
project(sons LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(bpe csrc/bpe.cpp)
