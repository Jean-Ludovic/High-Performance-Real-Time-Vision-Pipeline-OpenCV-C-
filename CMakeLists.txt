cmake_minimum_required(VERSION 3.15)
project(piplineRoiCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(piplineRoiCpp main.cpp)
target_link_libraries(piplineRoiCpp PRIVATE ${OpenCV_LIBS})
