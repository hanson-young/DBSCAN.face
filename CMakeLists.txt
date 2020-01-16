cmake_minimum_required(VERSION 3.14)
project(DBSCAN_face)

set(CMAKE_CXX_STANDARD 14)

add_executable(DBSCAN_face main.cpp DBSCAN.h DBSCAN.cpp)