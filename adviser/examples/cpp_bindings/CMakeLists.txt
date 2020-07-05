cmake_minimum_required (VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)
project(ExampleService)

# cppzmq
find_package(cppzmq)

add_executable(ExampleService example_service.cpp)
target_link_libraries(ExampleService cppzmq)