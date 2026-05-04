cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_simple_renderer)

add_executable(metal_example_simple_renderer)
target_sources(metal_example_simple_renderer PRIVATE simple_renderer.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_simple_renderer PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_simple_renderer TT::CommonPCH)
endif()
