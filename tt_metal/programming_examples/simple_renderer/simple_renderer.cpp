// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include "renderer_types.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace renderer = tt::tt_metal::programming_examples::simple_renderer;

namespace {

std::string next_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        std::cerr << "Expected argument after " << argv[i] << "\n";
        std::exit(1);
    }
    return argv[++i];
}

void print_help(std::string_view program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --device <id>    Device id to run. Default: 0\n";
    std::cout << "  --width <px>     Render width. Default: 128\n";
    std::cout << "  --height, -H <px> Render height. Default: 128\n";
    std::cout << "  --help, -h       Print this help text\n";
}

uint32_t parse_u32(const std::string& value, std::string_view name) {
    size_t consumed = 0;
    uint64_t parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size() || parsed > UINT32_MAX) {
        std::cerr << "Invalid " << name << ": " << value << "\n";
        std::exit(1);
    }
    return static_cast<uint32_t>(parsed);
}

renderer::RendererConfig make_config(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) {
        std::cerr << "Width and height must be non-zero.\n";
        std::exit(1);
    }
    if (width % renderer::kTileWidth != 0 || height % renderer::kTileHeight != 0) {
        std::cerr << "Width and height must be multiples of " << renderer::kTileWidth << " for this first skeleton.\n";
        std::exit(1);
    }

    return renderer::RendererConfig{
        .width = width,
        .height = height,
        .tiles_x = width / renderer::kTileWidth,
        .tiles_y = height / renderer::kTileHeight,
    };
}

}  // namespace

int main(int argc, char** argv) {
    using namespace tt;
    using namespace tt::tt_metal;

    uint32_t width = 128;
    uint32_t height = 128;
    int device_id = 0;

    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if (arg == "--device" || arg == "-d") {
            device_id = static_cast<int>(parse_u32(next_arg(i, argc, argv), "device id"));
        } else if (arg == "--width" || arg == "-w") {
            width = parse_u32(next_arg(i, argc, argv), "width");
        } else if (arg == "--height" || arg == "-H") {
            height = parse_u32(next_arg(i, argc, argv), "height");
        } else if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_help(argv[0]);
            return 1;
        }
    }

    const renderer::RendererConfig config = make_config(width, height);
    const uint32_t total_tiles = config.tiles_x * config.tiles_y;

    if (std::getenv("TT_METAL_DPRINT_CORES") == nullptr) {
        std::cerr << "WARNING: Set TT_METAL_DPRINT_CORES=0,0 to see the renderer skeleton kernel print.\n";
    }

    std::cout << "Simple renderer skeleton: " << config.width << "x" << config.height << ", " << config.tiles_x
              << "x" << config.tiles_y << " tiles, " << total_tiles << " total tiles.\n";

    constexpr CoreCoord core = {0, 0};
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    KernelHandle renderer_kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "simple_renderer/kernels/dataflow/renderer_hello.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    SetRuntimeArgs(
        program,
        renderer_kernel,
        core,
        {config.width, config.height, config.tiles_x, config.tiles_y, total_tiles});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::cout << "Renderer skeleton program completed.\n";
    mesh_device->close();
    return 0;
}
