// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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
    std::cout << "  --device <id>     Device id to run. Default: 0\n";
    std::cout << "  --width <px>      Render width. Default: 128\n";
    std::cout << "  --height, -H <px> Render height. Default: 128\n";
    std::cout << "  --dump-bins       Print every non-empty host tile bin\n";
    std::cout << "  --help, -h        Print this help text\n";
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

std::vector<renderer::ScreenTriangle> make_demo_scene(const renderer::RendererConfig& config) {
    const float w = static_cast<float>(config.width);
    const float h = static_cast<float>(config.height);

    return {
        renderer::ScreenTriangle{
            .v0 = {.x = 0.10F * w, .y = 0.15F * h, .z = 0.20F},
            .v1 = {.x = 0.75F * w, .y = 0.20F * h, .z = 0.20F},
            .v2 = {.x = 0.35F * w, .y = 0.85F * h, .z = 0.20F},
            .rgba = 0xFF3366CC,
        },
        renderer::ScreenTriangle{
            .v0 = {.x = 0.55F * w, .y = 0.30F * h, .z = 0.40F},
            .v1 = {.x = 0.95F * w, .y = 0.55F * h, .z = 0.40F},
            .v2 = {.x = 0.70F * w, .y = 0.95F * h, .z = 0.40F},
            .rgba = 0xFF33CC66,
        },
        renderer::ScreenTriangle{
            .v0 = {.x = 0.05F * w, .y = 0.70F * h, .z = 0.10F},
            .v1 = {.x = 0.25F * w, .y = 0.98F * h, .z = 0.10F},
            .v2 = {.x = 0.02F * w, .y = 0.95F * h, .z = 0.10F},
            .rgba = 0xFFCC6633,
        },
    };
}

float signed_area_2x(const renderer::ScreenTriangle& triangle) {
    const float ax = triangle.v1.x - triangle.v0.x;
    const float ay = triangle.v1.y - triangle.v0.y;
    const float bx = triangle.v2.x - triangle.v0.x;
    const float by = triangle.v2.y - triangle.v0.y;
    return ax * by - ay * bx;
}

struct TileBounds {
    uint32_t min_x = 0;
    uint32_t min_y = 0;
    uint32_t max_x = 0;
    uint32_t max_y = 0;
};

std::optional<TileBounds> compute_tile_bounds(
    const renderer::RendererConfig& config, const renderer::ScreenTriangle& triangle) {
    if (std::fabs(signed_area_2x(triangle)) < 1.0e-5F) {
        return std::nullopt;
    }

    const std::array<float, 3> xs = {triangle.v0.x, triangle.v1.x, triangle.v2.x};
    const std::array<float, 3> ys = {triangle.v0.y, triangle.v1.y, triangle.v2.y};
    const float min_x = *std::min_element(xs.begin(), xs.end());
    const float max_x = *std::max_element(xs.begin(), xs.end());
    const float min_y = *std::min_element(ys.begin(), ys.end());
    const float max_y = *std::max_element(ys.begin(), ys.end());

    if (max_x < 0.0F || max_y < 0.0F || min_x >= static_cast<float>(config.width) ||
        min_y >= static_cast<float>(config.height)) {
        return std::nullopt;
    }

    const float clipped_min_x = std::max(0.0F, min_x);
    const float clipped_min_y = std::max(0.0F, min_y);
    const float clipped_max_x = std::min(static_cast<float>(config.width - 1), max_x);
    const float clipped_max_y = std::min(static_cast<float>(config.height - 1), max_y);

    return TileBounds{
        .min_x = static_cast<uint32_t>(std::floor(clipped_min_x / renderer::kTileWidth)),
        .min_y = static_cast<uint32_t>(std::floor(clipped_min_y / renderer::kTileHeight)),
        .max_x = static_cast<uint32_t>(std::floor(clipped_max_x / renderer::kTileWidth)),
        .max_y = static_cast<uint32_t>(std::floor(clipped_max_y / renderer::kTileHeight)),
    };
}

renderer::TileBinningResult bin_triangles_to_tiles(
    const renderer::RendererConfig& config, const std::vector<renderer::ScreenTriangle>& triangles) {
    const uint32_t total_tiles = config.tiles_x * config.tiles_y;
    std::vector<std::vector<uint32_t>> per_tile_triangle_indices(total_tiles);

    for (uint32_t triangle_id = 0; triangle_id < triangles.size(); triangle_id++) {
        const std::optional<TileBounds> bounds = compute_tile_bounds(config, triangles[triangle_id]);
        if (!bounds.has_value()) {
            continue;
        }

        for (uint32_t tile_y = bounds->min_y; tile_y <= bounds->max_y; tile_y++) {
            for (uint32_t tile_x = bounds->min_x; tile_x <= bounds->max_x; tile_x++) {
                const uint32_t tile_id = tile_y * config.tiles_x + tile_x;
                per_tile_triangle_indices[tile_id].push_back(triangle_id);
            }
        }
    }

    renderer::TileBinningResult result;
    result.tile_bins.resize(total_tiles);
    for (uint32_t tile_id = 0; tile_id < total_tiles; tile_id++) {
        const std::vector<uint32_t>& tile_triangles = per_tile_triangle_indices[tile_id];
        result.tile_bins[tile_id].offset = static_cast<uint32_t>(result.triangle_indices.size());
        result.tile_bins[tile_id].count = static_cast<uint32_t>(tile_triangles.size());
        result.triangle_indices.insert(result.triangle_indices.end(), tile_triangles.begin(), tile_triangles.end());
    }

    return result;
}

void print_binning_summary(
    const renderer::RendererConfig& config,
    const std::vector<renderer::ScreenTriangle>& triangles,
    const renderer::TileBinningResult& binning,
    bool dump_bins) {
    uint32_t non_empty_tiles = 0;
    uint32_t max_triangles_per_tile = 0;
    for (const renderer::TileBin& bin : binning.tile_bins) {
        if (bin.count > 0) {
            non_empty_tiles++;
            max_triangles_per_tile = std::max(max_triangles_per_tile, bin.count);
        }
    }

    std::cout << "Host tile binning: " << triangles.size() << " triangles, " << binning.tile_bins.size()
              << " tile bins, " << binning.triangle_indices.size() << " triangle references, " << non_empty_tiles
              << " non-empty tiles, max " << max_triangles_per_tile << " triangles/tile.\n";

    const uint32_t max_rows = dump_bins ? static_cast<uint32_t>(binning.tile_bins.size()) : 8;
    uint32_t printed_rows = 0;
    for (uint32_t tile_id = 0; tile_id < binning.tile_bins.size(); tile_id++) {
        const renderer::TileBin& bin = binning.tile_bins[tile_id];
        if (bin.count == 0) {
            continue;
        }
        if (printed_rows >= max_rows) {
            break;
        }

        const uint32_t tile_x = tile_id % config.tiles_x;
        const uint32_t tile_y = tile_id / config.tiles_x;
        std::cout << "  tile (" << tile_x << "," << tile_y << "): offset=" << bin.offset << ", count=" << bin.count
                  << ", triangles=[";
        for (uint32_t i = 0; i < bin.count; i++) {
            if (i != 0) {
                std::cout << ",";
            }
            std::cout << binning.triangle_indices[bin.offset + i];
        }
        std::cout << "]\n";
        printed_rows++;
    }

    if (!dump_bins && printed_rows < non_empty_tiles) {
        std::cout << "  ... pass --dump-bins to print all non-empty tile bins.\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    using namespace tt;
    using namespace tt::tt_metal;

    uint32_t width = 128;
    uint32_t height = 128;
    int device_id = 0;
    bool dump_bins = false;

    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if (arg == "--device" || arg == "-d") {
            device_id = static_cast<int>(parse_u32(next_arg(i, argc, argv), "device id"));
        } else if (arg == "--width" || arg == "-w") {
            width = parse_u32(next_arg(i, argc, argv), "width");
        } else if (arg == "--height" || arg == "-H") {
            height = parse_u32(next_arg(i, argc, argv), "height");
        } else if (arg == "--dump-bins") {
            dump_bins = true;
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
    const std::vector<renderer::ScreenTriangle> triangles = make_demo_scene(config);
    const renderer::TileBinningResult binning = bin_triangles_to_tiles(config, triangles);

    if (std::getenv("TT_METAL_DPRINT_CORES") == nullptr) {
        std::cerr << "WARNING: Set TT_METAL_DPRINT_CORES=0,0 to see the renderer skeleton kernel print.\n";
    }

    std::cout << "Simple renderer skeleton: " << config.width << "x" << config.height << ", " << config.tiles_x
              << "x" << config.tiles_y << " tiles, " << total_tiles << " total tiles.\n";
    print_binning_summary(config, triangles, binning, dump_bins);

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
        {config.width,
         config.height,
         config.tiles_x,
         config.tiles_y,
         total_tiles,
         static_cast<uint32_t>(triangles.size()),
         static_cast<uint32_t>(binning.triangle_indices.size())});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::cout << "Renderer skeleton program completed.\n";
    mesh_device->close();
    return 0;
}
