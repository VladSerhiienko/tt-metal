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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "renderer_types.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace renderer = tt::tt_metal::programming_examples::simple_renderer;

namespace {

constexpr uint32_t kTransformScratchBytes = 32 * 1024;

struct VertexSoA {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
};

struct ScreenVertex {
    float x = 0.0F;
    float y = 0.0F;
    float depth = 0.0F;
};

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
    std::cout << "  --device, -d <id>    Device id to run. Default: 0\n";
    std::cout << "  --vertices, -n <n>   Vertex count to transform. Default: 64\n";
    std::cout << "  --width, -w <px>     Viewport width. Default: 128\n";
    std::cout << "  --height, -H <px>    Viewport height. Default: 128\n";
    std::cout << "  --help, -h           Print this help text\n";
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

renderer::VertexTransformConfig make_config(uint32_t vertex_count, uint32_t width, uint32_t height) {
    if (vertex_count == 0) {
        std::cerr << "Vertex count must be non-zero.\n";
        std::exit(1);
    }
    if (width == 0 || height == 0) {
        std::cerr << "Viewport width and height must be non-zero.\n";
        std::exit(1);
    }

    return renderer::VertexTransformConfig{
        .vertex_count = vertex_count,
        .vertex_tiles = (vertex_count + renderer::kElementsPerTile - 1) / renderer::kElementsPerTile,
        .viewport_width = width,
        .viewport_height = height,
    };
}

renderer::Mat4 make_identity_object_to_clip_matrix() {
    return renderer::Mat4{
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
    };
}

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

VertexSoA make_demo_vertices(const renderer::VertexTransformConfig& config) {
    const uint32_t padded_vertex_count = config.vertex_tiles * renderer::kElementsPerTile;
    VertexSoA vertices{
        .x = std::vector<float>(padded_vertex_count, 0.0F),
        .y = std::vector<float>(padded_vertex_count, 0.0F),
        .z = std::vector<float>(padded_vertex_count, 0.0F),
    };

    constexpr float pi = 3.14159265358979323846F;
    const float denominator = static_cast<float>(std::max<uint32_t>(1, config.vertex_count - 1));
    for (uint32_t i = 0; i < config.vertex_count; i++) {
        const float t = static_cast<float>(i) / denominator;
        vertices.x[i] = -0.8F + 1.6F * t;
        vertices.y[i] = 0.45F * std::sin(2.0F * pi * t);
        vertices.z[i] = -0.5F + t;
    }

    return vertices;
}

ScreenVertex transform_vertex_on_host(
    float x,
    float y,
    float z,
    const renderer::Mat4& object_to_clip,
    uint32_t width,
    uint32_t height) {
    const float clip_x = object_to_clip[0] * x + object_to_clip[1] * y + object_to_clip[2] * z + object_to_clip[3];
    const float clip_y = object_to_clip[4] * x + object_to_clip[5] * y + object_to_clip[6] * z + object_to_clip[7];
    const float clip_z = object_to_clip[8] * x + object_to_clip[9] * y + object_to_clip[10] * z + object_to_clip[11];
    const float clip_w = object_to_clip[12] * x + object_to_clip[13] * y + object_to_clip[14] * z + object_to_clip[15];
    const float inv_w = 1.0F / clip_w;
    const float ndc_x = clip_x * inv_w;
    const float ndc_y = clip_y * inv_w;
    const float ndc_z = clip_z * inv_w;

    return ScreenVertex{
        .x = (ndc_x * 0.5F + 0.5F) * static_cast<float>(width),
        .y = (0.5F - ndc_y * 0.5F) * static_cast<float>(height),
        .depth = ndc_z * 0.5F + 0.5F,
    };
}

void validate_results(
    const renderer::VertexTransformConfig& config,
    const renderer::Mat4& object_to_clip,
    const VertexSoA& vertices,
    const std::vector<float>& screen_x,
    const std::vector<float>& screen_y,
    const std::vector<float>& depth) {
    constexpr float tolerance = 1.0e-4F;
    const uint32_t vertices_to_check = std::min<uint32_t>(config.vertex_count, 16);
    for (uint32_t i = 0; i < vertices_to_check; i++) {
        const ScreenVertex expected = transform_vertex_on_host(
            vertices.x[i], vertices.y[i], vertices.z[i], object_to_clip, config.viewport_width, config.viewport_height);
        const bool matches =
            std::fabs(screen_x[i] - expected.x) <= tolerance && std::fabs(screen_y[i] - expected.y) <= tolerance &&
            std::fabs(depth[i] - expected.depth) <= tolerance;
        if (!matches) {
            std::cerr << "Vertex transform mismatch at vertex " << i << ": expected (" << expected.x << ", "
                      << expected.y << ", " << expected.depth << "), got (" << screen_x[i] << ", " << screen_y[i]
                      << ", " << depth[i] << ")\n";
            std::exit(1);
        }
    }
}

void print_sample_results(
    const renderer::VertexTransformConfig& config,
    const std::vector<float>& screen_x,
    const std::vector<float>& screen_y,
    const std::vector<float>& depth) {
    const uint32_t vertices_to_print = std::min<uint32_t>(config.vertex_count, 4);
    std::cout << std::fixed << std::setprecision(3);
    for (uint32_t i = 0; i < vertices_to_print; i++) {
        std::cout << "  vertex " << i << " -> screen=(" << screen_x[i] << ", " << screen_y[i]
                  << "), depth=" << depth[i] << "\n";
    }
    std::cout.unsetf(std::ios::floatfield);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace tt;
    using namespace tt::tt_metal;

    uint32_t vertex_count = 64;
    uint32_t width = 128;
    uint32_t height = 128;
    int device_id = 0;

    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if (arg == "--device" || arg == "-d") {
            device_id = static_cast<int>(parse_u32(next_arg(i, argc, argv), "device id"));
        } else if (arg == "--vertices" || arg == "-n") {
            vertex_count = parse_u32(next_arg(i, argc, argv), "vertex count");
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

    const renderer::VertexTransformConfig config = make_config(vertex_count, width, height);
    const renderer::Mat4 object_to_clip = make_identity_object_to_clip_matrix();
    const VertexSoA vertices = make_demo_vertices(config);

    if (std::getenv("TT_METAL_DPRINT_CORES") == nullptr) {
        std::cerr << "WARNING: Set TT_METAL_DPRINT_CORES=0,0 to see the vertex transform kernel print.\n";
    }

    std::cout << "Simple renderer vertex transform: " << config.vertex_count << " vertices, "
              << config.vertex_tiles << " coordinate tile(s), viewport " << config.viewport_width << "x"
              << config.viewport_height << ".\n";

    constexpr CoreCoord core = {0, 0};
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    const uint32_t coordinate_buffer_size_bytes = config.vertex_tiles * renderer::kCoordinateTileBytes;
    distributed::DeviceLocalBufferConfig coordinate_dram_config{
        .page_size = renderer::kCoordinateTileBytes,
        .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig scratch_l1_config{
        .page_size = kTransformScratchBytes,
        .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig coordinate_buffer_config{.size = coordinate_buffer_size_bytes};
    distributed::ReplicatedBufferConfig scratch_buffer_config{.size = kTransformScratchBytes};

    auto input_x_buffer =
        distributed::MeshBuffer::create(coordinate_buffer_config, coordinate_dram_config, mesh_device.get());
    auto input_y_buffer =
        distributed::MeshBuffer::create(coordinate_buffer_config, coordinate_dram_config, mesh_device.get());
    auto input_z_buffer =
        distributed::MeshBuffer::create(coordinate_buffer_config, coordinate_dram_config, mesh_device.get());
    auto screen_x_buffer =
        distributed::MeshBuffer::create(coordinate_buffer_config, coordinate_dram_config, mesh_device.get());
    auto screen_y_buffer =
        distributed::MeshBuffer::create(coordinate_buffer_config, coordinate_dram_config, mesh_device.get());
    auto depth_buffer =
        distributed::MeshBuffer::create(coordinate_buffer_config, coordinate_dram_config, mesh_device.get());
    auto scratch_buffer = distributed::MeshBuffer::create(scratch_buffer_config, scratch_l1_config, mesh_device.get());

    distributed::EnqueueWriteMeshBuffer(cq, input_x_buffer, vertices.x, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, input_y_buffer, vertices.y, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, input_z_buffer, vertices.z, /*blocking=*/false);
    std::cout << "Uploaded vertex SoA tiles: x/y/z=" << coordinate_buffer_size_bytes
              << " bytes each, outputs=" << coordinate_buffer_size_bytes << " bytes each, l1_scratch="
              << kTransformScratchBytes << " bytes.\n";

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    KernelHandle transform_kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "simple_renderer/kernels/dataflow/renderer_hello.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<uint32_t> runtime_args = {
        config.vertex_count,
        config.vertex_tiles,
        config.viewport_width,
        config.viewport_height,
        input_x_buffer->address(),
        input_y_buffer->address(),
        input_z_buffer->address(),
        screen_x_buffer->address(),
        screen_y_buffer->address(),
        depth_buffer->address(),
        scratch_buffer->address(),
    };
    for (float value : object_to_clip) {
        runtime_args.push_back(float_bits(value));
    }

    SetRuntimeArgs(program, transform_kernel, core, runtime_args);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<float> screen_x(config.vertex_tiles * renderer::kElementsPerTile);
    std::vector<float> screen_y(config.vertex_tiles * renderer::kElementsPerTile);
    std::vector<float> depth(config.vertex_tiles * renderer::kElementsPerTile);
    distributed::EnqueueReadMeshBuffer(cq, screen_x, screen_x_buffer, /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(cq, screen_y, screen_y_buffer, /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(cq, depth, depth_buffer, /*blocking=*/true);

    validate_results(config, object_to_clip, vertices, screen_x, screen_y, depth);
    std::cout << "Device vertex transform matched host reference for the first "
              << std::min<uint32_t>(config.vertex_count, 16) << " vertex/vertices.\n";
    print_sample_results(config, screen_x, screen_y, depth);

    mesh_device->close();
    return 0;
}
