// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

namespace {

constexpr uint32_t kElementsPerTile = 32 * 32;
constexpr uint32_t kCoordinateTileBytes = kElementsPerTile * sizeof(float);
constexpr uint32_t kNocAlignment = 64;
constexpr uint32_t kScratchSlotStride = kCoordinateTileBytes + kNocAlignment;

float uint32_to_float(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } value;
    value.u = bits;
    return value.f;
}

uint32_t aligned_l1_slot_addr(uint32_t aligned_scratch_l1_addr, uint32_t slot, uint64_t noc_addr) {
    return aligned_scratch_l1_addr + slot * kScratchSlotStride + static_cast<uint32_t>(noc_addr & 0x3F);
}

}  // namespace

void kernel_main() {
    const uint32_t vertex_count = get_arg_val<uint32_t>(0);
    const uint32_t vertex_tiles = get_arg_val<uint32_t>(1);
    const uint32_t viewport_width = get_arg_val<uint32_t>(2);
    const uint32_t viewport_height = get_arg_val<uint32_t>(3);
    const uint32_t input_x_addr = get_arg_val<uint32_t>(4);
    const uint32_t input_y_addr = get_arg_val<uint32_t>(5);
    const uint32_t input_z_addr = get_arg_val<uint32_t>(6);
    const uint32_t screen_x_addr = get_arg_val<uint32_t>(7);
    const uint32_t screen_y_addr = get_arg_val<uint32_t>(8);
    const uint32_t depth_addr = get_arg_val<uint32_t>(9);
    const uint32_t scratch_l1_addr = get_arg_val<uint32_t>(10);

    float object_to_clip[16];
    for (uint32_t i = 0; i < 16; i++) {
        object_to_clip[i] = uint32_to_float(get_arg_val<uint32_t>(11 + i));
    }

    InterleavedAddrGen<true> input_x = {.bank_base_address = input_x_addr, .page_size = kCoordinateTileBytes};
    InterleavedAddrGen<true> input_y = {.bank_base_address = input_y_addr, .page_size = kCoordinateTileBytes};
    InterleavedAddrGen<true> input_z = {.bank_base_address = input_z_addr, .page_size = kCoordinateTileBytes};
    InterleavedAddrGen<true> screen_x = {.bank_base_address = screen_x_addr, .page_size = kCoordinateTileBytes};
    InterleavedAddrGen<true> screen_y = {.bank_base_address = screen_y_addr, .page_size = kCoordinateTileBytes};
    InterleavedAddrGen<true> depth = {.bank_base_address = depth_addr, .page_size = kCoordinateTileBytes};

    const uint32_t aligned_scratch_l1_addr = (scratch_l1_addr + kNocAlignment - 1) & ~(kNocAlignment - 1);
    float first_screen_x = 0.0F;
    float first_screen_y = 0.0F;
    float first_depth = 0.0F;

    for (uint32_t tile_id = 0; tile_id < vertex_tiles; tile_id++) {
        const uint32_t tile_vertex_start = tile_id * kElementsPerTile;
        const uint32_t valid_vertices =
            vertex_count > tile_vertex_start ? vertex_count - tile_vertex_start : static_cast<uint32_t>(0);
        const uint32_t vertices_in_tile = valid_vertices < kElementsPerTile ? valid_vertices : kElementsPerTile;

        const uint64_t input_x_noc_addr = get_noc_addr(tile_id, input_x);
        const uint64_t input_y_noc_addr = get_noc_addr(tile_id, input_y);
        const uint64_t input_z_noc_addr = get_noc_addr(tile_id, input_z);
        const uint64_t screen_x_noc_addr = get_noc_addr(tile_id, screen_x);
        const uint64_t screen_y_noc_addr = get_noc_addr(tile_id, screen_y);
        const uint64_t depth_noc_addr = get_noc_addr(tile_id, depth);

        const uint32_t input_x_l1_addr = aligned_l1_slot_addr(aligned_scratch_l1_addr, 0, input_x_noc_addr);
        const uint32_t input_y_l1_addr = aligned_l1_slot_addr(aligned_scratch_l1_addr, 1, input_y_noc_addr);
        const uint32_t input_z_l1_addr = aligned_l1_slot_addr(aligned_scratch_l1_addr, 2, input_z_noc_addr);
        const uint32_t screen_x_l1_addr = aligned_l1_slot_addr(aligned_scratch_l1_addr, 3, screen_x_noc_addr);
        const uint32_t screen_y_l1_addr = aligned_l1_slot_addr(aligned_scratch_l1_addr, 4, screen_y_noc_addr);
        const uint32_t depth_l1_addr = aligned_l1_slot_addr(aligned_scratch_l1_addr, 5, depth_noc_addr);

        noc_async_read(input_x_noc_addr, input_x_l1_addr, kCoordinateTileBytes);
        noc_async_read(input_y_noc_addr, input_y_l1_addr, kCoordinateTileBytes);
        noc_async_read(input_z_noc_addr, input_z_l1_addr, kCoordinateTileBytes);
        noc_async_read_barrier();

        float* input_x_values = reinterpret_cast<float*>(input_x_l1_addr);
        float* input_y_values = reinterpret_cast<float*>(input_y_l1_addr);
        float* input_z_values = reinterpret_cast<float*>(input_z_l1_addr);
        float* screen_x_values = reinterpret_cast<float*>(screen_x_l1_addr);
        float* screen_y_values = reinterpret_cast<float*>(screen_y_l1_addr);
        float* depth_values = reinterpret_cast<float*>(depth_l1_addr);

        for (uint32_t i = 0; i < vertices_in_tile; i++) {
            const float x = input_x_values[i];
            const float y = input_y_values[i];
            const float z = input_z_values[i];
            const float clip_x =
                object_to_clip[0] * x + object_to_clip[1] * y + object_to_clip[2] * z + object_to_clip[3];
            const float clip_y =
                object_to_clip[4] * x + object_to_clip[5] * y + object_to_clip[6] * z + object_to_clip[7];
            const float clip_z =
                object_to_clip[8] * x + object_to_clip[9] * y + object_to_clip[10] * z + object_to_clip[11];
            const float clip_w =
                object_to_clip[12] * x + object_to_clip[13] * y + object_to_clip[14] * z + object_to_clip[15];
            const float inv_w = 1.0F / clip_w;
            const float ndc_x = clip_x * inv_w;
            const float ndc_y = clip_y * inv_w;
            const float ndc_z = clip_z * inv_w;

            screen_x_values[i] = (ndc_x * 0.5F + 0.5F) * static_cast<float>(viewport_width);
            screen_y_values[i] = (0.5F - ndc_y * 0.5F) * static_cast<float>(viewport_height);
            depth_values[i] = ndc_z * 0.5F + 0.5F;
        }

        if (tile_id == 0 && vertices_in_tile > 0) {
            first_screen_x = screen_x_values[0];
            first_screen_y = screen_y_values[0];
            first_depth = depth_values[0];
        }

        noc_async_write(screen_x_l1_addr, screen_x_noc_addr, kCoordinateTileBytes);
        noc_async_write(screen_y_l1_addr, screen_y_noc_addr, kCoordinateTileBytes);
        noc_async_write(depth_l1_addr, depth_noc_addr, kCoordinateTileBytes);
        noc_async_write_barrier();
    }

    DPRINT_DATA0(
        DPRINT << "Vertex transform kernel: vertices=" << vertex_count << ", coordinate_tiles=" << vertex_tiles
               << ", viewport=" << viewport_width << "x" << viewport_height << ", v0_screen=(" << F32(first_screen_x)
               << "," << F32(first_screen_y) << "), depth=" << F32(first_depth) << ENDL(););
}
