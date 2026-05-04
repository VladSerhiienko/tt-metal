// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

void kernel_main() {
    const uint32_t width = get_arg_val<uint32_t>(0);
    const uint32_t height = get_arg_val<uint32_t>(1);
    const uint32_t tiles_x = get_arg_val<uint32_t>(2);
    const uint32_t tiles_y = get_arg_val<uint32_t>(3);
    const uint32_t total_tiles = get_arg_val<uint32_t>(4);
    const uint32_t triangle_count = get_arg_val<uint32_t>(5);
    const uint32_t tile_ref_count = get_arg_val<uint32_t>(6);

    DPRINT_DATA0(
        DPRINT << "Simple renderer skeleton on dataflow core 0: frame=" << width << "x" << height
               << ", tile_grid=" << tiles_x << "x" << tiles_y << ", total_tiles=" << total_tiles
               << ", triangles=" << triangle_count << ", tile_refs=" << tile_ref_count << ENDL(););
}
