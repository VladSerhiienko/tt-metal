// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

namespace tt::tt_metal::programming_examples::simple_renderer {

constexpr uint32_t kTileWidth = 32;
constexpr uint32_t kTileHeight = 32;
constexpr uint32_t kElementsPerTile = kTileWidth * kTileHeight;
constexpr uint32_t kCoordinateTileBytes = kElementsPerTile * sizeof(float);

struct VertexTransformConfig {
    uint32_t vertex_count = 0;
    uint32_t vertex_tiles = 0;
    uint32_t viewport_width = 0;
    uint32_t viewport_height = 0;
};

using Mat4 = std::array<float, 16>;

}  // namespace tt::tt_metal::programming_examples::simple_renderer
