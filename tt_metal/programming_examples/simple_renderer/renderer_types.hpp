// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

namespace tt::tt_metal::programming_examples::simple_renderer {

constexpr uint32_t kTileWidth = 32;
constexpr uint32_t kTileHeight = 32;
constexpr uint32_t kPixelsPerTile = kTileWidth * kTileHeight;
constexpr uint32_t kRgbaBytesPerPixel = sizeof(uint32_t);
constexpr uint32_t kColorTileBytes = kPixelsPerTile * kRgbaBytesPerPixel;

struct RendererConfig {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t tiles_x = 0;
    uint32_t tiles_y = 0;
};

struct TileBin {
    uint32_t offset = 0;
    uint32_t count = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
};

static_assert(sizeof(TileBin) == 16);

struct ScreenVertex {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
};

struct ScreenTriangle {
    ScreenVertex v0;
    ScreenVertex v1;
    ScreenVertex v2;
    uint32_t rgba = 0;
};

struct TileBinningResult {
    std::vector<TileBin> tile_bins;
    std::vector<uint32_t> triangle_indices;
};

}  // namespace tt::tt_metal::programming_examples::simple_renderer
