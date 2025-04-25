import os
import sys
import argparse
import collections
import numpy as np
import scipy as sp
from PIL import Image
from enum import Enum

# Copied from LagPixelLOL/cog-sdxl.
def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong
    # background for transparent images. The call to `alpha_composite` handles this case.
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite

# Copied from LagPixelLOL/cog-sdxl.
def scale_and_crop(image_path, width, height):
    with Image.open(image_path) as img:
        img = convert_to_rgb(img)

    img_ratio = img.width / img.height
    target_ratio = width / height

    if img_ratio > target_ratio:
        scale_factor = height / img.height
        new_width = int(img.width * scale_factor)
        new_height = height
    else:
        scale_factor = width / img.width
        new_width = width
        new_height = int(img.height * scale_factor)

    img_resized = img.resize((new_width, new_height), Image.LANCZOS)

    left = (new_width - width) // 2
    top = (new_height - height) // 2
    right = left + width
    bottom = top + height

    return img_resized.crop((left, top, right, bottom))

class C64Colors(Enum):
    BLACK = (0x0, (0x00, 0x00, 0x00))
    WHITE = (0x1, (0xFF, 0xFF, 0xFF))
    RED = (0x2, (0x68, 0x37, 0x2B))
    CYAN = (0x3, (0x70, 0xA4, 0xB2))
    PURPLE = (0x4, (0x6F, 0x3D, 0x86))
    GREEN = (0x5, (0x58, 0x8D, 0x43))
    BLUE = (0x6, (0x35, 0x28, 0x79))
    YELLOW = (0x7, (0xB8, 0xC7, 0x6F))
    ORANGE = (0x8, (0x6F, 0x4F, 0x25))
    BROWN = (0x9, (0x43, 0x39, 0x00))
    PINK = (0xA, (0x9A, 0x67, 0x59))
    DARK_GREY = (0xB, (0x44, 0x44, 0x44))
    GREY = (0xC, (0x6C, 0x6C, 0x6C))
    LIGHT_GREEN = (0xD, (0x9A, 0xD2, 0x84))
    LIGHT_BLUE = (0xE, (0x6C, 0x5E, 0xB5))
    LIGHT_GREY = (0xF, (0x95, 0x95, 0x95))

    def __init__(self, id, rgb):
        self.id = np.uint8(id)
        self.rgb = np.array(rgb, dtype=np.uint8)
        self.rgb_normed = self.rgb.astype(np.float32) / 255

    @classmethod
    def get_array(cls):
        return np.array([c64color.rgb_normed for c64color in cls])

def select_palette(img, n_select, palette):
    chunk_y, chunk_x, region_y, region_x, n_channels = img.shape
    chunk_size = chunk_y * chunk_x
    img = img.reshape((chunk_size, region_y * region_x, n_channels))
    colors = []
    for chunk in img:
        colors.append(sp.cluster.vq.kmeans(chunk, n_select, 8, rng=42)[0])
    for i, color in enumerate(colors):
        colors[i] = np.resize(color, (n_select, n_channels))
    colors = np.array(colors)
    for i, chunk in enumerate(colors):
        selected_colors = set()
        candidates = []
        for color in chunk:
            candidate = []
            for palette_element in palette:
                dist = np.linalg.norm(palette_element - color)
                candidate.append((dist, palette_element))
            candidate.sort(key=lambda x: x[0], reverse=True)
            queue = collections.deque(maxlen=n_select)
            queue.extendleft(candidate)
            candidates.append(queue)
        while len(selected_colors) < n_select:
            dist = float("inf")
            index = None
            for j, candidate in enumerate(candidates):
                if candidate[0] is None or tuple(candidate[0][1]) in selected_colors:
                    candidate[0] = None
                    continue
                if candidate[0][0] < dist:
                    dist = candidate[0][0]
                    index = j
            if index is None:
                for candidate in candidates:
                    candidate.popleft()
                continue
            selected_colors.add(tuple(candidates[index][0][1]))
        selected_colors = np.array(list(selected_colors))
        colors[i] = selected_colors
    # for chunk in (colors * 255).astype(np.uint8):
    #     for r, g, b in chunk:
    #         print(f"\033[48;2;{r};{g};{b}m  ", end="", flush=True)
    #     print("\033[0m")
    colors = colors.reshape((chunk_y, chunk_x, n_select, n_channels))
    return colors

def chunk_view(x, side_y, side_x):
    y_len, x_len, n_channels = x.shape
    y_stride, x_stride, channel_stride = x.strides
    return np.lib.stride_tricks.as_strided(
        x,
        (y_len // side_y, x_len // side_x, side_y, side_x, n_channels),
        (y_stride * side_y, x_stride * side_x, y_stride, x_stride, channel_stride),
    )

def floyd_steinberg_dither(img, side_y, side_x, n_select, palette=None):
    img = img.copy()
    result = np.zeros_like(img)
    img_chunked = chunk_view(img, side_y, side_x)
    result_chunked = chunk_view(result, side_y, side_x)
    palette = C64Colors.get_array() if palette is None else palette
    palette = select_palette(img_chunked, n_select, palette)
    for y in range(side_y):
        for x in range(side_x):
            vec_errs = np.expand_dims(img_chunked[:, :, y, x], 2) - palette
            errs = np.linalg.norm(vec_errs, axis=-1)
            min_indices = np.argmin(errs, -1, keepdims=True)
            sel_indices = np.expand_dims(np.repeat(min_indices, img.shape[-1], -1), -2)
            result_chunked[:, :, y, x] = np.squeeze(np.take_along_axis(palette, sel_indices, -2), -2)
            vec_errs = np.squeeze(np.take_along_axis(vec_errs, sel_indices, -2), -2)
            if x + 1 < side_x:
                img_chunked[:, :, y, x + 1] += 7 / 16 * vec_errs
                if y + 1 < side_y:
                    img_chunked[:, :, y + 1, x + 1] += 1 / 16 * vec_errs
            if y + 1 < side_y:
                img_chunked[:, :, y + 1, x] += 5 / 16 * vec_errs
                if x > 0:
                    img_chunked[:, :, y + 1, x - 1] += 3 / 16 * vec_errs
    return result, result_chunked, palette

def parse_args():
    parser = argparse.ArgumentParser(description="Dither an image using Floyd-Steinberg algorithm.")
    parser.add_argument("-i", "--input", default="input.png", help="The path to the input image, default to \"input.png\"")
    parser.add_argument("-o", "--output", default="output.png", help="The path to save the processed image, use a path with extension \".c64bmp\" to save as c64bmp format, default to \"output.png\"")
    parser.add_argument("-m", "--mode", default="colored", choices=["colored", "mono"], help="The path to save the processed image, default to \"output.png\"")
    return parser.parse_args()

def main():
    args = parse_args()
    img = np.array(scale_and_crop(args.input, 320, 200), dtype=np.float32) / 255
    match args.mode:
        case "colored":
            palette = None
        case "mono":
            palette = np.array([C64Colors.BLACK.rgb_normed, C64Colors.WHITE.rgb_normed, C64Colors.DARK_GREY.rgb_normed, C64Colors.LIGHT_GREY.rgb_normed])
        case _:
            raise ValueError(f"Unknown mode \"{args.mode}\"!")
    img, img_chunked, palette = floyd_steinberg_dither(img, 8, 8, 2, palette)
    if os.path.splitext(args.output)[1] == ".c64bmp":
        img_chunked[...] = img_chunked == np.expand_dims(palette[:, :, 0, :], (2, 3))
        img = np.packbits((img != 0).all(-1)).tobytes()
        colors = np.zeros(palette.shape[:-1], dtype=np.uint8)
        for c64color in C64Colors:
            matched_indices = (palette == c64color.rgb_normed).all(-1).nonzero()
            colors[matched_indices] = c64color.id
        colors[:, :, 0] <<= 4
        colors = colors.sum(-1, dtype=np.uint8).tobytes()
        with open(args.output, "wb") as f:
            f.write(b"\x02")
            f.write(img)
            f.write(colors)
        return
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.save(args.output)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user, exiting...")
        sys.exit(1)
