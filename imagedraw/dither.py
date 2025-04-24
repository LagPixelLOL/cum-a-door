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
        self.id = id
        self.rgb = rgb

def select_palette(img, palette, n_select):
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

def chunk_view(x, chunk_side_len):
    y_len, x_len, n_channels = x.shape
    y_stride, x_stride, channel_stride = x.strides
    return np.lib.stride_tricks.as_strided(
        x,
        (y_len // chunk_side_len, x_len // chunk_side_len, chunk_side_len, chunk_side_len, n_channels),
        (y_stride * chunk_side_len, x_stride * chunk_side_len, y_stride, x_stride, channel_stride),
    )

def floyd_steinberg_dither_colored(img):
    img = img.copy()
    result = np.zeros_like(img)
    chunk_side_len = 8
    img_chunked = chunk_view(img, chunk_side_len)
    result_chunked = chunk_view(result, chunk_side_len)
    palette = [np.array(e.rgb).astype(np.float32) / 255 for e in C64Colors]
    palette = select_palette(img_chunked, palette, 2)
    for y in range(chunk_side_len):
        for x in range(chunk_side_len):
            vec_errs = np.expand_dims(img_chunked[:, :, y, x], 2) - palette
            errs = np.linalg.norm(vec_errs, axis=-1)
            min_indices = np.argmin(errs, -1, keepdims=True)
            sel_indices = np.expand_dims(np.repeat(min_indices, img.shape[-1], -1), -2)
            result_chunked[:, :, y, x] = np.squeeze(np.take_along_axis(palette, sel_indices, -2), -2)
            vec_errs = np.squeeze(np.take_along_axis(vec_errs, sel_indices, -2), -2)
            if x + 1 < chunk_side_len:
                img_chunked[:, :, y, x + 1] += 7 / 16 * vec_errs
                if y + 1 < chunk_side_len:
                    img_chunked[:, :, y + 1, x + 1] += 1 / 16 * vec_errs
            if y + 1 < chunk_side_len:
                img_chunked[:, :, y + 1, x] += 5 / 16 * vec_errs
                if x > 0:
                    img_chunked[:, :, y + 1, x - 1] += 3 / 16 * vec_errs
    return result

def floyd_steinberg_dither_mono(img):
    y_len, x_len = img.shape
    img = img.copy()
    result = np.zeros_like(img)
    for y in range(y_len):
        for x in range(x_len):
            quant_val = img[y, x] >= 0.5
            quant_err = img[y, x] - quant_val
            result[y, x] = quant_val
            if x + 1 < x_len:
                img[y, x + 1] += 7 / 16 * quant_err
                if y + 1 < y_len:
                    img[y + 1, x + 1] += 1 / 16 * quant_err
            if y + 1 < y_len:
                img[y + 1, x] += 5 / 16 * quant_err
                if x > 0:
                    img[y + 1, x - 1] += 3 / 16 * quant_err
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="Dither an image using Floyd-Steinberg algorithm.")
    parser.add_argument("-i", "--input", default="input.png", help="The path to the input image, default to \"input.png\"")
    parser.add_argument("-o", "--output", default="output.png", help="The path to save the processed image, use a path with extension \".c64bmp\" to save as c64bmp format, default to \"output.png\"")
    parser.add_argument("-m", "--mode", default="colored", choices=["colored", "mono"], help="The path to save the processed image, default to \"output.png\"")
    return parser.parse_args()

def main():
    args = parse_args()
    img = scale_and_crop(args.input, 320, 200)
    match args.mode:
        case "colored":
            img = np.array(img).astype(np.float32) / 255
            img = floyd_steinberg_dither_colored(img)
        case "mono":
            img = np.array(img.convert("L")).astype(np.float32) / 255
            img = floyd_steinberg_dither_mono(img)
            if os.path.splitext(args.output)[1] == ".c64bmp":
                img = np.packbits(img != 0).tobytes()
                with open(args.output, "wb") as f:
                    f.write(b"\x02")
                    f.write(img)
                    f.write(b"\x10" * 1000)
                    return
        case _:
            raise ValueError(f"Unknown mode \"{args.mode}\"!")
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.save(args.output)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user, exiting...")
        sys.exit(1)
