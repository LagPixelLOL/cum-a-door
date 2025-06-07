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

def select_palette(img, n_select, palette, global_colors=None):
    is_chunked = len(img.shape) == 5
    if not is_chunked:
        img = np.expand_dims(img, (0, 1))
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
        selected_colors = set() if global_colors is None else {tuple(global_color) for global_color in global_colors}
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
    if not is_chunked:
        colors = colors.squeeze((0, 1))
    return colors

def chunk_view(x, side_y, side_x):
    return np.lib.stride_tricks.as_strided(
        x,
        (x.shape[0] // side_y, x.shape[1] // side_x, side_y, side_x, *x.shape[2:]),
        (x.strides[0] * side_y, x.strides[1] * side_x, x.strides[0], x.strides[1], *x.strides[2:]),
    )

def floyd_steinberg_dither(img, side_y, side_x, n_select, palette=C64Colors.get_array(), global_colors=None):
    img = img.copy()
    result = np.zeros_like(img)
    img_chunked = chunk_view(img, side_y, side_x)
    result_chunked = chunk_view(result, side_y, side_x)
    palette = select_palette(img_chunked, n_select, palette, global_colors)
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
    return result, palette

def make_native_order(img_packed):
    img_packed = img_packed.reshape(-1)
    result = np.zeros_like(img_packed)
    for i, v in enumerate(img_packed):
        result[i // 320 * 320 + i % 320 // 40 + i % 40 * 8] = v
    return result

def save_sbm_c64bmp(img, save_path, side_y, side_x, palette, native_order):
    img = img.copy()
    img_chunked = chunk_view(img, side_y, side_x)
    img_chunked[...] = img_chunked == np.expand_dims(palette[:, :, 0, :], (2, 3))
    img = np.packbits((img != 0).all(-1))
    if native_order:
        img = make_native_order(img)
    img = img.tobytes()
    colors = np.zeros(palette.shape[:-1], dtype=np.uint8)
    for c64color in C64Colors:
        colors[(palette == c64color.rgb_normed).all(-1).nonzero()] = c64color.id
    colors[:, :, 0] <<= 4
    colors = colors.sum(-1, dtype=np.uint8).tobytes()
    with open(save_path, "wb") as f:
        f.write(b"\x42" if native_order else b"\x02")
        f.write(img)
        f.write(colors)

def save_mbm_c64bmp(img, save_path, side_y, side_x, palette, global_colors, native_order):
    img_chunked = chunk_view(img, side_y, side_x)
    result = np.zeros(img.shape[:-1], dtype=np.uint8)
    result_chunked = chunk_view(result, side_y, side_x)
    global_color = global_colors[0]
    palette = palette[(palette != global_color).any(-1).nonzero()].reshape((*palette.shape[0:2], 3, palette.shape[3]))
    for i in range(3):
        result_chunked[(img_chunked == np.expand_dims(palette[:, :, i, :], (2, 3))).all(-1).nonzero()] = i + 1
    result = result.reshape((result.shape[0], result.shape[1] // 4, 4))
    for i in range(3):
        result[:, :, i] <<= (3 - i) * 2
    result = result.sum(-1, dtype=np.uint8)
    if native_order:
        result = make_native_order(result)
    result = result.tobytes()
    screen_ram = np.zeros((*palette.shape[:2], 2), dtype=np.uint8)
    color_ram = np.zeros(palette.shape[:2], dtype=np.uint8)
    for c64color in C64Colors:
        screen_ram[(palette[:, :, :2, :] == c64color.rgb_normed).all(-1).nonzero()] = c64color.id
        color_ram[(palette[:, :, 2, :] == c64color.rgb_normed).all(-1).nonzero()] = c64color.id
        if not isinstance(global_color, bytes) and (global_color == c64color.rgb_normed).all():
            global_color = np.uint8(c64color.id).tobytes()
    screen_ram[:, :, 0] <<= 4
    screen_ram = screen_ram.sum(-1, dtype=np.uint8).tobytes()
    color_ram = color_ram.reshape((color_ram.shape[0], color_ram.shape[1] // 2, 2))
    color_ram[:, :, 0] <<= 4
    color_ram = color_ram.sum(-1, dtype=np.uint8).tobytes()
    with open(save_path, "wb") as f:
        f.write(b"\x43" if native_order else b"\x03")
        f.write(result)
        f.write(screen_ram)
        f.write(color_ram)
        f.write(global_color)

def parse_args():
    parser = argparse.ArgumentParser(description="Dither an image using Floyd-Steinberg algorithm.")
    parser.add_argument("-i", "--input", default="input.png", help="The path to the input image, default to \"input.png\"")
    parser.add_argument("-o", "--output", default="output.png", help="The path to save the processed image, use a path with extension \".c64bmp\" to save as c64bmp format, default to \"output.png\"")
    parser.add_argument("-m", "--mono", action="store_true", help="If set, use monochrome mode, otherwise use the full palette")
    parser.add_argument("-s", "--sbm", action="store_true", help="If set, use standard bitmap mode, otherwise use multicolor bitmap mode")
    parser.add_argument("-n", "--native-order", action="store_true", help="If set, use native order, which is faster, otherwise use linear order")
    return parser.parse_args()

def main():
    args = parse_args()
    img = scale_and_crop(args.input, 320, 200)
    if not args.sbm:
        img = img.resize((320 // 2, 200), Image.LANCZOS)
    img = np.array(img, dtype=np.float32) / 255
    if args.mono:
        palette = np.array([C64Colors.BLACK.rgb_normed, C64Colors.WHITE.rgb_normed, C64Colors.DARK_GREY.rgb_normed, C64Colors.GREY.rgb_normed, C64Colors.LIGHT_GREY.rgb_normed])
    else:
        palette = C64Colors.get_array()
    if args.sbm:
        global_colors = None
    else:
        global_colors = select_palette(img, 1, palette)
    chunk_shape = (8, 8 if args.sbm else 4)
    img, palette = floyd_steinberg_dither(img, *chunk_shape, 2 if args.sbm else 4, palette, global_colors)
    if os.path.splitext(args.output)[1] == ".c64bmp":
        if args.sbm:
            save_sbm_c64bmp(img, args.output, *chunk_shape, palette, args.native_order)
            return
        save_mbm_c64bmp(img, args.output, *chunk_shape, palette, global_colors, args.native_order)
        return
    if not args.sbm:
        img = img.repeat(2, -2)
    img = Image.fromarray((img * 255).astype(np.uint8)).resize((320 * 8, 200 * 8), Image.NEAREST)
    img.save(args.output)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user, exiting...")
        sys.exit(1)
