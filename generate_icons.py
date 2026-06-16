"""
生成浏览器插件所需的三个尺寸图标 (16/48/128 px)
"""
import struct
import zlib


def make_png(width, height, color=(0x4A, 0x90, 0xD9)):
    """生成包含纯色圆角盾牌图标的 PNG 文件"""

    def create_pixel_data(w, h, c):
        """创建 RGBA 像素数据，中心绘制盾牌形状"""
        pixels = []
        cx, cy = w / 2, h / 2
        rx, ry = w * 0.42, h * 0.44
        for y in range(h):
            # PNG 过滤器字节
            pixels.append(0)
            for x in range(w):
                # 盾牌形状：椭圆 + 底部尖角
                dx = (x - cx) / rx
                dy = (y - cy) / ry
                # 椭圆主体
                in_shield = (dx * dx + dy * dy) < 1.0 and y < h * 0.75
                # 底部三角形
                if not in_shield and y >= h * 0.55:
                    tri_width = (y - h * 0.55) / (h * 0.45)
                    if abs(x - cx) < w * 0.35 * (1 - tri_width):
                        in_shield = True
                if in_shield:
                    pixels.extend(c)
                else:
                    pixels.extend([0, 0, 0, 0])  # 透明
        return bytes(pixels)

    raw_data = create_pixel_data(width, height, color)

    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack('>I', zlib.crc32(c) & 0xffffffff)
        return struct.pack('>I', len(data)) + c + crc

    signature = b'\x89PNG\r\n\x1a\n'
    ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0))
    compressed = zlib.compress(raw_data)
    idat = chunk(b'IDAT', compressed)
    iend = chunk(b'IEND', b'')

    return signature + ihdr + idat + iend


if __name__ == '__main__':
    for size, name in [(16, 'icon16.png'), (48, 'icon48.png'), (128, 'icon128.png')]:
        png_data = make_png(size, size)
        path = f'icons/{name}'
        with open(path, 'wb') as f:
            f.write(png_data)
        print(f'✓ {path} ({size}x{size})')
    print('图标生成完毕！')
