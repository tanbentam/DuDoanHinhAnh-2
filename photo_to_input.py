"""
Chuyển một ảnh chụp thật (điện thoại, Internet, ...) thành input kiểu CIFAR-10:
ảnh màu 32x32, vật thể nằm giữa khung.

    from photo_to_input import photo_to_input
    x = photo_to_input("xe.jpg")      # np.ndarray uint8 (32, 32, 3)

Xem trước ảnh mà mô hình thực sự "nhìn thấy":
    python photo_to_input.py xe.jpg
"""
import sys

import numpy as np
from PIL import Image, ImageOps


def photo_to_input(img, size: int = 32) -> np.ndarray:
    """img: đường dẫn file, PIL.Image hoặc np.ndarray (H, W, 3)."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif not isinstance(img, Image.Image):
        img = ImageOps.exif_transpose(Image.open(img))
    img = img.convert("RGB")

    # Cắt hình vuông ở giữa (CIFAR-10 gần như luôn đặt vật thể ở giữa ảnh).
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    # Thu nhỏ về 32x32. Thu nhỏ một lần từ ảnh lớn dễ bị răng cưa nên giảm dần.
    while img.size[0] >= 4 * size:
        img = img.resize((img.size[0] // 2,) * 2, Image.BOX)
    img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    original = ImageOps.exif_transpose(Image.open(sys.argv[1])).convert("RGB")
    x = photo_to_input(original)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(original)
    ax[0].set_title("Ảnh gốc")
    ax[1].imshow(x, interpolation="nearest")
    ax[1].set_title("Input 32x32 đưa vào mô hình")
    for a in ax:
        a.axis("off")
    plt.show()
