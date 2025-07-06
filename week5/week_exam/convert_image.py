import os
import gdown
from typing import Optional, Literal
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def DownloadFile(file_id: str,
                 output_dir: str = "images",
                 filename: str = "default.jpeg") -> Optional[str]:

    os.makedirs(output_dir, exist_ok=True)

    # Đường dẫn file đầu ra
    output_path = os.path.join(output_dir, filename)

    # Nếu file đã tồn tại thì không tải lại
    if os.path.exists(output_path):
        print(f"✅ File đã tồn tại tại: {output_path}")
        return output_path

    # Tạo URL Google Drive từ file_id
    url = f"https://drive.google.com/uc?id={file_id}"

    # Tải file
    try:
        gdown.download(url, output_path, quiet=False)
        print(f"📥 Tải file thành công: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Lỗi khi tải file: {e}")
        return None


def RGB_TO_GRAY(
    image: np.ndarray,
    method: Literal["lightness", "average", "luminosity"]
) -> np.ndarray:

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Ảnh phải có dạng (H, W, 3)")

    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    if method == "lightness":
        max_rgb = np.maximum(np.maximum(R, G), B)
        min_rgb = np.minimum(np.minimum(R, G), B)
        gray = (max_rgb.astype(np.float32) + min_rgb) / 2
    elif method == "average":
        gray = (R.astype(np.float32) + G + B) / 3
    elif method == "luminosity":
        gray = 0.21 * R + 0.72 * G + 0.07 * B
    else:
        raise ValueError(
            "Phải chọn 1 trong 3 phương pháp: lightness, average, luminosity")

    return gray.astype(np.uint8)


def main():
    file_path = DownloadFile(
        file_id="1i9dqan21DjQoG5Q_VEvm0LrVwAlXD0vB",
        output_dir="images",
        filename="dog.jpeg"
    )
    img = mpimg.imread(file_path)

    # Trường hợp ảnh float (0.0 - 1.0), chuyển sang 0 - 255
    if img.dtype == np.float32 and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    # ['lightness', 'average', 'luminosity']
    gray_img = RGB_TO_GRAY(img, method="lightness")

    plt.imshow(gray_img, cmap='gray')
    plt.axis("off")
    plt.title("Ảnh xám - Luminosity method")
    plt.show()


if __name__ == "__main__":
    main()
