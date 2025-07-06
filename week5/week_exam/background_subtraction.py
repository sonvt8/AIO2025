import os
import numpy as np
import cv2


def standardized_images(output_dir: str,
                        filenames: list,
                        size=(678, 381)) -> dict:
    imgs = {}
    os.makedirs(output_dir, exist_ok=True)
    for fn in filenames:
        output_path = os.path.join(output_dir, fn)
        image = cv2.imread(output_path, 1)
        if image is None:
            raise FileNotFoundError(f"Không tìm thấy file: {output_path}")
        image = cv2.resize(image, size)
        key = os.path.splitext(fn)[0]  # Lấy tên file không có đuôi mở rộng
        imgs[key] = image
    return imgs


def replace_background(bg1_image, bg2_image, ob_image):
    pass


def main():
    filenames = ["GreenBackground.png", "Object.png", "NewBackground.png"]
    imgs = standardized_images("images", filenames)

    # Truy xuất rõ ràng:
    bg1_img = imgs["GreenBackground"]
    obj_img = imgs["Object"]
    bg2_img = imgs["NewBackground"]

    diff = cv2.absdiff(bg1_img, obj_img)
    # thresh = 0 if < 15 else 255
    _, thresh = cv2.threshold(diff, 0.05, 255, cv2.THRESH_BINARY)

    output = np.where(thresh == 0, bg2_img, obj_img)
    cv2.imwrite('output.png', output)


if __name__ == "__main__":
    main()
