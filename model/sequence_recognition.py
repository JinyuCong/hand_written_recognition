import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import LeNet


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
    return binary


def concatenate_images(image1: np.array, image2: np.array) -> np.array:
    image1 = cv2.resize(image1, (64, 64))
    image2 = cv2.resize(image2, (64, 64))
    concatenated = np.concatenate((image1, image2), axis=1)
    return concatenated


def separate_characters(img: np.array) -> list[np.array]:
    pixel_sequence = np.sum(img, axis=0)
    zero_segments = []
    in_zero_segment = False
    start = None

    for i, num in enumerate(pixel_sequence):
        if num == 16320:
            if not in_zero_segment:  # 开始一个新的0的区间
                start = i
                in_zero_segment = True
        else:
            if in_zero_segment:  # 结束当前0的区间
                zero_segments.append((start, i - 1))
                in_zero_segment = False

    if in_zero_segment:
        zero_segments.append((start, len(pixel_sequence) - 1))

    split_index = []
    for segment in zero_segments[1:-1]:
        index_to_separate = (segment[0] + segment[1]) // 2
        split_index.append(index_to_separate)

    return np.split(img, split_index, axis=1)


def sequence_recognition(img: np.array):
    model = LeNet()
    model.load_state_dict(torch.load("./model.pth"))
    chars = separate_characters(img)
    r = []
    for char in chars:
        char = cv2.resize(char, (64, 64))
        char = torch.tensor(char).to(torch.float).view(1, 1, 64, 64)
        res = model(char)
        res = torch.argmax(res, dim=1)
        r.append(res)
    return r


test_img1 = read_image("./test_images/4.png")
test_img2 = read_image("./test_images/7.png")
test_img3 = read_image("./test_images/h.png")
test_img4 = read_image("./test_images/q.png")

concat_img = concatenate_images(test_img1, test_img2)
concat_img = concatenate_images(concat_img, test_img3)
concat_img = concatenate_images(concat_img, test_img4)

plt.imshow(concat_img)
plt.show()

r = sequence_recognition(concat_img)
print(r[3])