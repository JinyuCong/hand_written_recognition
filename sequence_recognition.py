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
    concatenated = np.concatenate((image1, image2), axis=1)
    return concatenated


def separate_characters(img: np.array) -> list[np.array]:
    pixel_sequence = np.sum(img, axis=0)
    zero_segments = []
    in_zero_segment = False
    start = None

    for i, num in enumerate(pixel_sequence):
        if num == 0:
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
    for segment in zero_segments:
        index_to_separate = (segment[0] + segment[1]) // 2
        split_index.append(index_to_separate)

    return np.split(img, split_index, axis=1)


def sequence_recognition(img: np.array):
    model = LeNet()
    model.load_state_dict(torch.load("./model.pth", map_location='cpu'))
    model.eval()

    chars = separate_characters(img)
    # print(f"nb de caractères séparés : {len(chars)}")

    r = []
    for char in chars:
        char = cv2.resize(char, (64, 64))
        char = char / 255.0 # 新增：归一化到 [0,1]
        char = torch.tensor(char, dtype=torch.float32, device='cpu').view(1, 1, 64, 64)
        with torch.no_grad():
            res = model(char)
            res = torch.argmax(res, dim=1)
        r.append(res.item())
    return r


test_img1 = read_image("./test_images/input_1_4_1.jpg")
test_img2 = read_image("./test_images/input_1_4_2.jpg")
test_img3 = read_image("./test_images/input_1_4_3.jpg")

concat_img = concatenate_images(test_img1, test_img2)
concat_img = concatenate_images(concat_img, test_img3)

r = sequence_recognition(concat_img)
# print(len(r))
print(r[3])