from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import torch
from model.model import LeNet

letter_to_label = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11,
    'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22,
    'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33,
    'Y': 34, 'Z': 35
}

label_to_letter = {label: letter for letter, label in letter_to_label.items()}

app = FastAPI()

# 允许跨域请求（CORS 处理）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)
# 新增
def sort_contours(contours):
    """ 按从左到右，再从上到下排序轮廓 """
    bounding_boxes = [cv2.boundingRect(ctr) for ctr in contours]

    # 将轮廓按行分组（假设字符大小相近，y 坐标相近的归为同一行）
    lines = []
    threshold = 20  # 行间最小 y 方向间距

    for x, y, w, h in sorted(bounding_boxes, key=lambda b: b[1]):  # 先按 y 排序
        placed = False
        for line in lines:
            if abs(line[-1][1] - y) < threshold:  # y 坐标相近，认为是同一行
                line.append((x, y, w, h))
                placed = True
                break
        if not placed:
            lines.append([(x, y, w, h)])

    # 每一行按 x 坐标从左到右排序
    sorted_boxes = [sorted(line, key=lambda b: b[0]) for line in lines]

    # 按行顺序重新整理轮廓
    sorted_contours = []
    for line in sorted_boxes:
        for box in line:
            for ctr in contours:
                if cv2.boundingRect(ctr) == box:
                    sorted_contours.append(ctr)
                    break

    return sorted_contours
# 新增结束


def split_letters(image, padding=5, min_width=10, min_height=10):
    """
    从图片中分割字母并保存为单独的图片。

    参数:
        image_path (str): 输入图片路径。
        output_folder (str): 保存字母图片的文件夹路径。
        min_width (int): 字母的最小宽度（过滤小噪声）。
        min_height (int): 字母的最小高度（过滤小噪声）。
    """

    # 二值化处理（反转颜色，字母为白色，背景为黑色）
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按从左到右排序轮廓
    # contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # 新增
    contours = sort_contours(contours)
    # 新增结束

    chars = []

    # 遍历轮廓并裁剪字母
    for i, contour in enumerate(contours):
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 过滤太小的轮廓（噪声）
        if w < min_width or h < min_height:
            continue

        # 扩展边界框（留出边距）
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = min(w + 2 * padding, image.shape[1] - x)
        h = min(h + 2 * padding, image.shape[0] - y)

        # 裁剪字母
        letter = image[y:y+h, x:x+w]

        chars.append(letter)

    return chars


def sequence_recognition(img: np.array):
    model = LeNet()
    model.load_state_dict(torch.load("./model/model.pth", map_location=torch.device('cpu')))
    chars = split_letters(img, padding=3)
    results = []
    for char in chars:
        char = cv2.resize(char, (64, 64))
        char = torch.tensor(char).to(torch.float).view(1, 1, 64, 64)
        res = model(char)
        res = int(torch.argmax(res, dim=1))
        res = label_to_letter[res]
        results.append(res)
    return results


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """处理前端传来的手写图片，并进行识别"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    predicted_char = sequence_recognition(img)

    return {"predicted_class": "".join(map(str, predicted_char))}  # 组合识别结果并返回

