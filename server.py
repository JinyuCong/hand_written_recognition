from fastapi import FastAPI, UploadFile, File, Form
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

def split_letters(image, padding=5, min_width=10, min_height=10, space_threshold=20, line_threshold=15,
                  from_canvas=False):
    """
    增强的字符分割：检测换行和空格。

    参数:
        image (np.array): 输入图片（灰度图）。
        padding (int): 字符边界扩展的像素值。
        min_width (int): 过滤小噪声的最小宽度。
        min_height (int): 过滤小噪声的最小高度。
        space_threshold (int): 插入空格的最小间距。
        line_threshold (int): 行间距的阈值（大于该值视为换行）。
        from_canvas (bool): 是否来自手写板输入。
    """
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: (b[1], b[0]))

    lines = []
    current_line = []
    prev_y = None

    for x, y, w, h in bounding_boxes:
        if w < min_width or h < min_height:
            continue  # 过滤噪声

        if prev_y is not None and abs(y - prev_y) > line_threshold:
            lines.append(sorted(current_line, key=lambda b: b[0]))  # 结束当前行
            current_line = []  # 开始新行

        current_line.append((x, y, w, h))
        prev_y = y

    if current_line:
        lines.append(sorted(current_line, key=lambda b: b[0]))

    chars = []
    for line in lines:
        prev_x, prev_w = None, None
        for x, y, w, h in line:
            if prev_x is not None:
                space_width = x - (prev_x + prev_w)
                if space_width > space_threshold and not from_canvas:
                    chars.append(" ")  # 仅在不是手写板输入时插入空格

            prev_x, prev_w = x, w
            letter = image[y:y + h, x:x + w]
            chars.append(letter)

        if not from_canvas:
            chars.append("\n")  # 仅在不是手写板输入时添加换行

    return chars


def sequence_recognition(img: np.array, from_canvas=False):
    model = LeNet()
    model.load_state_dict(torch.load("./model/model.pth", map_location=torch.device('cpu')))

    chars = split_letters(img, padding=3, from_canvas=from_canvas)  # 传递 `from_canvas`

    results = []
    for char in chars:
        if isinstance(char, str):  # 处理空格或换行
            if from_canvas and char in [" ", "\n"]:
                continue  # 手写板不考虑空格和换行
            results.append(char)
            continue

        char = cv2.resize(char, (64, 64))
        char = torch.tensor(char).to(torch.float).view(1, 1, 64, 64)
        res = model(char)
        res = int(torch.argmax(res, dim=1))
        res = label_to_letter[res]
        results.append(res)

    return results


@app.post("/predict")
async def predict(file: UploadFile = File(...), from_canvas: bool = Form(False)):
    """处理前端传来的手写图片或上传图片"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    predicted_char = sequence_recognition(img, from_canvas=from_canvas)

    return {"predicted_class": "".join(map(str, predicted_char))}

