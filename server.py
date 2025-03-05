from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import torch
from model import LeNet
from PIL import Image

app = FastAPI()

# 允许跨域请求（CORS 处理）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)

def preprocess_image(image_file):
    """ 预处理上传的图片 """
    try:
        image = Image.open(image_file).convert("L")  # 1. 转灰度
        image = image.resize((64, 64))  # 2. 调整大小
        image = np.array(image)  # 3. 转为 NumPy 数组
        _, binary = cv2.threshold(image, 64, 255, cv2.THRESH_BINARY)  # 4. 二值化
        image = binary / 255.0  # 5. 归一化
        image = np.expand_dims(image, axis=0).astype(np.float32)  # 6. 调整维度 (1, 64, 64)
        return torch.tensor(image).unsqueeze(0)  # 7. 转 PyTorch 张量
    except Exception as e:
        print("图片预处理失败:", e)
        return None

def separate_characters(img: np.array) -> list[np.array]:
    """将整张图片分割成多个字符"""
    pixel_sequence = np.sum(img, axis=0)  # 计算每一列的像素值总和
    zero_segments = []
    in_zero_segment = False
    start = None

    for i, num in enumerate(pixel_sequence):
        if num == 0:
            if not in_zero_segment:
                start = i
                in_zero_segment = True
        else:
            if in_zero_segment:
                zero_segments.append((start, i - 1))
                in_zero_segment = False

    if in_zero_segment:
        zero_segments.append((start, len(pixel_sequence) - 1))

    split_index = [(segment[0] + segment[1]) // 2 for segment in zero_segments]

    return np.split(img, split_index, axis=1)

def sequence_recognition(img: np.array):
    """识别手写字符"""
    model = LeNet()
    model.load_state_dict(torch.load("./model.pth", map_location='cpu'))
    model.eval()

    chars = separate_characters(img)
    results = []
    for char in chars:
        char = cv2.resize(char, (64, 64))
        char = char / 255.0
        char = torch.tensor(char, dtype=torch.float32, device='cpu').view(1, 1, 64, 64)
        with torch.no_grad():
            res = model(char)
            res = torch.argmax(res, dim=1)
        results.append(res.item())

    return results

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """处理前端传来的手写图片，并进行识别"""
    try:
        image = preprocess_image(file.file)
        if image is None:
            return {"error": "无法处理图片"}

        with torch.no_grad():
            output = LeNet()(image)
            prediction = torch.argmax(output, dim=1).item()
    
        return {"predicted_class": prediction}
    except Exception as e:
        print("错误:", e)
        return {"error": str(e)}
