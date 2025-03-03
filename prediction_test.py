import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from model import PictureDataset, LeNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet()

model.load_state_dict(torch.load("./model.pth"))

for test_img in glob.glob("./test_images/*"):
    test_image = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE) / 255.0
    print(test_image)
    plt.imshow(test_image)
    plt.show()
    test_image = torch.tensor(test_image).to(torch.float).view(1, 1, 64, 64)
    res = model.forward(test_image)
    res = torch.argmax(res, dim=1)
