import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from model import PictureDataset, LeNet, AlexNet

X_train, X_test, y_train, y_test = [], [], [], []

label_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11,
                 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23,
                 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35}

for image_path_train in glob.glob('./data/training_data/*/*'):
    label = image_path_train.split('\\')[-2]
    image = cv2.imread(image_path_train)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
    X_train.append(cv2.resize(binary, (64, 64)))
    y_train.append(label_mapping[label])

for image_path_test in glob.glob('./data/testing_data/*/*'):
    label = image_path_test.split('\\')[-2]
    image = cv2.imread(image_path_test)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
    X_test.append(cv2.resize(binary, (64, 64)))
    y_test.append(label_mapping[label])


train_dataset = PictureDataset(X_train, y_train)
test_dataset = PictureDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0001
epochs = 100
criterion = nn.CrossEntropyLoss()

model = LeNet().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)


for epoch in range(epochs):

    train_loss, train_total = 0, 0
    for images, labels in train_loader:
        images = images.to(torch.float).to(device)
        labels = labels.to(device)

        predictions = model.forward(images)
        loss = criterion(predictions, labels)

        loss.backward()
        opt.step()
        opt.zero_grad()

        train_loss += loss.item() * images.size(0)
        train_total += images.size(0)

    test_correct, test_total = 0, 0
    for test_images, test_labels in test_loader:
        test_images = test_images.to(torch.float).to(device)
        test_labels = test_labels.to(device)

        pre = model.forward(test_images)
        test_correct += int(torch.sum(torch.argmax(pre, dim=1) == test_labels).item())
        test_total += test_labels.size(0)

    epoch_loss = train_loss / train_total
    epoch_acc = test_correct / test_total

    print(f'Epoch [{epoch + 1}/{epochs}]: Train loss: {epoch_loss:.4f} | '
          f'Test accuracy: {epoch_acc * 100:.4f}%')

torch.save(model.state_dict(), 'model.pth')
