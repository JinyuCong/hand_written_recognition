import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from model import PictureDataset, LeNet

images = []
labels = []

for image_path in glob.glob('./archive/data/data/*'):
    label = int(image_path.split('_')[-1].split('.')[0]) - 1
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255.0
    images.append(image)
    labels.append(label)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

train_dataset = PictureDataset(X_train, y_train)
test_dataset = PictureDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.001
epochs = 30
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