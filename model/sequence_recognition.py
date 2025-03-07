import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import LeNet


letter_to_label = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11,
                    'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23,
                    'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35}

label_to_letter = {label: letter for letter, label in letter_to_label.items()}


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


def split_letters(image, padding=5, min_width=10, min_height=10):
    """
    Segmenter les lettres d'une image
    et les enregistrer sous forme d'images individuelles.

    PM:
        image (np.array) : image d'entrée en niveaux de gris ;
        padding (int): marge ajouté autour de chaque caractère ;
        min_width (int): largeur minimale d'un caractère pour filtrer le bruit ;
        min_height (int): hauteur minimale.
    """

    # Binariser l'image : les lettres deviennent blanches sur fond noir
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Contours de gauche à droite
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    chars = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        if w < min_width or h < min_height: # Parcours des contours et découpage des lettres
            continue

        # Ajouter des marges
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = min(w + 2 * padding, image.shape[1] - x)
        h = min(h + 2 * padding, image.shape[0] - y)

        letter = image[y:y+h, x:x+w]

        plt.imshow(letter)
        plt.show()

        chars.append(letter)

    return chars


def sequence_recognition(img: np.array):
    model = LeNet()
    model.load_state_dict(torch.load("./model.pth", map_location=torch.device('cpu')))
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


test_img1 = read_image("./test_images/4.png")
test_img2 = read_image("./test_images/7.png")
test_img3 = read_image("./test_images/h.png")
test_img4 = read_image("./test_images/q.png")

concat_img = concatenate_images(test_img1, test_img2)
concat_img = concatenate_images(concat_img, test_img3)
concat_img = concatenate_images(concat_img, test_img4)

res = sequence_recognition(concat_img)
for i in res:
    print(i)