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

# Autoriser les requêtes inter-domaines (traitement CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autoriser toutes les sources
    allow_credentials=True,
    allow_methods=["*"],  # Autoriser toutes les méthodes HTTP
    allow_headers=["*"],  # Autoriser tous les en-têtes de requête
)
# 新增
def sort_contours(contours):
    """ Classer les descriptifs de gauche à droite, puis de haut en bas """
    bounding_boxes = [cv2.boundingRect(ctr) for ctr in contours]

    # Regrouper les contours par ligne (en supposant que des tailles de caractères et des coordonnées y similaires soient regroupées dans la même ligne)
    lines = []
    threshold = 20  # espacement minimal entre les lignes dans le sens des ordonnées

    for x, y, w, h in sorted(bounding_boxes, key=lambda b: b[1]):
        placed = False
        for line in lines:
            if abs(line[-1][1] - y) < threshold:
                line.append((x, y, w, h))
                placed = True
                break
        if not placed:
            lines.append([(x, y, w, h)])

    # Chaque ligne est triée de gauche à droite par la coordonnée x
    sorted_boxes = [sorted(line, key=lambda b: b[0]) for line in lines]

    # Réorganisation de l'aperçu dans l'ordre des lignes
    sorted_contours = []
    for line in sorted_boxes:
        for box in line:
            for ctr in contours:
                if cv2.boundingRect(ctr) == box:
                    sorted_contours.append(ctr)
                    break

    return sorted_contours


def split_letters(image, padding=5, min_width=10, min_height=10, space_threshold=20, line_threshold=15,
                  from_canvas=False):
    """
    Segmentation améliorée des caractères : détection des nouvelles lignes et des espaces.

    Paramètres.
        image (np.array) : image d'entrée (échelle de gris).
        padding (int) : valeur en pixel pour l'extension des limites du caractère.
        min_width (int) : Largeur minimale pour filtrer les petits bruits.
        min_height (int) : hauteur minimale pour le filtrage des petits bruits.
        space_threshold (int) : espacement minimum pour l'insertion d'espaces.
        line_threshold (int) : Seuil pour l'espacement des lignes (un espacement supérieur à cette valeur est considéré comme un saut de ligne).
        from_canvas (bool) : si l'entrée se fait à partir du bloc-notes.
    """
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: (b[1], b[0]))

    lines = []
    current_line = []
    prev_y = None

    for x, y, w, h in bounding_boxes:
        if w < min_width or h < min_height:
            continue

        if prev_y is not None and abs(y - prev_y) > line_threshold:
            lines.append(sorted(current_line, key=lambda b: b[0]))
            current_line = []

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
                    chars.append(" ")

            prev_x, prev_w = x, w
            letter = image[y:y + h, x:x + w]
            chars.append(letter)

        if not from_canvas:
            chars.append("\n")

    return chars


def sequence_recognition(img: np.array, from_canvas=False):
    model = LeNet()
    model.load_state_dict(torch.load("./model/model.pth", map_location=torch.device('cpu')))

    chars = split_letters(img, padding=3, from_canvas=from_canvas)  # transfer canvas `from_canvas`

    results = []
    for char in chars:
        if isinstance(char, str):
            if from_canvas and char in [" ", "\n"]:
                continue
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
    """Traiter les images manuscrites ou téléchargées à partir de l'interface utilisateur"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    predicted_char = sequence_recognition(img, from_canvas=from_canvas)

    return {"predicted_class": "".join(map(str, predicted_char))}

