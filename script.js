const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Réglage du style de pinceau
ctx.strokeStyle = "black";
ctx.lineWidth = 10;
ctx.lineCap = "round";

// événement de clic de souris
canvas.addEventListener("mousedown", (e) => {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
});

// événement de survol
canvas.addEventListener("mousemove", draw);

// événement enlève souris
canvas.addEventListener("mouseup", () => isDrawing = false);
canvas.addEventListener("mouseout", () => isDrawing = false);

// fonction de traçage
function draw(e) {
    if (!isDrawing) return;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();

    [lastX, lastY] = [e.offsetX, e.offsetY];
}

// Effacer le canevas
function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById("result").innerText = "";
}

// Envoi d'images au backend
async function predict() {
    const image = canvas.toDataURL("image/png");  // Obtenir une image de la toile
    const blob = await fetch(image).then(res => res.blob());

    const formData = new FormData();
    formData.append("file", blob, "canvas.png");  // Ajouter un nom de fichier par défaut
    formData.append("from_canvas", "true");  // Indiquer au backend qu'il s'agit d'un pad input

    const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
    });

    const result = await response.json();
    document.getElementById("result").innerText = result.predicted_class;
}



let imageData = null;

document.getElementById("fileInput").addEventListener("change", function (event) {
    let file = event.target.files[0];
    if (file) {
        let reader = new FileReader();
        reader.onload = function (e) {
            document.getElementById("imagePreview").src = e.target.result;
            document.getElementById("imagePreview").style.display = "block";
            imageData = file;
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById("pasteBox").addEventListener("paste", function (event) {
    let items = (event.clipboardData || event.originalEvent.clipboardData).items;
    for (let item of items) {
        if (item.type.indexOf("image") !== -1) {
            let blob = item.getAsFile();
            let reader = new FileReader();
            reader.onload = function (e) {
                document.getElementById("imagePreview").src = e.target.result;
                document.getElementById("imagePreview").style.display = "block";
                imageData = blob;
            };
            reader.readAsDataURL(blob);
            break;
        }
    }
});

function uploadImage() {
    if (!imageData) {
        alert("Veuillez d'abord télécharger ou coller l'image");
        return;
    }

    let formData = new FormData();

    if (imageData instanceof File) {
        formData.append("file", imageData);  // Téléchargement direct Fichier
    } else {
        formData.append("file", imageData, "pasted_image.png");
    }

    fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log("API return:", data);
        if (data.predicted_class) {
            document.getElementById("result").innerText = data.predicted_class;
        } else {
            document.getElementById("result").innerText = "échec de la reconnaissance: " + (data.error || "erreur inconnue");
        }
    })
    .catch(error => console.error("Erreur de demande:", error));
}
