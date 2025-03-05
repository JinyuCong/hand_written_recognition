const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// 设置画笔样式
ctx.strokeStyle = "black";
ctx.lineWidth = 10;
ctx.lineCap = "round";

// 鼠标按下事件
canvas.addEventListener("mousedown", (e) => {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
});

// 鼠标移动事件
canvas.addEventListener("mousemove", draw);

// 鼠标抬起事件
canvas.addEventListener("mouseup", () => isDrawing = false);
canvas.addEventListener("mouseout", () => isDrawing = false);

// 绘制函数
function draw(e) {
    if (!isDrawing) return;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();

    [lastX, lastY] = [e.offsetX, e.offsetY];
}

// 清除画布
function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById("result").innerText = "";
}

// 发送图像到后端
async function predict() {
    const image = canvas.toDataURL("image/png");  // 获取画布图像
    const blob = await fetch(image).then(res => res.blob());

    const formData = new FormData();
    formData.append("file", blob, "image.png");

    const response = await fetch("http://localhost:8000/predict", {
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
        alert("请先上传或粘贴图片");
        return;
    }
    let formData = new FormData();
    formData.append("file", imageData);

    fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            console.log("API 返回:", data);  // 👉 这里查看 API 返回的数据
            if (data.predicted_class) {
                document.getElementById("result").innerText = data.predicted_class;
            } else {
                document.getElementById("result").innerText = "识别失败: " + (data.error || "未知错误");
            }
        })
        .catch(error => console.error("请求出错:", error));

}