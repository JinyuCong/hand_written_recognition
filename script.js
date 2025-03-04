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