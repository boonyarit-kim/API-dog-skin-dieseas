from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import uvicorn

app = FastAPI(title="Dog Skin Disease API")

# --- 1. Load Model ---
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet50(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)
    
    try:
        model.load_state_dict(torch.load('Dog_Skin_disease_ResNetModel.pth', map_location=device))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, device

    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# --- 2. เตรียม Transform ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity', 'demodicosis', 'ringworm']
class_names_th = [
    'ผิวหนังอักเสบ',
    'การติดเชื้อรา',
    'สุขภาพดี',
    'ภาวะภูมิไวเกิน',
    'โรคขี้เรื้อนแห้ง',
    'กลาก'
]

# --- 3. สร้างหน้าแรก ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return '''
    <!doctype html>
    <title>Dog Skin Disease Prediction</title>
    <h1>FastAPI Server is Running!</h1>
    <p>ไปที่ <a href="/docs">http://127.0.0.1:8000/docs</a> เพื่อทดสอบอัปโหลดรูปภาพผ่าน Swagger UI</p>
    '''

# --- 4. API สำหรับทำนายผล ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    # เช็คว่าเป็นไฟล์รูปภาพหรือไม่
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # อ่านไฟล์ที่อัปโหลดมา (ใช้ await เพราะ FastAPI เป็น Async)
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)
        
        predicted_class = class_names[preds.item()]
        confidence_score = confidence.item() * 100

        # ส่งค่ากลับเป็น JSON อัตโนมัติ
        return {
            "result": predicted_class,
            "confidence": f"{confidence_score:.2f}%",
            "result_th": class_names_th[preds.item()]  # เพิ่มผลลัพธ์ภาษาไทย
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    # รันเซิร์ฟเวอร์ด้วย uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)