from fastapi import FastAPI, UploadFile, File
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import tritonclient.http as httpclient
import io

app = FastAPI()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

client = httpclient.InferenceServerClient(url="localhost:8000")

@app.post("/infer/")
async def infer(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        image_tensor = transform(img)
        transformed_img = image_tensor.numpy().astype(np.float32)
        transformed_img = np.expand_dims(transformed_img, axis=0)  # Add batch dimension
        
        inputs = httpclient.InferInput("data_0", transformed_img.shape, datatype="FP32")
        inputs.set_data_from_numpy(transformed_img, binary_data=True)
        
        outputs = httpclient.InferRequestedOutput("fc6_1", binary_data=True)
        
        results = client.infer(model_name="densenet_onnx", inputs=[inputs], outputs=[outputs])
        inference_output = results.as_numpy('fc6_1')
        
        return {"prediction": inference_output.tolist()[:5]}  # Return top 5 results
    except Exception as e:
        return {"error": str(e)}
