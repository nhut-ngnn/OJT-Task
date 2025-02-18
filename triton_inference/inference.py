import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import tritonclient.http as httpclient

image_path = "C:/Users/minhn/Documents/Triton_Inference/img1.jpg"
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

image_tensor = transform(image)

transformed_img = image_tensor.numpy().astype(np.float32)  
client = httpclient.InferenceServerClient(url="localhost:8000")
inputs = httpclient.InferInput("data_0", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

outputs = httpclient.InferRequestedOutput("fc6_1", binary_data=True, class_count=1000)

results = client.infer(model_name="densenet_onnx", inputs=[inputs], outputs=[outputs])

inference_output = results.as_numpy('fc6_1').astype(str)

print(np.squeeze(inference_output)[:5])
