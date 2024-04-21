import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from model2 import MobileNetV2  # 导入修改后的MobileNetV2模型类

# Load pre-trained MobileNetV2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cuda")
model = MobileNetV2(num_classes=38).to(device)  # 实例化修改后的MobileNetV2模型
model.load_state_dict(torch.load("mobilenetv2_3_model.pth", map_location=device))  # 加载权重到模型
model.eval()

# Load student data
students_data = pd.read_excel("students.xlsx")

# Load and preprocess image
image_path = "cs/cs-0.jpg"
image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_tensor = preprocess(image).unsqueeze(0).to(device)

# Perform inference
with torch.no_grad():
    outputs = model(image_tensor)
_, predicted = torch.max(outputs, 1)
class_index = predicted.item()

# Print result
if class_index < len(students_data):
    student_info = students_data.iloc[class_index]
    name = student_info['姓名']
    student_id = student_info['学号']
    print(f"识别结果：{name} ({student_id})")
else:
    print("查无此人")
