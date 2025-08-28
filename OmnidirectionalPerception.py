import onnxruntime as ort
import numpy as np
from PIL import Image

# 1. 读取模型
session = ort.InferenceSession("test.onnx", providers=["CPUExecutionProvider"])

# 2. 读图 & 预处理 (假设输入是 [N, H, W, C] 格式)
img = Image.open(r"C:\Users\28921\Desktop\6A332CDB29BB8780A1104C5E88D3E224.jpg").resize((640,384))
arr = np.array(img).astype(np.float32) / 255.0
arr = np.expand_dims(arr, 0)   # batch = 1

# 3. 获取模型输入名 (你的模型里可能是 'input')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 4. 推理
outputs = session.run([output_name], {input_name: arr})
print(outputs[0].shape)
