import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, LlavaProcessor

# 1. 加载处理器和模型（以llava-hf为例）
processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto"
)

# 2. 加载一张图片
image = Image.open("your_image.jpg")

# 3. 输入你的问题
question = "这张图片里有几只猫？"

# 4. 模型推理
inputs = processor(text=question, images=image, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(output[0], skip_special_tokens=True)

print("LLaVA回答:", response)