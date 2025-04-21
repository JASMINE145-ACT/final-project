import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
import random
from torch.utils.data import Subset
# Download latest version
path = kagglehub.dataset_download("msambare/fer2013")
print("Path to dataset files:", path)

dataset_path = r"C:\Users\m1774\.cache\kagglehub\datasets\msambare\fer2013\versions\1"
#Compose 是 组合多个图像预处理步骤 的工具 它会把括号里的所有变换 按顺序一个个应用到图像上。
#原始图像（PIL）是 48 x 48，每个像素是 0~255 的整数。用 ToTensor() 后，会变成 PyTorch 张量，shape 是 (1, 48, 48)，每个像素是 0~1 的小数。
processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
perturbation_transform = transforms.Compose([
    transforms.GaussianBlur(3),
    transforms.ColorJitter(brightness=0.3),
])
class CustomDatasetWithProcessor(ImageFolder):
    def __init__(self, root, perturb_transform=None, processor=None):
        super().__init__(root)
        self.perturb_transform = perturb_transform
        self.processor = processor

# 改写 __getitem__：只返回扰动后的 PIL 图像
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)
        if self.perturb_transform:
            image = self.perturb_transform(image)
        return image, label  # 不再提前做 processor


train_dataset = CustomDatasetWithProcessor(
    root=dataset_path + "/train",
    perturb_transform=perturbation_transform,
    processor=processor
)
test_dataset = CustomDatasetWithProcessor(
    root=dataset_path + "/test",
    perturb_transform=perturbation_transform,  # 也可不加扰动
    processor=processor
)
train_indices = list(range(len(train_dataset)))
random.shuffle(train_indices)
small_train_dataset = Subset(train_dataset, train_indices[:len(train_indices)//10])
#transform=...：对每张图片应用的预处理操作（如 ToTensor() 等）
train_loader = DataLoader(small_train_dataset, batch_size=64, shuffle=True,collate_fn=lambda batch: tuple(zip(*batch)))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,collate_fn=lambda batch: tuple(zip(*batch)))
#把数据集装到一个“数据批发商”里，训练的时候按 64 个一组送过来。训练时打乱顺序，测试时按顺序。
print("训练集中类别：", train_dataset.classes)
label_counts = Counter(train_dataset.targets)
#train_dataset.targets 是一个 标签列表，形如 [3, 0, 1, 2, 3, 0, 3, ...]，表示每张图对应的情绪类别编号。
#Counter(...) 会返回一个字典结构，统计每个类别编号出现了多少次。
print("训练集中各类样本数：")
for label, count in label_counts.items():
    print(f"类 {label}: {count} 张图像")
print(test_dataset.class_to_idx)
import os
model_path = r"C:\Users\m1774\Desktop\507\my_finetuned_model"
# 如果目录不存在就创建它
os.makedirs(model_path, exist_ok=True)
if os.path.exists(model_path):
    print("📦 发现已有 fine-tuned 模型，直接加载跳过训练")
    model = AutoModelForImageClassification.from_pretrained(model_path,local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(model_path,local_files_only=True)
else:
    processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
    model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")
# 加载模型和处理器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()
# Fine-tune
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        # 1. 转成 RGB PIL
        pil_batch = [img.convert("RGB") for img in images]

        # 2. Processor 处理
        inputs = processor(images=pil_batch, return_tensors="pt", padding=True).to(device)
        labels = torch.tensor(labels).to(device)

        # 3. 前向 + loss
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)

        # 4. 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(train_loader):.4f}")
if epoch == num_epochs - 1:
    model.save_pretrained(model_path)
    processor.save_pretrained(model_path)  # 保存处理器一起用

model.eval()
to_pil = ToPILImage()
y_true, y_pred = [], []

print(">>> 开始 batch 推理...")

for batch_imgs, batch_labels in tqdm(test_loader, desc="Batch inference"):
    # 1. 把 Tensor 图像转换成 RGB PIL（列表）
    pil_batch = [img.convert("RGB") for img in batch_imgs]

    # 2. 用 Hugging Face 的 processor 做 batch 处理
    inputs = processor(images=pil_batch, return_tensors="pt").to(device)
    

    # 3. 模型前向传播
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=-1).cpu().tolist()

    # 4. 收集标签和预测
# 收集标签和预测
    y_true.extend(batch_labels.detach().cpu().numpy().tolist())
y_pred.extend(preds)


print(">>> 推理完成，开始计算准确率")

# 分类指标输出
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")
print(classification_report(y_true, y_pred, target_names=labels))


