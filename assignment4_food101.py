import os
import tempfile
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import Food101
from torch.quantization import quantize_dynamic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 101)
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = Food101(root="./data", split="train", download=True, transform=transform)
test_dataset = Food101(root="./data", split="test", download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

@torch.no_grad()
def accuracy(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total * 100

fp32_acc_no_fine = accuracy(model, test_loader, device)
print(f"1) Test Accuracy of pre-trained model without any fine tuning: {fp32_acc_no_fine:.2f}%")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

model.train()
running_loss = 0.0
for X, y in train_loader:
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
print(f"Training loss after 1 epoch: {running_loss / len(train_loader):.4f}")

fp32_acc = accuracy(model, test_loader, device)
print(f"2) Test accuracy of pre-trained model after fine-tuning and before quantization: {fp32_acc:.2f}%")

def model_size_mb(net, fname):
    torch.save(net.state_dict(), fname)
    return os.path.getsize(fname) / 1_000_000

fp32_tmp = tempfile.NamedTemporaryFile(delete=False)
fp32_size = model_size_mb(model.cpu(), fp32_tmp.name)
print(f"3) Model size before quantization: {fp32_size:.2f} MB")

torch.backends.quantized.engine = 'qnnpack'
quantised = quantize_dynamic(model.cpu(), {nn.Linear}, dtype=torch.qint8)

int8_acc = accuracy(quantised, test_loader, device="cpu")
print(f"4) Test accuracy of pre-trained model after fine-tuning and after quantization: {int8_acc:.2f}%")

int8_tmp = tempfile.NamedTemporaryFile(delete=False)
int8_size = model_size_mb(quantised, int8_tmp.name)
print(f"5) Model size after INT8 quantization: {int8_size:.2f} MB")

print(f"6) Memory saving after Quantization: {(1 - int8_size / fp32_size) * 100:.2f}%")
print(f"7) Accuracy drop after Quantization: {(fp32_acc - int8_acc):.2f} percentage points")

def measure_latency(model, name):
    model.eval()
    total_time = 0.0
    count = 0
    for X, _ in test_loader:
        if count >= 10:
            break
        X = X.to("cpu")
        start = time.time()
        _ = model(X)
        end = time.time()
        total_time += (end - start)
        count += 1
    avg_latency = total_time / count * 1000
    print(f"8) Average inference latency ({name}): {avg_latency:.2f} ms")

measure_latency(model.cpu(), "Before Quantization")
measure_latency(quantised, "After Quantization")
