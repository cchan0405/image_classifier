import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import timm
import matplotlib.pyplot as plt

# transform settings
transformation = transforms.Compose([transforms.Resize([128,128]), transforms.ToTensor()])
pathname = "/Users/chesterchan/Downloads/Cofilin1KD"
raw_data = ImageFolder(root = pathname, transform=transformation)

# img, lbl = raw_data[10] (need to __getitem__)
class DataCharacteristic(Dataset):
    def __init__(self, raw_data):
        self.data = raw_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]
    
processed_images = DataCharacteristic(raw_data)

# loader
loader = DataLoader(processed_images, batch_size= 4, shuffle=True)

# model
# final output layer (control/experimental)
output_layer = 2 

class Machine(nn.Module):
    def __init__ (self, output_layer):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0',pretrained=True)
        self.nofinallayer = nn.Sequential(*list(self.model.children())[:-1])
        self.finallayer = nn.Linear(self.model.classifier.in_features, output_layer)
        
    def forward(self,x):
        x = self.nofinallayer(x)
        x = self.finallayer(x)
        x = x.flatten(1)
        return x
    
machine = Machine(output_layer)

# optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(machine.parameters(), lr = 1e-5)

epoch_num = 10
avg_loss_epoch = []
epoch_runs = []
epoch_start = 0
val_running_loss = 0.0
avg_val_loss_epoch = []
for epoch in range(epoch_num):
    machine.train()
    running_loss = 0.0

    for img, labels in loader:
        optimizer.zero_grad()
        outputs = machine(img)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        val_running_loss += loss.item()
    epoch_start += 1
    epoch_runs.append(epoch_start)

    epoch_loss=running_loss/len(loader)
    avg_loss_epoch.append(epoch_loss)

    machine.eval()
    torch.no_grad
    val_epoch_loss = val_running_loss / len(loader)
    avg_val_loss_epoch.append(val_epoch_loss)

# print(avg_loss_epoch)
# print(epoch_runs)
    
plt.figure(figsize=(12, 4))
plt.plot(epoch_runs, avg_loss_epoch)
plt.title("Average Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)

plt.figure(figsize=(12, 4))
plt.plot(epoch_runs, avg_val_loss_epoch)
plt.title("Average Validation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)

plt.show()