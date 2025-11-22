import torch
import argparse
import json
import time
from torchvision import datasets, transforms, models
from collections import OrderedDict
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser(description="Train a deep learning model for image classification.")
parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'alexnet', 'resnet18'])
parser.add_argument('--data_path', type=str, default='flowers')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--hidden', type=int, default=4096)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(args.data_path + '/train', transform=train_transforms)
valid_dataset = datasets.ImageFolder(args.data_path + '/valid', transform=valid_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)

if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif args.arch == 'alexnet':
    model = models.alexnet(pretrained=True)
elif args.arch == 'resnet18':
    model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, args.hidden)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(0.2)),
    ('fc2', nn.Linear(args.hidden, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))

if args.arch in ['vgg16', 'alexnet']:
    model.classifier = classifier
elif args.arch == 'resnet18':
    model.fc = classifier

model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

epochs = args.epochs
best_val_loss = float('inf')
save_path = 'best_model.pth'

print("Undergoing training!\n", flush=True)

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    val_loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            val_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train loss: {running_loss/len(train_loader):.3f} | "
          f"Validation loss: {val_loss/len(valid_loader):.3f} | "
          f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint = {
            'state_dict': model.state_dict(),
            'class_to_idx': train_dataset.class_to_idx,
            'arch': args.arch,
            'hidden_units': args.hidden,
            'epochs': epochs,
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(checkpoint, save_path)
        print(f"Best model saved to {save_path}")

print("\nTraining completed!")
