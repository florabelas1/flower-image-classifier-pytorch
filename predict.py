import argparse
import torch
import json
import numpy as np
from collections import OrderedDict
from torchvision import models
from PIL import Image

def load_checkpoint(filepath, arch):
    checkpoint = torch.load(filepath, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        input_size = 25088
        classifier = torch.nn.Sequential(OrderedDict([
            ("fc1", torch.nn.Linear(input_size, 4096)),
            ("relu", torch.nn.ReLU()),
            ("dropout", torch.nn.Dropout(0.2)),
            ("fc2", torch.nn.Linear(4096, 102)),
            ("output", torch.nn.LogSoftmax(dim=1))
        ]))
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
        input_size = 1024
        classifier = torch.nn.Sequential(OrderedDict([
            ("fc1", torch.nn.Linear(input_size, 512)),
            ("relu", torch.nn.ReLU()),
            ("dropout", torch.nn.Dropout(0.2)),
            ("fc2", torch.nn.Linear(512, 102)),
            ("output", torch.nn.LogSoftmax(dim=1))
        ]))
    else:
        raise ValueError("Unsupported model architecture. Choose vgg16 or densenet121.")

    model.classifier = classifier

    # Fix state_dict key mismatches
    new_state_dict = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        new_key = key.replace("classifier.fc1", "classifier.0").replace("classifier.fc2", "classifier.3")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=False)
    model.class_to_idx = checkpoint["class_to_idx"]
    
    return model

def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))

    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    
    return torch.tensor(np_image, dtype=torch.float32)

def predict(image_path, model, top_k=5, category_names=None, gpu=False):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    image = process_image(image_path).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logps = model(image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(top_k, dim=1)
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i.item()] for i in top_class[0]]
    
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_classes = [cat_to_name.get(c, f"Class {c}") for c in top_classes]
    
    return top_p.cpu().numpy().flatten(), top_classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained deep learning model.")
    
    parser.add_argument("image_path", type=str, help="Path to the image.")
    parser.add_argument("checkpoint", type=str, help="Path to the trained model checkpoint.")
    parser.add_argument("--arch", type=str, default="vgg16", choices=["vgg16", "densenet121"], help="Model architecture to use (vgg16 or densenet121).")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes.")
    parser.add_argument("--category_names", type=str, help="Path to category mapping JSON file.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference.")

    args = parser.parse_args()
    
    model = load_checkpoint(args.checkpoint, args.arch)
    probs, classes = predict(args.image_path, model, args.top_k, args.category_names, args.gpu)
    
    print("\nTop probabilities:", probs)
    print("Predicted classes:", classes)
