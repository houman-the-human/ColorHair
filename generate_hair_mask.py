import os
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from model import BiSeNet  # Make sure model.py is in the same folder or PYTHONPATH

# Define transform for BiSeNet input
to_tensor = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_bisenet_model(weight_path='res/79999_iter.pth', n_classes=19):
    net = BiSeNet(n_classes=n_classes)
    net.load_state_dict(torch.load(weight_path, map_location='cpu'))
    net.eval()
    return net

def get_hair_mask(image_path, weight_path='res/79999_iter.pth'):
    # Load and preprocess image
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        original_size = img.size
        image_tensor = to_tensor(img).unsqueeze(0)

# Example usage:
if __name__ == "__main__":
    image_path = "example.jpg"             # change to your image
    save_path = "output/hair_mask.png"     # output path
    weight_path = "res/79999_iter.pth"     # model checkpoint
    save_hair_mask(image_path, save_path, weight_path)    # Predict parsing map
    net = load_bisenet_model(weight_path)
    with torch.no_grad():
        out = net(image_tensor)[0]
        parsing = out.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)

    # Resize parsing map to original image size
    parsing_resized = cv2.resize(parsing, original_size, interpolation=cv2.INTER_NEAREST)

    # Extract hair region (label 17)
    hair_mask = (parsing_resized == 17).astype(np.uint8) * 255  # binary mask: 255 for hair

    return hair_mask

def save_hair_mask(image_path, save_path='hair_mask.png', weight_path='res/79999_iter.pth'):
    hair_mask = get_hair_mask(image_path, weight_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, hair_mask)
    print(f"[✓] Hair mask saved to {save_path}")
