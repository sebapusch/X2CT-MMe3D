import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

# Paths
INPUT_DIR = "..."
OUTPUT_DIR = "..."
TARGET_SIZE = (512, 512)  # Can be changed to (512, 512)
SAVE_AS_PT = False  # Save as .pt instead of .png

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the preprocessing pipeline
transform = transforms.Compose([
    transforms.CenterCrop(2048),  # Adjust if images are smaller
                                  # crop images using fixed crop strategy (better with bounding box)
    transforms.Resize(TARGET_SIZE), # Resize to image to TARGET_SIZE
    transforms.Grayscale(),  # Ensure 1 channel
    transforms.ToTensor(),  # normalize from [0, 255] to [0, 1]
])

def preprocess_image(filepath):
    """
    Load, crop, resize, and save an X-ray image.
    """
    img = Image.open(filepath).convert("L")  # Ensure grayscale
    tensor = transform(img).type(torch.float32)
    
    filename = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(OUTPUT_DIR, filename + ".png")

    print(f"Saving to: {out_path}")
    
    img_out = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    Image.fromarray(img_out).save(out_path)

# Collect all PNG image files
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".png")]

i = 0
# Process all images
for fname in tqdm(image_files, desc="Preprocessing X-rays"):
    if i == 50:
        break
    preprocess_image(os.path.join(INPUT_DIR, fname))
    i += 1

print(f"✅ Done. Preprocessed images saved to: {OUTPUT_DIR}")
