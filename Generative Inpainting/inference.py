import torch
from torchvision import transforms
from PIL import Image
import os

from models.generator import AOTGenerator as Generator  
from utils.image_utils import save_tensor_image

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_image_with_mask(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = transform(image)
    mask = (image == 0).all(dim=0, keepdim=True).float()
    masked = image * (1 - mask)
    input_tensor = torch.cat([masked, mask], dim=0).unsqueeze(0)
    return input_tensor.to(device), image.unsqueeze(0)

model = Generator().to(device)
model.load_state_dict(torch.load("checkpoints/generator_epoch_100.pth", map_location=device))
model.eval()

img_path = "test_images/sample.jpg"
input_tensor, original = load_image_with_mask(img_path)

with torch.no_grad():
    output = model(input_tensor)

os.makedirs("outputs", exist_ok=True)
save_tensor_image(output, "outputs/inpainted_sample.png")
