import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class ImageInpaintingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

      
        mask = (image.mean(dim=0, keepdim=True) < 0.5).float()
        masked_image = image * (1 - mask)

        input_tensor = torch.cat([masked_image, mask], dim=0)
        return input_tensor, image
