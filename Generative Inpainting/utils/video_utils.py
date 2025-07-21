import os
import cv2
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms


def video_to_frames(video_path, output_folder, size=(256, 256)):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frame_path = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")


def frames_to_video(frames_folder, output_path, fps=24):
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".png")])
    if not frame_files:
        print("No frames found.")
        return

    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for file in frame_files:
        frame = cv2.imread(os.path.join(frames_folder, file))
        out.write(frame)

    out.release()
    print(f"Saved video to {output_path}")


def inpaint_frames(input_folder, output_folder, model, device, transform=None):
    os.makedirs(output_folder, exist_ok=True)
    model.eval()

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    with torch.no_grad():
        for filename in tqdm(sorted(os.listdir(input_folder))):
            if not filename.endswith(".png"):
                continue

            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            # Detect black pixels as mask (0,0,0)
            mask = (img_tensor == 0).all(dim=1, keepdim=True).float()
            input_tensor = torch.cat([img_tensor, mask], dim=1)

            output = model(input_tensor).cpu()
            out_img = transforms.ToPILImage()(output.squeeze(0).clamp(0, 1))
            out_img.save(os.path.join(output_folder, filename))
