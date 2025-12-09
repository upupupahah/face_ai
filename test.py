import torch
from PIL import Image, ImageDraw
import os
import torchvision.transforms as T
import numpy as np

from model import FaceDetector


def cxcywh_to_xyxy_norm(cx, cy, w, h):
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return [x1, y1, x2, y2]


def xyxy_norm_to_pixels(box, img_w, img_h):
    x1 = int(round(box[0] * img_w))
    y1 = int(round(box[1] * img_h))
    x2 = int(round(box[2] * img_w))
    y2 = int(round(box[3] * img_h))
    x1 = max(0, min(x1, img_w - 1))
    x2 = max(0, min(x2, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    y2 = max(0, min(y2, img_h - 1))
    return [x1, y1, x2, y2]


def run_inference(model_path, data_root, out_dir, device, img_size=256):
    device = torch.device(device)
    model = FaceDetector()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

    # Предпочитаем тестовую папку data/test/images или data/test, затем обычные data/images
    img_dir_candidates = [os.path.join(data_root, 'test', 'images'), os.path.join(data_root, 'test'), os.path.join(data_root, 'images'), data_root]
    img_paths = []
    for d in img_dir_candidates:
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_paths.append(os.path.join(d, fn))
    img_paths.sort()

    if not img_paths:
        print('No images found in', data_root)
        return

    os.makedirs(out_dir, exist_ok=True)

    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        inp = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp)[0].cpu().numpy()
        # convert
        p_xyxy = cxcywh_to_xyxy_norm(pred[0], pred[1], pred[2], pred[3])
        p_pixels = xyxy_norm_to_pixels(p_xyxy, orig_w, orig_h)

        # draw
        draw = ImageDraw.Draw(img)
        draw.rectangle(p_pixels, outline='red', width=2)
        base = os.path.basename(img_path)
        out_path = os.path.join(out_dir, base)
        img.save(out_path)
        print('Saved', out_path)


if __name__ == '__main__':
    model_path = 'best_model.pt'
    data_root = 'boss_test'
    out_dir = 'out_boss'
    device = 'cuda'

    run_inference(model_path, data_root, out_dir, device=device)
