import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

IMG_SIZE = 64
BATCH_SIZE = 64
CLASS_NAMES = ["happy", "sad", "angry", "surprised", "sleepy"]
NUM_CLASSES = len(CLASS_NAMES)


def _jitter(rng, low, high):
    return rng.randint(low, high)


def add_soft_noise(img, rng, amount=35):
    arr = np.array(img).astype(np.int16)
    gaussian = np.random.default_rng(rng.randint(0, 10_000)).normal(0, amount, size=arr.shape)
    arr = np.clip(arr + gaussian, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def draw_emoji(label, img_size=IMG_SIZE, seed=0, hard_mode=False):
    rng = random.Random(seed)

    bg_color = (
        245 + _jitter(rng, -8, 8),
        245 + _jitter(rng, -8, 8),
        250 + _jitter(rng, -8, 5),
    )
    img = Image.new("RGB", (img_size, img_size), bg_color)
    draw = ImageDraw.Draw(img)

    # Background dots
    for _ in range(40 if not hard_mode else 70):
        x = _jitter(rng, 0, img_size - 1)
        y = _jitter(rng, 0, img_size - 1)
        r = _jitter(rng, 1, 2)
        c = 235 + _jitter(rng, -18, 12)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(c, c, c + _jitter(rng, -5, 5)))

    cx = img_size // 2 + _jitter(rng, -3, 3)
    cy = img_size // 2 + _jitter(rng, -3, 3)
    radius = 22 + _jitter(rng, -2, 2)

    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=(252, 215 + _jitter(rng, -12, 8), 70 + _jitter(rng, -10, 8)),
        outline=(120, 90, 20),
        width=2,
    )

    if rng.random() < 0.55:
        cheek_color = (245, 165 + _jitter(rng, -10, 10), 165 + _jitter(rng, -10, 10))
        draw.ellipse((cx - 16, cy + 2, cx - 10, cy + 8), fill=cheek_color)
        draw.ellipse((cx + 10, cy + 2, cx + 16, cy + 8), fill=cheek_color)

    ex = 9 + _jitter(rng, -1, 1)
    ey = 8 + _jitter(rng, -1, 1)

    left_eye = (cx - ex, cy - ey)
    right_eye = (cx + ex, cy - ey)

    def draw_open_eye(center, w=4, h=5):
        x, y = center
        draw.ellipse((x - w // 2, y - h // 2, x + w // 2, y + h // 2), fill=(20, 20, 20))

    def draw_closed_eye(center, width=6):
        x, y = center
        draw.line((x - width // 2, y, x + width // 2, y), fill=(20, 20, 20), width=2)

    def draw_arc_eye(center, start, end, width=2):
        x, y = center
        box = (x - 4, y - 4, x + 4, y + 4)
        draw.arc(box, start=start, end=end, fill=(20, 20, 20), width=width)

    if label == "angry":
        draw.line((left_eye[0] - 5, left_eye[1] - 6, left_eye[0] + 2, left_eye[1] - 8), fill=(80, 40, 10), width=2)
        draw.line((right_eye[0] - 2, right_eye[1] - 8, right_eye[0] + 5, right_eye[1] - 6), fill=(80, 40, 10), width=2)
    elif label == "sad":
        draw.line((left_eye[0] - 5, left_eye[1] - 8, left_eye[0] + 2, left_eye[1] - 6), fill=(80, 40, 10), width=2)
        draw.line((right_eye[0] - 2, right_eye[1] - 6, right_eye[0] + 5, right_eye[1] - 8), fill=(80, 40, 10), width=2)
    elif label == "surprised":
        draw.line((left_eye[0] - 4, left_eye[1] - 8, left_eye[0] + 4, left_eye[1] - 8), fill=(90, 50, 10), width=2)
        draw.line((right_eye[0] - 4, right_eye[1] - 8, right_eye[0] + 4, right_eye[1] - 8), fill=(90, 50, 10), width=2)

    if label == "happy":
        draw_arc_eye(left_eye, 10, 170, width=2)
        draw_arc_eye(right_eye, 10, 170, width=2)
        draw.arc((cx - 11, cy - 1, cx + 11, cy + 15), start=15, end=165, fill=(120, 30, 30), width=3)

    elif label == "sad":
        draw_open_eye(left_eye, 3, 5)
        draw_open_eye(right_eye, 3, 5)
        draw.arc((cx - 11, cy + 8, cx + 11, cy + 20), start=200, end=340, fill=(120, 30, 30), width=3)
        if rng.random() < 0.65:
            tx = left_eye[0] - 2
            ty = left_eye[1] + 6
            draw.ellipse((tx - 2, ty, tx + 2, ty + 6), fill=(100, 180, 255))

    elif label == "angry":
        draw_open_eye(left_eye, 3, 5)
        draw_open_eye(right_eye, 3, 5)
        draw.arc((cx - 10, cy + 7, cx + 10, cy + 19), start=200, end=340, fill=(120, 10, 10), width=3)

    elif label == "surprised":
        draw_open_eye(left_eye, 4, 6)
        draw_open_eye(right_eye, 4, 6)
        mouth_r = 5 + _jitter(rng, -1, 1)
        draw.ellipse((cx - mouth_r, cy + 6, cx + mouth_r, cy + 6 + 2 * mouth_r), outline=(120, 30, 30), width=3)

    elif label == "sleepy":
        draw_closed_eye(left_eye, width=7)
        draw_closed_eye(right_eye, width=7)
        draw.arc((cx - 10, cy + 6, cx + 10, cy + 12), start=5, end=175, fill=(120, 30, 30), width=2)
        if rng.random() < 0.5:
            bx = right_eye[0] + 5
            by = right_eye[1] - 2
            draw.ellipse((bx, by, bx + 5, by + 5), outline=(120, 180, 255), width=2)

    if hard_mode and rng.random() < 0.45:
        y0 = cy + _jitter(rng, -3, 6)
        draw.rectangle((cx - 14, y0, cx + 14, y0 + 3), fill=(230, 230, 230))

    img = img.rotate(_jitter(rng, -10, 10) if not hard_mode else _jitter(rng, -18, 18), fillcolor=bg_color)

    if rng.random() < (0.4 if not hard_mode else 0.75):
        img = add_soft_noise(img, rng, amount=14 if not hard_mode else 24)

    if hard_mode and rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

    return img


def generate_split(n_per_class, base_seed=0, hard_mode=False):
    images, labels = [], []
    seed = base_seed
    for label_idx, label_name in enumerate(CLASS_NAMES):
        for _ in range(n_per_class):
            img = draw_emoji(label_name, img_size=IMG_SIZE, seed=seed, hard_mode=hard_mode)
            images.append(img)
            labels.append(label_idx)
            seed += 1
    return images, labels


def pil_to_tensor(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr, dtype=torch.float32)


def random_augment(img):
    out = img.copy()

    angle = random.uniform(-15, 15)
    out = out.rotate(angle, fillcolor=(245, 245, 250))

    max_shift = 4
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (245, 245, 250))
    canvas.paste(out, (dx, dy))
    out = canvas

    for enhancer_cls, low, high in [
        (ImageEnhance.Brightness, 0.9, 1.1),
        (ImageEnhance.Contrast, 0.9, 1.15),
        (ImageEnhance.Color, 0.9, 1.15),
    ]:
        factor = random.uniform(low, high)
        out = enhancer_cls(out).enhance(factor)

    if random.random() < 0.35:
        out = add_soft_noise(out, random.Random(random.randint(0, 99999)), amount=10)

    return out


class EmojiDataset(Dataset):
    def __init__(self, images, labels, augment=False, mean=None, std=None):
        self.images = images
        self.labels = labels
        self.augment = augment
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.augment:
            img = random_augment(img)

        x = pil_to_tensor(img)

        if self.mean is not None and self.std is not None:
            x = (x - self.mean[:, None, None]) / self.std[:, None, None]

        y = torch.tensor(label, dtype=torch.long)
        return x, y


def compute_channel_stats(images):
    tensors = torch.stack([pil_to_tensor(img) for img in images], dim=0)
    mean = tensors.mean(dim=(0, 2, 3))
    std = tensors.std(dim=(0, 2, 3)).clamp_min(1e-6)
    return mean, std


def get_dataloaders(batch_size=BATCH_SIZE):
    train_imgs, train_labels = generate_split(260, base_seed=1000, hard_mode=True)
    val_imgs, val_labels = generate_split(70, base_seed=5000, hard_mode=True)
    test_imgs, test_labels = generate_split(90, base_seed=9000, hard_mode=True)

    mean, std = compute_channel_stats(train_imgs)

    train_ds = EmojiDataset(train_imgs, train_labels, augment=True, mean=mean, std=std)
    val_ds = EmojiDataset(val_imgs, val_labels, augment=False, mean=mean, std=std)
    test_ds = EmojiDataset(test_imgs, test_labels, augment=False, mean=mean, std=std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, mean, std