import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_charset():
    char_to_idx = {c: i + 1 for i, c in enumerate(ALPHABET)}
    idx_to_char = {i + 1: c for i, c in enumerate(ALPHABET)}
    return char_to_idx, idx_to_char


def normalize_label(s: str) -> str:
    s = str(s).strip().upper()
    return "".join([c for c in s if c in ALPHABET])


def encode(text: str, char_to_idx: dict) -> torch.Tensor:
    ids = [char_to_idx[c] for c in text if c in char_to_idx]
    return torch.tensor(ids, dtype=torch.long)


def greedy_decode(logits: torch.Tensor, idx_to_char: dict) -> str:
    preds = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
    out = []
    prev = None
    for p in preds:
        if p != 0 and p != prev:
            out.append(idx_to_char.get(p, ""))
        prev = p
    return "".join(out)


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_pad(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = out_h / max(h, 1)
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(img, (new_w, out_h), interpolation=cv2.INTER_AREA)

    if new_w >= out_w:
        resized = cv2.resize(resized, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return resized

    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    canvas[:, :new_w] = resized
    return canvas


@dataclass
class Sample:
    image_path: str
    label: str


class IIIT5KWordDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        base_dir: str,
        img_h: int = 32,
        img_w: int = 128,
        limit: int | None = None,
    ):
        df = pd.read_csv(csv_path)
        if "ImgName" not in df.columns or "GroundTruth" not in df.columns:
            raise ValueError("CSV must contain ImgName and GroundTruth columns")

        samples = []
        for _, r in df.iterrows():
            rel = str(r["ImgName"]).strip().replace("\\", "/")
            lab = normalize_label(r["GroundTruth"])
            if not lab:
                continue
            full = str(Path(base_dir) / rel)
            samples.append(Sample(full, lab))

        if limit is not None:
            samples = samples[:limit]

        self.samples = samples
        self.img_h = img_h
        self.img_w = img_w

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = read_gray(s.image_path)
        img = resize_pad(img, self.img_h, self.img_w)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        x = torch.from_numpy(img).unsqueeze(0)
        return x, s.label


def collate(batch, char_to_idx):
    xs = []
    ys = []
    y_lens = []
    for x, lab in batch:
        xs.append(x)
        y = encode(lab, char_to_idx)
        ys.append(y)
        y_lens.append(len(y))

    xs = torch.stack(xs, dim=0)
    ys_cat = torch.cat(ys, dim=0)
    y_lens = torch.tensor(y_lens, dtype=torch.long)
    return xs, ys_cat, y_lens


class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)
        b, c, h, w = f.shape
        f = f.mean(dim=2)
        f = f.permute(2, 0, 1).contiguous()
        y, _ = self.rnn(f)
        y = self.fc(y)
        return y


@torch.no_grad()
def evaluate(model, loader, idx_to_char, device):
    model.eval()
    total = 0
    exact = 0
    for xs, ys_cat, y_lens in loader:
        xs = xs.to(device)
        logits = model(xs)
        t, n, c = logits.shape
        for i in range(n):
            pred = greedy_decode(logits[:, i, :], idx_to_char)
            start = int(y_lens[:i].sum().item())
            end = start + int(y_lens[i].item())
            gt_ids = ys_cat[start:end].tolist()
            gt = "".join(idx_to_char.get(j, "") for j in gt_ids)
            total += 1
            if pred == gt:
                exact += 1
    return exact / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="iiit-5k/traindata.csv")
    ap.add_argument("--test_csv", default="iiit-5k/testdata.csv")
    ap.add_argument("--base_dir", default="iiit-5k/IIIT5K-Word_V3.0/IIIT5K")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train_limit", type=int, default=2000)
    ap.add_argument("--test_limit", type=int, default=1000)
    ap.add_argument("--out_dir", default="outputs/models")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    char_to_idx, idx_to_char = build_charset()
    num_classes = len(char_to_idx) + 1

    train_ds = IIIT5KWordDataset(args.train_csv, args.base_dir, limit=args.train_limit)
    test_ds = IIIT5KWordDataset(args.test_csv, args.base_dir, limit=args.test_limit)

    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate(b, char_to_idx),
    )
    test_ld = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate(b, char_to_idx),
    )

    model = CRNN(num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for xs, ys_cat, y_lens in train_ld:
            xs = xs.to(device)
            ys_cat = ys_cat.to(device)

            logits = model(xs)
            log_probs = logits.log_softmax(dim=-1)

            t, n, _ = log_probs.shape
            input_lens = torch.full((n,), t, dtype=torch.long, device=device)

            loss = ctc(log_probs, ys_cat, input_lens, y_lens.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            steps += 1

        acc = evaluate(model, test_ld, idx_to_char, device)
        avg_loss = total_loss / max(steps, 1)
        print(f"epoch={epoch} loss={avg_loss:.4f} exact_word_acc={acc:.4f}")

    out_path = Path(args.out_dir) / "crnn_iiit5k.pt"
    torch.save({"model": model.state_dict()}, out_path)
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
