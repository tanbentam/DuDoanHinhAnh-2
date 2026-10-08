"""
BASELINE — MLP đơn giản cho bài tập lớn CIFAR-10.

Đây là điểm xuất phát CỐ Ý ĐƠN GIẢN: SGD thuần, learning rate cố định,
khởi tạo mặc định, không regularization, không residual.
Nhiệm vụ của nhóm là cải thiện nó bằng các kỹ thuật đã học (và hơn thế).

Chạy:
    python baseline.py
Kết quả:
    submissions/baseline.csv   (định dạng nộp bài: id,label)
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "submissions"
SEED = 42
EPOCHS = 10
BATCH_SIZE = 128
LR = 0.01
VAL_SIZE = 5000


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data():
    train = np.load(DATA_DIR / "train.npz")
    test = np.load(DATA_DIR / "test.npz")
    # Ảnh (N, 32, 32, 3) uint8 -> vector 3072 chiều trong [0, 1]
    x = train["images"].reshape(-1, 3072).astype(np.float32) / 255.0
    y = train["labels"].astype(np.int64)
    x_test = test["images"].reshape(-1, 3072).astype(np.float32) / 255.0
    return x, y, x_test, test["ids"]


class MLP(nn.Module):
    def __init__(self, in_dim=3072, hidden=256, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += criterion(logits, yb).item()
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict(model, x, device, batch_size=1024):
    model.eval()
    preds = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).to(device)
        preds.append(model(xb).argmax(1).cpu())
    return torch.cat(preds).numpy()


def main():
    set_seed(SEED)
    device = get_device()
    x, y, x_test, test_ids = load_data()

    # Tách tập validation từ dữ liệu có nhãn (KHÔNG dùng test để chọn mô hình).
    perm = np.random.permutation(len(x))
    val_idx, tr_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]
    train_ds = TensorDataset(torch.from_numpy(x[tr_idx]), torch.from_numpy(y[tr_idx]))
    val_ds = TensorDataset(torch.from_numpy(x[val_idx]), torch.from_numpy(y[val_idx]))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024)

    model = MLP().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Thiết bị: {device} | Số tham số: {n_params:,}")

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        tr_loss, tr_acc = evaluate(model, train_loader, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.1f}s")

    preds = predict(model, x_test, device)
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "baseline.csv"
    with open(out, "w") as f:
        f.write("id,label\n")
        for i, p in zip(test_ids, preds):
            f.write(f"{i},{p}\n")
    print(f"Đã ghi file nộp bài: {out}")


if __name__ == "__main__":
    main()
