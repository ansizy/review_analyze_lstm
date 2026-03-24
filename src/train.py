import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src import config
from src.dataset import get_dataloader
from src.model import ReviewAnalyzeModel
from src.tokenizer import JiebaTokenizer


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    total_loss = 0
    model.train()
    for inputs, targets in tqdm(dataloader, desc="Training"):
        # targets shape [batch_size]
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs) # output shape [batch_size]
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
    # 1 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2 数据
    dataloader = get_dataloader(train=True)
    # 3 分词器
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / "vocab.txt")
    # 4 模型
    model = ReviewAnalyzeModel(tokenizer.vocab_size, padding_index=tokenizer.pad_token_index).to(device)
    # 5 损失函数
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # 6 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 7 tensorboard
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    best_loss = float("inf")
    for epoch in range(config.EPOCHS):
        print(f"epoch {epoch + 1}")
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"Loss: {loss:.4f}")

        writer.add_scalar("Loss", loss, epoch)

        writer.close()

        # save model
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODELS_DIR / "model.pt")
            print("model saved")


if __name__ == '__main__':
    train()