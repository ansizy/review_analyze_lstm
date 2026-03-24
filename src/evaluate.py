import torch
from tqdm import tqdm

from src import config
from src.dataset import get_dataloader
from src.model import ReviewAnalyzeModel
from src.predict import predict_batch
from src.tokenizer import JiebaTokenizer


def evaluate(model, test_dataloader, device):
    total_count = 0
    correct_count = 0
    for inputs, targets in tqdm(test_dataloader, desc="Evaluating"):
        inputs = inputs.to(device)
        targets = targets.tolist()

        batch_result = predict_batch(inputs, model)

        for result, target in zip(batch_result, targets):
            result = 1 if result > 0.5 else 0
            if result == target:
                correct_count += 1
            total_count += 1

    return correct_count / total_count




def run_evaluate():
    # 1 device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2 词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / "vocab.txt")
    print("tokenizer loaded")

    # 3 模型
    model = ReviewAnalyzeModel(tokenizer.vocab_size, tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / "model.pt", map_location=device))
    print("model loaded")

    # 4 test 数据集
    test_dataloader = get_dataloader(train=False)

    # 5 评估
    acc = evaluate(model, test_dataloader, device)
    print("评估结果")
    print(f"acc: {acc}")




if __name__ == '__main__':
    run_evaluate()