import torch

from src import config
from src.model import ReviewAnalyzeModel
from src.tokenizer import JiebaTokenizer


def predict_batch(inputs, model):
    model.eval()
    with torch.no_grad():
        output = model(inputs)  # output shape [batch_size]
        batch_result = torch.sigmoid(output)

    return batch_result.tolist()




def predict(user_input, model, tokenizer, device):
    # 1 处理输入
    indexes = tokenizer.encode(user_input)
    input_tensor = torch.tensor([indexes], dtype=torch.long).to(device)

    # 2 预测
    result_batch = predict_batch(input_tensor, model)

    return result_batch[0]



def run_predict():
    # 1 device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2 词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / "vocab.txt")
    print("tokenizer loaded")

    # 3 模型
    model = ReviewAnalyzeModel(tokenizer.vocab_size, tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / "model.pt", map_location=device))
    print("model loaded")

    print("欢迎使用情感分析模型, 请输入句子(输入 q 或 quit 退出)")
    while True:
        user_input = input("> ")
        if user_input in ["q", "quit"]:
            break
        if user_input == "":
            print("请输入内容")
            continue

        result = predict(user_input, model, tokenizer, device)

        if result > 0.5:
            print(f"正向, 置信度: {result}")
        else:
            print(f"负向, 置信度: {1 - result}")


if __name__ == '__main__':
    run_predict()