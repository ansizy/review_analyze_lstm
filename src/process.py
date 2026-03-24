import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.tokenizer import JiebaTokenizer


def process():
    print("开始数据处理!")

    # 数据读取
    df = pd.read_csv(config.RAW_DATA_DIR / "online_shopping_10_cats.csv",
                     usecols=["label", "review"],
                     encoding="utf-8").dropna().sample(frac=0.1)

    # print(df.head())

    # 划分数据集
    # stratify 根据 label 分层抽样
    train_df, test_df = train_test_split(df, train_size=0.8, stratify=df["label"])

    # 构建词表
    JiebaTokenizer.build_vocab(train_df["review"].to_list(), config.MODELS_DIR / "vocab.txt")

    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / "vocab.txt")

    # 计算序列长度 res = 116.0
    # print(train_df["review"].apply(lambda x: len(tokenizer.tokenize(x))).quantile(0.95))

    # 构建训练集 保存训练集
    train_df["review"] = train_df["review"].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN))
    train_df.to_json(config.PROCESSED_DATA_DIR / "train.json", orient="records", lines=True)

    # 构建测试集 保存测试集
    test_df["review"] = test_df["review"].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN))
    test_df.to_json(config.PROCESSED_DATA_DIR / "test.json", orient="records", lines=True)

    print("数据处理完成!")



if __name__ == '__main__':
    process()