import jieba
from tqdm import tqdm

import config


class JiebaTokenizer:
    unk_token = "<unk>"
    pad_token = "<pad>"
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)

        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}

        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]

    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)

    def encode(self, text, seq_len=config.SEQ_LEN):
        tokens = self.tokenize(text)

        # 截取 or 填充
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        elif len(tokens) < seq_len:
            tokens = tokens + [self.pad_token] * (seq_len - len(tokens))

        indexes = []
        for token in tokens:
            indexes.append(self.word2index.get(token, self.unk_token_index))
        return indexes

    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        """
        创建词表并保存
        :param sentences:
        :param vocab_path:
        :return: no return value
        """
        vocab_set = set()
        for sentence in tqdm(sentences, desc="Building vocabulary"):
            vocab_set.update(jieba.lcut(sentence))

        # vocab_set 中去除空格
        # vocab_list = []
        # for token in vocab_set:
        #     if token.strip() != '':
        #         vocab_list.append(token)
        # 加入特殊token
        # vocab_list = [cls.unk_token, cls.pan_token] + vocab_list
        vocab_list = [cls.pad_token, cls.unk_token] +  [token for token in vocab_set if token.strip() != ""]

        # 保存词表
        with open(vocab_path, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_path):
        vocab_list = []
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                vocab_list.append(line.strip())
        return cls(vocab_list)
