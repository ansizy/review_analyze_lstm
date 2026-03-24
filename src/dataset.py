import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src import config


class ReviewAnalyzeDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, orient='records', lines=True).to_dict(orient='records')

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.data[idx]['review'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[idx]['label'], dtype=torch.float32)
        return input_tensor, target_tensor

    def __len__(self):
        return len(self.data)

def get_dataloader(train=True):

    if train:
        dataset = ReviewAnalyzeDataset(config.PROCESSED_DATA_DIR / "train.json")
    else:
        dataset = ReviewAnalyzeDataset(config.PROCESSED_DATA_DIR / "test.json")

    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=train)
    return dataloader

if __name__ == '__main__':
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader))
    print(len(test_dataloader))

    for inputs, targets in train_dataloader:
        print(inputs.shape)
        print(targets.shape)
        break
