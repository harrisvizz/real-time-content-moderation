
import torch
from torch.utils.data import Dataset

class ToxicCommentDataset(Dataset):

    def __init__(self, dataframe):
        self.texts = dataframe["comment_text"].tolist()

        self.labels = dataframe[
            [
                "toxic",
                "severe_toxic",
                "obscene",
                "threat",
                "insult",
                "identity_hate"
            ]
        ].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        labels = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return text, labels
