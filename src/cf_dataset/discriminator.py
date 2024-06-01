import argparse

import torch
from datasets import load_dataset, tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

torch.set_default_device("cuda")
torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser(description="This script train a discriminator that "
                                             "will distinguish human from machine generated code.")
parser.add_argument(
    "--dataset_name",
    type=str,
    default="stojchet/test_ds",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=128,
)
parser.add_argument(
    "--optim_lr",
    type=float,
    default=1e-4,
)

MAX_FEATURE_SIZE = 10000


class CodeBert:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")

        special_tokens_dict = {"cls_token": "<CLS>", 'pad_token': '[PAD]', }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def get_sentence_embedding(self, sentence: str) -> torch.Tensor:
        encoded = self.tokenizer.encode(sentence)
        tokenized_sent = torch.tensor(encoded)

        max_size = 512
        final = []
        while True:
            embeddings = self.model(tokenized_sent[:max_size].unsqueeze(0))
            embeddings = embeddings.last_hidden_state[0][0]
            final += embeddings
            if len(tokenized_sent) < max_size:
                break
            tokenized_sent = tokenized_sent[max_size:]

        return torch.tensor(final)


class MLP(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int, out: int = 1) -> None:
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out)
        )

    def forward(self, batch) -> torch.Tensor:
        return self.layers(batch)


class Discriminator(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super(Discriminator, self).__init__()
        self.mlp = MLP(embedding_size=MAX_FEATURE_SIZE, hidden_size=hidden_size)

    def forward(self, inputs):
        features, attention_masks = inputs
        output_y0 = self.mlp(features[:, 0])
        output_y1 = self.mlp(features[:, 1])

        return torch.tanh(output_y0 - output_y1)

    # model state dict
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, hidden_size: int) -> 'Discriminator':
        model = Discriminator(hidden_size)
        return model.load_state_dict(torch.load(path))

    # whole model
    def save(self, path: str):
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> 'Discriminator':
        return torch.load(path)


def train(model, train_loader, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs in tqdm(train_loader):
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, torch.zeros_like(outputs))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(0, f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")


def collate_fn(batch):
    code_bert = CodeBert()

    dataset = []

    for i, element in enumerate(batch):
        x, y_0, y_1 = element["func_documentation_string"], element["func_code_string"], element["prediction"]

        x_y0 = x + y_0
        x_y1 = x + y_1

        x_y0_enc = code_bert.get_sentence_embedding(x_y0)
        x_y1_enc = code_bert.get_sentence_embedding(x_y1)

        dataset.append((x_y0_enc, x_y1_enc))

    # max_len = max([max(len(elem[0]), len(elem[1])) for elem in dataset])
    max_len = MAX_FEATURE_SIZE

    features = torch.zeros((len(dataset), 2, max_len))
    for i, (f1, f2) in enumerate(dataset):
        features[i, 0, :len(f1)] = torch.tensor(f1)
        features[i, 1, :len(f2)] = torch.tensor(f2)

    attention_mask = torch.zeros((len(dataset), 2, max_len))

    original_lengths = torch.tensor([(len(seq[0]), len(seq[1])) for seq in dataset], dtype=torch.float)
    indices = torch.arange(max_len).unsqueeze(0).unsqueeze(1)

    mask = indices < original_lengths.unsqueeze(2)
    attention_mask[mask] = 1

    return features, attention_mask


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = load_dataset(args.dataset_name, split="train")

    train_loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=args.batch_size)

    discriminator = Discriminator(args.hidden_size)
    train(discriminator, train_loader, args.epochs, args.optim_lr)
