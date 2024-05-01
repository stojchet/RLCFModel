import argparse

import torch
from datasets import load_dataset
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, pipeline

torch.set_default_device("cuda")
torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser(description="This script fine tunes a model with SFT.")
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
    default=1,
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


class CodeBert:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")

        special_tokens_dict = {"cls_token": "<CLS>", 'pad_token': '[PAD]', }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.pipe_model = pipeline('feature-extraction', model=self.model, tokenizer=self.tokenizer, truncation=True)

    def get_sentence_embedding(self, sentence: str) -> torch.Tensor:
        return self.pipe_model(sentence)[0][0]


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

    def forward(self, batch):
        return self.layers(batch)


class Discriminator(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super(Discriminator, self).__init__()
        self.mlp = MLP(embedding_size=768, hidden_size=hidden_size)
        self.code_bert = CodeBert()

    def forward(self, inputs):
        x, y_0, y_1 = inputs["func_documentation_string"], inputs["func_code_string"], inputs["prediction"]

        x_y0 = [x_part + y0_part for x_part, y0_part in zip(x, y_0)]
        x_y1 = [x_part + y1_part for x_part, y1_part in zip(x, y_1)]

        x_y0_enc = torch.tensor([self.code_bert.get_sentence_embedding(part) for part in x_y0])
        x_y1_enc = torch.tensor([self.code_bert.get_sentence_embedding(part) for part in x_y1])

        output_y0 = self.mlp(x_y0_enc)
        output_y1 = self.mlp(x_y1_enc)

        return torch.tanh(output_y0 - output_y1)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


def train(model, train_loader, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, torch.zeros_like(outputs))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(0, f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = load_dataset(args.dataset_name, split="train")
    train_loader = DataLoader(dataset, batch_size=args.batch_size)

    discriminator = Discriminator(args.hidden_size)
    train(discriminator, train_loader, args.epochs, args.optim_lr)


