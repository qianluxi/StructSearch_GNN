import torch
from torch_geometric.loader import DataLoader

from datasets.structure_dataset import StructureDataset
from models.model import GraphEncoder
from training.split_dataset import split_dataset


def build_dataset(file_list):

    dataset = []

    for path in file_list:

        ds = StructureDataset(path)

        dataset.append(ds.data)

    return dataset


def train():

    dxf_dir = "data/dxf"

    train_files, val_files = split_dataset(dxf_dir)

    train_dataset = build_dataset(train_files)
    val_dataset = build_dataset(val_files)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = GraphEncoder()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    for epoch in range(50):

        model.train()

        total_loss = 0

        for batch in train_loader:

            batch = batch.to(device)

            emb = model(batch.x, batch.edge_index, batch.batch)

            loss = torch.mean(emb ** 2)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} loss {total_loss:.4f}")

    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    train()
