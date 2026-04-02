# Imports
import torch
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def load_and_preprocess_data(batch_size, seed=45):
    ## Load dataset and convert labels to 0-based indexing
    covtype_data = fetch_covtype(random_state=seed)
    x, y = covtype_data.data, covtype_data.target - 1

    # --- Preprocessing ---
    # Split dataset into train, val, test
    x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, stratify=y,
                                                                test_size=0.2, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full,stratify=y_train_full, test_size=0.15, random_state=seed)

    # Scale dataset (standard scaler)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create TensorDataset object
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Create Dataloader
    g = torch.Generator()   #for deterministic DataLoader shuffling
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_and_preprocess_data(batch_size=32)
    print("Sanity check:", len(train_loader))

