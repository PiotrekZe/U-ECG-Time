import Dataset
import ECGDataset
import UTime
import FocalLoss
from RunModel import train_model, test_model

import torch
from torch.utils.data import DataLoader
import torch.optim as optim


def main():
    epochs = 100
    lengths = [1800, 900, 450, 225]
    channels = [32, 64, 128, 256]
    num_classes = 5
    length = 1800
    learning_rate = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    path = "D:/Databases/21/mit-bih-arrhythmia-database-1.0.0/"

    dataset = Dataset.Dataset(path=path, length=length)
    X, y, peaks, peaks_idx = dataset.read_dataset()

    X_train, X_test = X[: int(0.8 * X.shape[0])], X[int(0.8 * X.shape[0]) :]
    y_train, y_test = y[: int(0.8 * y.shape[0])], y[int(0.8 * y.shape[0]) :]
    peaks_train, peaks_test = (
        peaks[: int(0.8 * peaks.shape[0])],
        peaks[int(0.8 * peaks.shape[0]) :],
    )
    peaks_idx_train, peaks_idx_test = (
        peaks_idx[: int(0.8 * peaks_idx.shape[0])],
        peaks_idx[int(0.8 * peaks_idx.shape[0]) :],
    )

    train_dataset = ECGDataset(
        X_train.transpose(0, 2, 1), y_train, peaks_train, peaks_idx_train
    )
    test_dataset = ECGDataset(
        X_test.transpose(0, 2, 1), y_test, peaks_test, peaks_idx_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model = UTime.UTime(lengths=lengths, channels=channels, num_classes=num_classes)
    criterion = FocalLoss.FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_tab, test_tab = [], []
    for epoch in range(epochs):
        print(f"EPOCH: {epoch+1}/{epochs}")
        train_list = train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            num_classes=num_classes,
        )
        test_list = test_model(
            model=model,
            criterion=criterion,
            data_loader=test_loader,
            device=device,
            num_classes=num_classes,
        )
        train_tab.append(train_list)
        test_tab.append(test_list)

    # to do: save data, make plots


if __name__ == "__main__":
    main()
