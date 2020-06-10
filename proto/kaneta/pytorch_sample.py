import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader,
                              TensorDataset,
                              random_split)


class sample_model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(sample_model, self).__init__()
        self.linear_1 = torch.nn.Linear(input_dim, int(input_dim / 2))
        self.linear_2 = torch.nn.Linear(int(input_dim / 2), int(input_dim / 2))
        self.output = torch.nn.Linear(int(input_dim / 2), output_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        x = F.relu(x)
        y = self.output(x)

        return y


def main():
    ###########################################################################
    # 学習データ作成コーナー ##################################################
    ###########################################################################
    input_dim = 5
    output_dim = 1
    num_data = 50000

    x = np.random.randn(num_data, input_dim)
    y = np.sin(x[:, 0]) * np.power(x[:, 1], 3) + 3 * x[:, 2]\
        + 2 * x[:, 3] + np.sin(x[:, 4]) + np.random.randn(num_data)*0.8
    # plt.scatter(x, y)
    # plt.show()
    ###########################################################################

    x = torch.from_numpy(x).float()
    y = y.reshape(-1, output_dim)
    y = torch.from_numpy(y).float()

    dataset = TensorDataset(x, y)

    n_samples = len(dataset)
    train_size = int(len(dataset) * 0.8)  # 8割学習に
    val_size = n_samples - train_size     # 2割テストに

    train_iterator, test_iterator = random_split(dataset, [train_size, val_size])
    train_iterator = DataLoader(train_iterator, batch_size=100, shuffle=True, drop_last=True)
    test_iterator = DataLoader(test_iterator, batch_size=100, shuffle=True, drop_last=True)

    model = sample_model(input_dim, output_dim)
    print(model)                # モデルの内容を確認できる
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = torch.nn.MSELoss()

    losses = {'train': [], 'test': []}
    for e in range(50):
        # train
        model.train()           # 重要!!
        train_loss = 0
        for n_batch, (x, y) in enumerate(train_iterator):
            # update the gradients to zero
            optimizer.zero_grad()  # 重要!!

            # forward pass
            y_hat = model(x)
            # loss
            loss = mse(y_hat, y)

            # backward pass
            loss.backward()     # 重要!!
            optimizer.step()    # 重要!!
            train_loss += loss.item()
        train_loss = train_loss / len(train_iterator)
        losses['train'].append(train_loss)

        # predict
        model.eval()
        test_loss = 0
        with torch.no_grad():   # これで勾配が保存されない
            for n_batch, (x, y) in enumerate(test_iterator):
                # forward pass
                y_hat = model(x)
                # loss
                loss = mse(y_hat, y)
                test_loss += loss.item()
        test_loss = test_loss / len(test_iterator)
        losses['test'].append(test_loss)

        print(f'epoch: {e + 1}, train_loss: {train_loss:.10f}, test_loss: {test_loss:.10f}')
    model_path = './test_model_pth'
    torch.save({'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_stata_dict': optimizer.state_dict(),
                'loss': losses}, model_path)


if __name__ == '__main__':
    main()
