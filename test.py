import torch
from dataset_manager import DatasetManager
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score


def test(device):
    batch_size = 25
    cid = DatasetManager(train=False).get_ds()
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=True)
    model = torch.load("models/cnn_trans.h5")
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    true_ys = []
    pred_ys = []

    for (x, y) in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        pred = torch.argmax(y_hat, dim=1, keepdim=True)
        correct += pred.eq(y.data.view_as(pred)).sum()
        total += x.shape[0]

        for a_y in y:
            true_ys.append(a_y.detach().cpu().numpy())

        for a_y in pred_ys:
            pred_ys.append(a_y.detach().cpu().numpy())

    print(f'Total:{total}, Correct:{correct}, Accuracy:{correct/total*100:.2f}')
    print(r2_score(true_ys, pred_ys))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(device)
