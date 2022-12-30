import torch
import train
import test

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ",device)

    print("Training started...")
    train.train(device)

    print("Testing started...")
    test.test(device)