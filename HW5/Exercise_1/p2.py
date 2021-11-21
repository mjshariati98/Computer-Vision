import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from utils import is_data_augmented, train_model, topK_accuracy, plot_loss_and_accuracy
from data_augmentation import augment

TRAIN_DIR = '../resources/Data/Train'
VALID_DIR = '../resources/Data/Test'
TRAIN_AUGMENTED_DIR = 'augmented_data/train5'

CLASS_NUM = 15
AUGMENTATION_COEFFICIENT = 5
IMAGE_INPUT_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.4


def main():
    # data augmentation
    if not is_data_augmented():
        augment(AUGMENTATION_COEFFICIENT, IMAGE_INPUT_SIZE, src_dir=TRAIN_DIR, dst_dir=TRAIN_AUGMENTED_DIR)

    # initial transform
    input_transform = transforms.Compose([
        transforms.Resize(IMAGE_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # datasets
    train_dataset = torchvision.datasets.ImageFolder(TRAIN_AUGMENTED_DIR, input_transform)
    valid_dataset = torchvision.datasets.ImageFolder(VALID_DIR, input_transform)

    # dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    dataloaders = {
        'train': train_dataloader,
        'valid': valid_dataloader
    }
    dataset_sizes = {
        'train': len(train_dataset),
        'valid': len(valid_dataset)
    }

    print("train dataset: " + str(dataset_sizes['train']) + " images")
    print("valid dataset: " + str(dataset_sizes['valid']) + " images")

    # set device (cpu/gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define model
    model = P2Model()
    model = model.to(device)
    print(model.eval())

    # loss function
    loss_func = nn.CrossEntropyLoss()
    # gradient descent with momentum
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # ReduceLROnPlateau scheduler: reduce learning rate when validation loss has stopped improving
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=7)

    # train the model
    model, history = train_model(model, dataloaders, dataset_sizes, loss_func, optimizer, scheduler, device, num_epochs=50)

    # plot loss, top-1 accuracy and top-5 accuracy
    plot_loss_and_accuracy(history)
    plt.savefig("out/2.jpg")


class P2Model(nn.Module):
    def __init__(self, num_classes: int = CLASS_NUM) -> None:
        super(P2Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 1024),  # 4096 -> 1024
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 256),  # 4096 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    main()
