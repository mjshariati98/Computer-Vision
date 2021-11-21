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
BATCH_SIZE = 8
LEARNING_RATE = 0.001
MOMENTUM = 0.6


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
    model = torchvision.models.alexnet(pretrained=False)
    # add another dropout layer before output layer and change output layer
    model.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 2048), # 4096 -> 2048
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(2048, 512), # 4096 -> 512
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(512, CLASS_NUM),
    )
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
    plt.savefig("out/3.jpg")


if __name__ == '__main__':
    main()
