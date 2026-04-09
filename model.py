import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("./data")

# from input, mean and std for cifar100
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# from input, imagenet mean and std for FashionMnist and Plant Disease
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NUM_CLASSES = {
    'cifar100': 100,
    'fashion_mnist': 10,
    'plant_disease': 38
}

def get_transforms(dname='cifar100', augment=False):
    
    if dname == 'cifar100':
        # the training tansform for RGB datasets
        if augment:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ])
        else:
            # both splits will be the same for the baseline (no aug in exp. #1)
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        return train_transform, test_transform

    # adding a grayscale to fashion_mnist --> convert to 3 channels
    if dname == 'fashion_mnist':
        if augment:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                # convert to 3-channels
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            # both splits will be the same for the baseline (no aug in exp. #1)
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # convert to 3-channels
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        
        test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # convert to 3-channels
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return train_transform, test_transform
    
    # plant disease (RGB, ImageNet normalization)
    if dname == 'plant_disease':
        if augment:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            # baseline, no aug
             train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        
        test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return train_transform, test_transform


# creates data loaders for all 3 datasets
def load_datasets(val_ratio=0.2, seed=42, augment=False):
    datasets = {}

    # 1. CIFAR-100
    print('----CIFAR-100 Loading----')
    train_transform, test_transform = get_transforms('cifar100', augment=augment)

    cifar_training = torchvision.datasets.CIFAR100(
        root=ROOT / "cifar100", train=True, download=True, transform=train_transform
    )

    cifar_testing = torchvision.datasets.CIFAR100(
        root=ROOT / "cifar100", train=False, download=True, transform=test_transform
    )
    
    cifar_training = Subset(cifar_training, range(5000))
    cifar_testing = Subset(cifar_testing, range(1000))

    datasets["cifar100"] = (
        DataLoader(cifar_training, batch_size=128, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(cifar_testing, batch_size=128, shuffle=False, num_workers=2, pin_memory=True),
    )


    # Fashion-MNIST
    print('----Fashion-MNIST Loading----')
    train_transform, test_transform = get_transforms('fashion_mnist', augment=augment)

    fm_training = torchvision.datasets.FashionMNIST(
        root=ROOT / "fmnist", train=True, download=True, transform=train_transform
    )
    fm_testing = torchvision.datasets.FashionMNIST(
        root=ROOT / "fmnist", train=False, download=True, transform=test_transform
    )

    fm_training = Subset(fm_training, range(5000))
    fm_testing = Subset(fm_testing, range(1000))

    datasets['fashion_mnist'] = (
        DataLoader(fm_training, batch_size=128, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(fm_testing, batch_size=128, shuffle=False, num_workers=2, pin_memory=True),
    )


    # plant disease dataset
    # based on input
    plant_train = ROOT / "plant_diseases" / "plant_disease_data" / "train"
    plant_valid = ROOT / "plant_diseases" / "plant_disease_data" / "valid"
    
    if plant_train.exists() and plant_valid.exists():
        print('----Plant Disease Loading----')
        train_transform, val_transform = get_transforms('plant_disease', augment=augment)

        pd_train = torchvision.datasets.ImageFolder(str(plant_train), transform=train_transform)
        pd_val = torchvision.datasets.ImageFolder(str(plant_valid), transform=val_transform)

        datasets["plant_disease"] = (
            DataLoader(pd_train, batch_size=32, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(pd_val, batch_size=32, shuffle=False, num_workers=2, pin_memory=True),
        )
    else:
        print("Plant Disease dataset was not found!!")
  
    return datasets

# VGG16 setup
def vgg16_model(num_classes, pretrained=True, freeze_features=False):
    # num_classes - number of output classes
    # pretrained uses ImageNet weights
    # freeze_features freeze convolutional layers (for fine-tuning)

    # load model using weights parameter
    if pretrained:
        print('Loading pre-trained VGG16 - ImageNet weights')
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    else:
        print('Loading VGG16 from scratch - random weights')
        model = models.vgg16(weights=None)

    # freeze convolutional layers if requested
    if freeze_features:
        print('Freezing convolutional layers')
        for param in model.features.parameters():
            param.requires_grad = False

    # replace classifier head for fine tuning
    # get the input feature of the last layer
    in_features = model.classifier[6].in_features

    # create new classifier with dropout
    model.classifier[6] = nn.Sequential(
        # dropout for regularization
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )

    print(f"Modified VGG16 for {num_classes} classes.")

    return model

def resnet18_model(num_classes, pretrained=True, freeze_features=False):
    # num_classes - number of output classes
    # pretrained uses ImageNet weights
    # freeze_features freeze all layers except final FC

    # load model
    if pretrained:
        print('Loading pre-trained ResNet18 - ImageNet weights')
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        print('Loading ResNet18 from scratch - random weights')
        model = models.resnet18(weights=None)

    # freeze layers if requested
    if freeze_features:
        print("Freezing all layers except final fc")
        for param in model.parameters():
            param.requires_grad = False

    # replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # if frozen, ensure fc trains
    for i in model.fc.parameters():
        i.requires_grad = True

    print(f"Modified ResNet18 for {num_classes} classes.")

    return model

def train_model(model, train_loader, test_loader, num_epochs=10, device='cuda'):
    # move model to device
    model = model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer - SGD with momentum
    optimizer = optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()), 
        lr=0.001, 
        momentum=0.9,
        weight_decay = 1e-4
    )

    # learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # training history
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(num_epochs):
        # training phase
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # zero gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs 

            # print every 100 batches
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}: loss = {loss.item():.3f}")

        epoch_train_loss = train_loss_sum / train_count

        # testing 
        model.eval()
        test_loss_sum = 0.0
        test_count = 0
        correct = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                bs = labels.size(0)
                test_loss_sum += loss.item() * bs 
                test_count += bs

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        
        # calculate metrics
        epoch_test_loss = test_loss_sum / test_count
        epoch_test_acc = 100.0 * correct / test_count

        # save history
        history['train_loss'].append(epoch_train_loss)
        history['test_loss'].append(epoch_test_loss)
        history['test_acc'].append(epoch_test_acc)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {epoch_train_loss:.3f}")
        print(f"Test Loss: {epoch_test_loss:.3f}")
        print(f"Test Accuracy: {epoch_test_acc:.3f}%\n")

        # step the scheduler
        scheduler.step()

    return model, history

def main():
    # check for device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')


    for experiment in ['baseline', 'augmented']:
        augment = experiment == 'augmented'
        # load datasets
        print('Loading Datasets\n')
        datasets = load_datasets(augment=augment)

        for dname in ['cifar100', 'fashion_mnist', 'plant_disease']:
            # this is a debug to see if plant_disease was not found or downloaded
            if dname not in datasets:
                print(f"Dataset {dname} was not found, so skip\n")
                continue
            
            
            train_loader, test_loader = datasets[dname]

            n_classes = NUM_CLASSES[dname]

            # train vgg16 
            print(f"Training VGG16 on {dname} with {experiment}\n")

            v_model = vgg16_model(
                n_classes,
                pretrained = True,
                freeze_features = True # fine-tune only the class head
            )
    
            v_model, v_history = train_model(
                v_model, train_loader, test_loader,
                num_epochs = 5,
                device = device
            )


            # saving the model for later
            torch.save(v_model.state_dict(), f'vgg16_{dname}_{experiment}.pth')
            print(f"\nSave VGG16 {dname} model\n")


            # train resnet18 baseline
            print(f"Training ResNet18 on {dname} with {experiment}\n")

            r_model = resnet18_model(
                n_classes,
                pretrained = True,
                freeze_features = False
            )

            r_model, r_history = train_model(
                r_model, train_loader, test_loader,
                num_epochs = 5,
                device = device
            )

            torch.save(r_model.state_dict(), f'resnet18_{dname}_{experiment}.pth')
            print("\nSave ResNet18 baseline model\n")

            # plotting training curves
            print("\nPlotting Training Curves\n")

            plt.figure(figsize=(12,6))
            plt.subplot(1, 2, 1)
            plt.plot(v_history['test_acc'], label="VGG16")
            plt.plot(r_history['test_acc'], label="ResNet18")
            plt.title(f'Test Accuracy: {dname}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(v_history['test_loss'], label="VGG16")
            plt.plot(r_history['test_loss'], label="ResNet18")
            plt.title(f'Test Loss: {dname}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()
            plt.savefig(f'{dname}_{experiment}_training_curves.png')
            plt.show()

    print("ALL Training is Done")
    
if __name__ == "__main__":
    main()
    
    