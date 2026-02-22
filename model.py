import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt


NUM_CLASSES = {
    'cifar100': 100,
    'fashion_mnist': 10,
    'plant_disease': 38
}

def get_transforms(dname='cifar100'):
    # ImageNet normalization values
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    # defining the training transform for RGB datasets
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm
    ])

    # defining the validation transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm
    ])

    # adding a grayscale to fashion_mnist --> convert to 3 channels
    if dname == 'fashion_mnist':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # convert to 3-channels
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            norm
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            norm
        ])

    return train_transform, val_transform


# creates data loaders for all 3 datasets
def load_datasets(val_ratio=0.2, seed=42):
    datasets = {}

    # 1. CIFAR-100
    print('----CIFAR-100 Loading----')
    train_transform, val_transform = get_transforms('cifar100')

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    # train/val split 
    n_total = len(trainset)
    n_val = int(val_ratio * n_total)
    n_train = n_total - n_val

    train_subset, val_subset = random_split(
        trainset,
        [n_train, n_val],
        generator = torch.Generator().manual_seed(seed)
    )

    # swap val subset dataset transform to val_transform
    trainset_valT = torchvision.datasets.CIFAR100(
        root = "./data", train = True, download=False, transform = val_transform
    )
    val_subset.dataset = trainset_valT

    datasets['cifar100'] = (train_subset, val_subset)

    # Fashion-MNIST
    print('----Fashion-MNIST Loading----')
    train_transform, val_transform = get_transforms('fashion_mnist')

    trainset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    # train/val split 
    n_total = len(trainset)
    n_val = int(val_ratio * n_total)
    n_train = n_total - n_val

    train_subset, val_subset = random_split(
        trainset,
        [n_train, n_val],
        generator = torch.Generator().manual_seed(seed)
    )

    # swap val subset dataset transform to val_transform
    trainset_valT = torchvision.datasets.FashionMNIST(
        root = "./data", train = True, download=False, transform = val_transform
    )
    val_subset.dataset = trainset_valT

    datasets['fashion_mnist'] = (train_subset, val_subset)

    # plant disease dataset
    # because this database is not available in torchvision, it needs a custome loading
    # it expects the structure to be
    # plant_disease/
    #   train/class1/, train/class2/, etc
    #   val/class1/, val/class2/, etc
    '''
    print("----Plant Disease Dataset Manual Setup----")
    train_transform, val_transform = get_transforms('plant_disease')

    plant_train = torchvision.datasets.ImageFolder(
        root='./data/plant_disease/train',
        transform = train_transform
    )
    plant_val = torchvision.datasets.ImageFolder(
        root='./data/plant_disease/val',
        transform = val_transform
    )
    datasets['plant_disease'] = (plant_train, plant_val)
    '''

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

    print(f"Modified ResNet18 fro {num_classes} classes.")

    return model

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
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
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

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

        # validation 
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        correct = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                bs = labels.size(0)
                val_loss_sum += loss.item() * bs 
                val_count += bs

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        
        # calculate metrics
        epoch_val_loss = val_loss_sum / val_count
        epoch_val_acc = 100.0 * correct / val_count

        # save history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {epoch_train_loss:.3f}")
        print(f"Validation Loss: {epoch_val_loss:.3f}")
        print(f"Validation Accurazy: {epoch_val_acc:.3f}%\n")

        # step the scheduler
        scheduler.step()

    return model, history

def main():
    # check for device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')

    # load datasets
    print('Loading Datasets\n')
    datasets = load_datasets()

    # create data loaders for CIFAR-100 (the baseline)
    print('Creating Data Loaders - CIFAR-100, the baseline')

    # debuggin, making small training and validation
    print("debug start----")
    small_train = Subset(datasets['cifar100'][0], list(range(200)))
    small_val = Subset(datasets['cifar100'][1], list(range(50)))

    train_loader = DataLoader(small_train, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(small_val, batch_size=4, shuffle=False, num_workers=0)

    # train vgg16 baseline (no augmentation)
    print('Training VGG16 Baselein with No Augmentation\n')

    v_model = vgg16_model(
        NUM_CLASSES['cifar100'],
        pretrained = True,
        freeze_features = True # fine-tune all layers
    )
    
    v_model, v_history = train_model(
        v_model, train_loader, val_loader,
        num_epochs = 1,
        device = device
    )

    print("Debug end---")

    # saving the model for later
    torch.save(v_model.state_dict(), 'vgg16_cifar100_baseline.pth')
    print("\nSave VGG16 baseline model\n")

    # train resnet18 baseline
    print("Training ResNet18 Basline with No Augmentation\n")

    r_model = resnet18_model(
        NUM_CLASSES['cifar100'],
        pretrained = True,
        freeze_features = False
    )

    r_model, r_history = train_model(
        r_model, train_loader, val_loader,
        num_epochs = 10,
        device = device
    )

    torch.save(r_model.state_dict(), 'resnet18_cifar100_baseline.pth')
    print("\nSave ResNet18 baseline model\n")

    # plotting training curves
    print("\nPlotting Training Curves\n")

    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)
    plt.plot(v_history['val_acc'], label="VGG16")
    plt.plot(r_history['val_acc'], label="ResNet18")
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(v_history['val_loss'], label="VGG16")
    plt.plot(r_history['val_loss'], label="ResNet18")
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('baseline_training_curves.png')
    plt.show()

    print("\nBaseline Training is Done")
    
if __name__ == "__main__":
    main()
    
    