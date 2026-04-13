import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision import transforms

from model import vgg16_model, resnet18_model, NUM_CLASSES

# from input, mean and std for cifar100
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# from input, imagenet mean and std for FashionMnist and Plant Disease
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        #this will capture gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmaps(self, input_tensor, class_index=None):
        # make sure model is in eval mode
        self.model.eval()
        
        # forward pass
        output = self.model(input_tensor)

        #if no class specified, use the predicted one
        if class_index is None:
            class_index = torch.argmax(output[0]).item()

        # backward pass for the specific class
        self.model.zero_grad()  # zero gradients
        loss = output[0, class_index]
        loss.backward()

        # weight the channels by the gradient
        # average pooling of gradients 
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # weighted combination of activation maps
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()

        # apply ReLU 
        heatmap = F.relu(heatmap)

        # upsample heatmap to input images
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        heatmap = F.interpolate(heatmap, size=input_tensor.shape[2:],
                                mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze().cpu().detach().numpy()

        #normalie to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap
    
def heatmap_cover(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    # image is the numpy array in RGB format, values in [0, 1] or [0, 255]
    # heatmap is the numpy array with values in [0, 1]
    # alpha is the weight for the heatmap cover

    # converts image to uint8 [0, 255] if needed
    if image.max() <= 1.0:
        image_unit8 = np.uint8(255 * image)
    else:
        image_unit8 = image.astype(np.uint8)
    
    # convert heatmap to unit8
    heatmap_uint8 = np.uint8(255 * heatmap)

    # apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    covered = cv2.addWeighted(image_unit8, 1 - alpha, heatmap_colored, alpha, 0)

    # returns unit8 numu array in RGB
    return covered

def concentration_score(heatmap, method='variance'):
    if method == 'variance':
        # lower variance means more concentrated
        score = np.var(heatmap)
    elif method == 'peak_ratio':
        # higher ration = more concentrated
        score = heatmap.max() / (heatmap.mean() + 1e-8)
    elif method == 'area_top10':
        flat = heatmap.flatten()
        threshold = np.percentile(flat, 90)
        top_mass = flat[flat >= threshold].sum()
        score = top_mass / (flat.sum() + 1e-8)
    else:
        raise ValueError("Unknown method")
    
    return score

def save_heatmap_and_covered(image_tensor, heatmap, save_path, og_image_np=None):
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    # convert tensor to numpy image (assumming norm was undone or image is in [0, 1])
    if og_image_np is None:
        # assumming image_tensor is in [0, 1]
        image_np = image_tensor.squeeze(0).permute(1,2,0).cpu().numpy()
        image_np = np.clip(image_np, 0, 1)
    else:
        image_np = og_image_np

    # save heatmap as grayscale image 
    plt.imsave(f'{save_path}_heatmap.png', heatmap, cmap='jet', vmin=0, vmax=1)

    # save covered image
    covered = heatmap_cover(image_np, heatmap, alpha=0.5)
    Image.fromarray(covered).save(f"{save_path}_covered.png")

    # side by side comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Heatmap')
    axes[1].axis('off')
    axes[2].imshow(covered)
    axes[2].set_title('Covered')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_path}_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def get_preprocess(dname):
    if dname == 'cifar100':
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif dname == 'fashion_mnist':
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            # convert to 3-channels
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    elif dname == 'plant_disease':
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    else:
        raise ValueError("Unknown Dataset")
    
    # denormalize for visualization
    mean_t = torch.tensor(mean).view(3,1,1)
    std_t = torch.tensor(std).view(3,1,1)
    def denorm(tensor):
        image = tensor.squeeze(0).cpu() * std_t + mean_t
        image = image.permute(1,2,0).numpy()
        return np.clip(image, 0, 1)
    
    return preprocess, denorm

def load_trained_model(arch, dname, experiment, device):
    n_classes = NUM_CLASSES[dname]
    ckpt_path = f'{arch}_{dname}_{experiment}.pth'

    if arch == 'vgg16':
        model = vgg16_model(n_classes, pretrained=False)
        target_layer = model.features[28]   # last conv layer in vgg16 features block
    elif arch == 'resnet18':
        model = resnet18_model(n_classes, pretrained=False)
        target_layer = model.layer4[-1].conv2   # last conv before pool + new Sequential fc
    else:
        raise ValueError("Unknown architecture")
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded: {ckpt_path}")

    return model, target_layer

def run_gradcam(arch, dname, experiment, image_path, device, score_method='peak_ratio'):
    # runs grad-cam pipeline for one model/dataset/experiment 
    preprocess, denorm = get_preprocess(dname)

    image_pil = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    og_image_np = denorm(input_tensor)

    model, target_layer = load_trained_model(arch, dname, experiment, device)

    grad_cam = GradCam(model, target_layer)
    heatmap = grad_cam.generate_heatmaps(input_tensor)
    score = concentration_score(heatmap, method=score_method)

    save_path = f'outputs/{arch}_{dname}_{experiment}'
    save_heatmap_and_covered(input_tensor, heatmap, save_path, og_image_np=og_image_np)

    print(f'[{arch} | {dname} | {experiment}] Concentration Score ({score_method}): {score:.2f}')
    print(f"Saved to {save_path}_*.png\n")

    return heatmap, score


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')

    image_paths = {
        'cifar100': 'samples/cifar_sample.png',
        'fashion_mnist': 'samples/fashion_sample.png',
        'plant_disease': 'samples/plant_sample.png',
    }
    # or use 'variance' or 'area_top10'
    score_method = 'peak_ratio'

    all_scores = {}

    for experiment in ['baseline', 'augmented']:
        for dname in ['cifar100', 'fashion_mnist', 'plant_disease']:
            for arch in ['vgg16', 'resnet18']:
                image_path = image_paths.get(dname)
                if image_path is None or not os.path.exists(image_path):
                    print(f"No sample image found for {dname}\n")
                    continue

                check_path = f'{arch}_{dname}_{experiment}.pth'
                if not os.path.exists(check_path):
                    print(f"Checkpoint not found: {check_path}")
                    continue

                heatmap, score = run_gradcam(
                    arch, dname, experiment, image_path, device, score_method
                )
                all_scores[(arch, dname, experiment)] = score

        
    print("\nConcentration Score")
    for (arch, dname, exp), score in all_scores.items():
            print(f"{arch}\t {dname}\t    {exp}\t {score}")

if __name__ == '__main__':
    main()