import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define Data Augmentation (SimCLR style )
transform = transforms.Compose([
    transforms.Resize(64),                  
    transforms.RandomResizedCrop(50),       # 50x50
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])

class SmallDataset(Dataset):
    def __init__(self, transform=None, num_samples=5):
        # using CIFAR-10
        self.data = torchvision.datasets.CIFAR10(root='./simCLR/data', train=True, download=True)
        self.data = [(self.data[i][0], self.data[i][1]) for i in range(num_samples)]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, _ = self.data[idx]
        # 4 time data augment
        augmented_images = torch.stack([self.transform(img) for _ in range(10)])
        return augmented_images.to(device)
    
def show_images(images, title="Augmented Images"):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for i, img in enumerate(images):
        # turn tensor into numpy, and change channel sorted (C, H, W) -> (H, W, C)
        img = img.cpu().permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.suptitle(title)
    plt.show()
    
dataset = SmallDataset(transform=transform, num_samples=10)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for i, augmented_images in enumerate(dataloader):
    # the shape is [1, 4, C, H, W], and remove batch dis
    augmented_images = augmented_images.squeeze(0)
    print(f"Original Image {i+1}: Generated {len(augmented_images)} augmented versions")
    show_images(augmented_images, title=f"Augmented Versions of Image {i+1}")