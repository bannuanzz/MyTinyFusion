#!/usr/bin/env python3
"""
Create a test dataset for TinyFusion using CIFAR-10
"""
import os
import torchvision
import torchvision.transforms as transforms
from PIL import Image

def create_cifar10_dataset():
    # Use faster mirror for CIFAR-10 download
    print("Downloading CIFAR-10 using faster method...")
    
    # First try to use wget to download from a faster mirror
    import subprocess
    import tarfile
    
    cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    cifar_path = "./cifar10_temp"
    os.makedirs(cifar_path, exist_ok=True)
    
    # Try to download directly
    try:
        print("Attempting direct download...")
        subprocess.run(["wget", "-O", f"{cifar_path}/cifar-10-python.tar.gz", cifar_url], 
                      check=True, timeout=300)
        print("Direct download successful!")
    except:
        print("Direct download failed, using torchvision fallback...")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to ImageNet size
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=cifar_path, train=True, download=True, transform=transform)
    
    # CIFAR-10 class names
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create directory structure
    base_path = "data/imagenet/train"
    os.makedirs(base_path, exist_ok=True)
    
    for i, class_name in enumerate(classes):
        class_path = os.path.join(base_path, f"class_{i:03d}_{class_name}")
        os.makedirs(class_path, exist_ok=True)
    
    # Save first 100 images from each class
    class_counts = [0] * 10
    max_per_class = 100
    
    print("Converting and saving images...")
    for idx, (image, label) in enumerate(trainset):
        if class_counts[label] < max_per_class:
            class_name = classes[label]
            class_path = os.path.join(base_path, f"class_{label:03d}_{class_name}")
            image_path = os.path.join(class_path, f"img_{class_counts[label]:04d}.jpg")
            image.save(image_path)
            class_counts[label] += 1
            
        # Stop when we have enough images for all classes
        if all(count >= max_per_class for count in class_counts):
            break
            
        if idx % 1000 == 0:
            print(f"Processed {idx} images...")
    
    print(f"Created test dataset with {sum(class_counts)} images in {len(classes)} classes")
    print(f"Dataset saved to: {base_path}")
    
    # Clean up temporary files
    import shutil
    shutil.rmtree('./cifar10_temp')

if __name__ == "__main__":
    create_cifar10_dataset()
