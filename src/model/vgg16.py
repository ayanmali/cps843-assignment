"""
defining model architecture
Simplified VGG16-based CNN for character recognition
Optimized for 28x28 grayscale images and smaller datasets
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBlock(nn.Module):
    """
    VGG-style convolutional block.
    Consists of multiple 3x3 convolutions followed by max pooling.
    """
    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2, 
                 use_pooling: bool = True):
        super(VGGBlock, self).__init__()
        
        layers = []
        
        # Add multiple 3x3 convolutions (VGG's signature)
        for i in range(num_convs):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        # Add max pooling if specified
        if use_pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

# Categorical cross-entropy loss
class VGG16CharClassifier(nn.Module):
    """
    Simplified VGG16-based CNN for character recognition.
    
    Architecture:
    - Input: 28x28 grayscale images (1 channel)
    - Output: 52 classes (26 uppercase + 26 lowercase letters)
    
    Design choices for smaller dataset:
    - Fewer blocks than standard VGG16 (3 blocks instead of 5)
    - Reduced channel widths (64, 128, 256 instead of 64, 128, 256, 512, 512)
    - Smaller fully connected layers
    - Global average pooling to reduce parameters
    - Batch normalization for better training stability
    """
    def __init__(self, num_classes: int = 52):
        super(VGG16CharClassifier, self).__init__()
        
        # Block 1: 28x28 -> 14x14
        # 2 conv layers with 64 channels
        self.block1 = VGGBlock(1, 64, num_convs=2, use_pooling=True)
        
        # Block 2: 14x14 -> 7x7
        # 2 conv layers with 128 channels
        self.block2 = VGGBlock(64, 128, num_convs=2, use_pooling=True)
        
        # Block 3: 7x7 -> 3x3 (or use global avg pool)
        # 2 conv layers with 256 channels
        self.block3 = VGGBlock(128, 256, num_convs=2, use_pooling=True)
        
        # Global average pooling - reduces spatial dimensions to 1x1
        # This significantly reduces parameters compared to fully connected layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head
        # Dropout for regularization (important for smaller datasets)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Input: (batch_size, 1, 28, 28)
        x = self.block1(x)  # -> (batch_size, 64, 14, 14)
        x = self.block2(x)  # -> (batch_size, 128, 7, 7)
        x = self.block3(x)  # -> (batch_size, 256, 3, 3)
        
        # Global average pooling: (batch_size, 256, 3, 3) -> (batch_size, 256, 1, 1)
        x = self.global_avg_pool(x)
        
        # Flatten: (batch_size, 256, 1, 1) -> (batch_size, 256)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


def create_vgg16_model(num_classes: int = 52, pretrained: bool = False) -> VGG16CharClassifier:
    """
    Factory function to create a VGG16 character classifier model.
    
    Args:
        num_classes: Number of output classes (default: 52 for A-Z, a-z)
        pretrained: Whether to load pretrained weights (not implemented yet)
    
    Returns:
        VGG16CharClassifier model instance
    """
    model = VGG16CharClassifier(num_classes=num_classes)
    
    # Initialize weights
    _initialize_weights(model)
    
    return model


def _initialize_weights(model: nn.Module):
    """
    Initialize model weights using Kaiming initialization for better training.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = create_vgg16_model(num_classes=52)
    
    # Print model architecture
    print("VGG16 Character Classifier")
    print("=" * 50)
    print(model)
    print("=" * 50)
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {num_params:,}")
    print(f"Model size: {num_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(4, 1, 28, 28)  # Batch of 4, 1 channel, 28x28 images
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Expected output: (batch_size, num_classes) = (4, 52)")
    
    # Verify output
    assert output.shape == (4, 52), f"Expected output shape (4, 52), got {output.shape}"
    print("\nModel test passed!")

