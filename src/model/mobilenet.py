"""
defining model architecture
Simplified MobileNet-based CNN for character recognition
Optimized for 28x28 grayscale images and smaller datasets
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution block - MobileNet's key building block.
    More efficient than standard convolution by separating depthwise and pointwise operations.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution: each input channel is convolved separately
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            groups=in_channels,  # Groups=in_channels makes it depthwise
            bias=False
        )
        
        # Batch normalization after depthwise
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution: 1x1 conv to combine channels
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            bias=False
        )
        
        # Batch normalization after pointwise
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu6(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu6(x)
        
        return x


class MobileNetCharClassifier(nn.Module):
    """
    Simplified MobileNet-based CNN for character recognition.
    
    Architecture:
    - Input: 28x28 grayscale images (1 channel)
    - Output: 52 classes (26 uppercase + 26 lowercase letters)
    
    Design choices for smaller dataset:
    - Fewer layers than standard MobileNet
    - Reduced channel widths
    - Depthwise separable convolutions for efficiency
    - Global average pooling to reduce parameters
    """
    def __init__(self, num_classes: int = 52):
        super(MobileNetCharClassifier, self).__init__()
        
        # Initial standard convolution to increase channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Depthwise separable convolution blocks
        # Each block reduces spatial dimensions and increases channels
        self.ds_conv1 = DepthwiseSeparableConv(16, 32, stride=2)   # 28x28 -> 14x14
        self.ds_conv2 = DepthwiseSeparableConv(32, 64, stride=2)  # 14x14 -> 7x7
        self.ds_conv3 = DepthwiseSeparableConv(64, 128, stride=1)  # 7x7 -> 7x7
        
        # Global average pooling - reduces spatial dimensions to 1x1
        # This significantly reduces parameters compared to fully connected layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final classifier
        # Dropout for regularization (important for smaller datasets)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Input: (batch_size, 1, 28, 28)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu6(x)
        
        # Depthwise separable convolutions
        x = self.ds_conv1(x)
        x = self.ds_conv2(x)
        x = self.ds_conv3(x)
        
        # Global average pooling: (batch_size, 128, 7, 7) -> (batch_size, 128, 1, 1)
        x = self.global_avg_pool(x)
        
        # Flatten: (batch_size, 128, 1, 1) -> (batch_size, 128)
        x = x.view(x.size(0), -1)
        
        # Dropout and classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def create_mobilenet_model(num_classes: int = 52, pretrained: bool = False) -> MobileNetCharClassifier:
    """
    Factory function to create a MobileNet character classifier model.
    
    Args:
        num_classes: Number of output classes (default: 52 for A-Z, a-z)
        pretrained: Whether to load pretrained weights (not implemented yet)
    
    Returns:
        MobileNetCharClassifier model instance
    """
    model = MobileNetCharClassifier(num_classes=num_classes)
    
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
    model = create_mobilenet_model(num_classes=52)
    
    # Print model architecture
    print("MobileNet Character Classifier")
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
