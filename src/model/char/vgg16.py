
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_SIZE = (24, 24)

class VGGBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super(VGGBlock, self).__init__()
        
        layers = []
        
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
            layers.append(nn.Sigmoid())
       
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class VGG16CharClassifier(nn.Module):
    def __init__(self, num_classes: int = 52):
        super(VGG16CharClassifier, self).__init__()
        
        self.block1 = VGGBlock(1, 64, num_convs=2)
        self.block2 = VGGBlock(64, 128, num_convs=2)
        self.block3 = VGGBlock(128, 256, num_convs=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.global_avg_pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        
        return x


def create_vgg16_model(num_classes: int = 52):
    model = VGG16CharClassifier(num_classes=num_classes)
    
    _initialize_weights(model)
    
    return model


def _initialize_weights(model: nn.Module):
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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = create_vgg16_model(num_classes=52)
    
    print("VGG16 Character Classifier")
    print("=" * 50)
    print(model)
    print("=" * 50)
    
    num_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {num_params:,}")
    print(f"Model size: {num_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    print("\nTesting forward pass...")
    dummy_input = torch.randn(4, 1, *TARGET_SIZE)  # Batch of 4, 1 channel, 24x24 images
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Expected output: (batch_size, num_classes) = (4, 52)")
    
    # Verify output
    assert output.shape == (4, 52), f"Expected output shape (4, 52), got {output.shape}"
    print("\nModel test passed")

