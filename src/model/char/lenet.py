
from labels import LABELS
import torch
import torch.nn as nn

TARGET_SIZE = (24, 24)

class LeNetCharClassifier(nn.Module):
    def __init__(self, num_classes: int = len(LABELS)):
        super(LeNetCharClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=0)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(24, 48, kernel_size=5, stride=1, padding=0)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(48 * 3 * 3, 384)
        
        self.fc2 = nn.Linear(384, 256)
        
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):

        x = self.conv1(x)
        x = torch.sigmoid(x)
        
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = torch.sigmoid(x)
        
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = torch.sigmoid(x)
        
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        x = self.fc3(x)
        
        return x


def create_lenet_model(num_classes: int = len(LABELS)):
    model = LeNetCharClassifier(num_classes=num_classes)
    
    _initialize_weights(model)
    
    return model


def _initialize_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = create_lenet_model(num_classes=len(LABELS))
    
    print("LeNet Character Classifier")
    print("=" * 50)
    print(model)
    print("=" * 50)
    
    num_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {num_params:,}")
    print(f"Model size: {num_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    print("\nTesting forward pass")
    dummy_input = torch.randn(4, 1, *TARGET_SIZE)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output: (batch_size, num_classes) = (4, {len(LABELS)})")
    
    assert output.shape == (4, len(LABELS)), f"Expected output shape (4, {len(LABELS)}), got {output.shape}"
    print("\nModel test passed!")
    
    print("\n" + "=" * 50)
    print("Model Architecture:")
    print("=" * 50)
    print("Trainable Layers:")
    print("  1. Conv1: 1 -> 24 channels, 3x3 kernel")
    print("  2. Conv2: 24 -> 48 channels, 5x5 kernel")
    print("  3. FC1: 432 -> 384")
    print("  4. FC2: 384 -> 256")
    print(f"  5. FC3 (Output): 256 -> {len(LABELS)}")
    print("\nActivation: Sigmoid (for conv and FC layers)")
    print("Pooling: MaxPool2d (2x2, stride=2)")