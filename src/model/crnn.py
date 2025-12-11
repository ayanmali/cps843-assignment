"""
defining model architecture
CRNN (Convolutional Recurrent Neural Network) for character recognition
Combines CNN for feature extraction with RNN for sequence modeling
Optimized for 28x28 grayscale images and smaller datasets
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# CTC loss
class CRNNCharClassifier(nn.Module):
    """
    CRNN model for character recognition.
    
    Architecture:
    1. CNN Backbone: Extracts spatial features from 28x28 images
    2. Feature Map Reshaping: Converts CNN features to sequence format
    3. RNN Layers: Processes features as sequence (bidirectional LSTM/GRU)
    4. Classification Head: Maps RNN output to character classes
    
    Design choices for small images (28x28):
    - Lightweight CNN with 2-3 conv blocks
    - Global average pooling or spatial-to-sequence conversion
    - Bidirectional RNN for better feature modeling
    - Compact hidden dimensions to reduce parameters
    """
    def __init__(
        self,
        num_classes: int = 52,
        cnn_out_channels: int = 64,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        rnn_type: str = "LSTM",  # "LSTM" or "GRU"
        use_bidirectional: bool = True,
        dropout: float = 0.3
    ):
        super(CRNNCharClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_type = rnn_type
        self.use_bidirectional = use_bidirectional
        
        # CNN Backbone: Extract spatial features
        # Input: (batch, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        
        self.conv3 = nn.Conv2d(64, cnn_out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cnn_out_channels)
        # No pooling here - keep spatial dimensions for sequence conversion
        
        # Convert CNN feature maps to sequence
        # After conv3: (batch, cnn_out_channels, 7, 7)
        # We'll reshape to: (batch, sequence_length, feature_dim)
        # Option 1: Flatten spatial dimensions -> (batch, 49, cnn_out_channels)
        # Option 2: Use global pooling -> (batch, 1, cnn_out_channels)
        # We'll use Option 1 for better sequence modeling
        
        # RNN layers
        rnn_input_size = cnn_out_channels
        rnn_direction_multiplier = 2 if use_bidirectional else 1
        
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=use_bidirectional,
                dropout=dropout if rnn_num_layers > 1 else 0
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=use_bidirectional,
                dropout=dropout if rnn_num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}. Use 'LSTM' or 'GRU'")
        
        # Classification head
        # RNN output: (batch, sequence_length, hidden_size * num_directions)
        # We'll use the last output or average pooling
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden_size * rnn_direction_multiplier, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # CNN feature extraction
        # Input: (batch, 1, 28, 28)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)  # -> (batch, 32, 14, 14)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)  # -> (batch, 64, 7, 7)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)  # -> (batch, cnn_out_channels, 7, 7)
        
        # Reshape CNN features to sequence format
        # (batch, channels, height, width) -> (batch, height*width, channels)
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 2, 3, 1)  # -> (batch, height, width, channels)
        x = x.contiguous().view(batch_size, height * width, channels)  # -> (batch, 49, cnn_out_channels)
        
        # RNN processing
        # Input: (batch, sequence_length=49, feature_dim=cnn_out_channels)
        rnn_out, _ = self.rnn(x)  # -> (batch, 49, hidden_size * num_directions)
        
        # Use the last time step output for classification
        # Alternative: could use average pooling or attention
        rnn_out = rnn_out[:, -1, :]  # -> (batch, hidden_size * num_directions)
        
        # Classification
        x = self.dropout(rnn_out)
        x = self.fc(x)  # -> (batch, num_classes)
        
        return x


def create_crnn_model(
    num_classes: int = 52,
    cnn_out_channels: int = 64,
    rnn_hidden_size: int = 128,
    rnn_num_layers: int = 2,
    rnn_type: str = "LSTM",
    use_bidirectional: bool = True,
    dropout: float = 0.3,
    pretrained: bool = False
) -> CRNNCharClassifier:
    """
    Factory function to create a CRNN character classifier model.
    
    Args:
        num_classes: Number of output classes (default: 52 for A-Z, a-z)
        cnn_out_channels: Number of output channels from CNN backbone
        rnn_hidden_size: Hidden size of RNN layers
        rnn_num_layers: Number of RNN layers
        rnn_type: Type of RNN ("LSTM" or "GRU")
        use_bidirectional: Whether to use bidirectional RNN
        dropout: Dropout probability
        pretrained: Whether to load pretrained weights (not implemented yet)
    
    Returns:
        CRNNCharClassifier model instance
    """
    model = CRNNCharClassifier(
        num_classes=num_classes,
        cnn_out_channels=cnn_out_channels,
        rnn_hidden_size=rnn_hidden_size,
        rnn_num_layers=rnn_num_layers,
        rnn_type=rnn_type,
        use_bidirectional=use_bidirectional,
        dropout=dropout
    )
    
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
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            # Initialize RNN weights
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Set forget gate bias to 1 for LSTM
                    if isinstance(m, nn.LSTM):
                        n = param.size(0)
                        param.data[(n // 4):(n // 2)].fill_(1)


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
    model = create_crnn_model(
        num_classes=52,
        cnn_out_channels=64,
        rnn_hidden_size=128,
        rnn_num_layers=2,
        rnn_type="LSTM",
        use_bidirectional=True
    )
    
    # Print model architecture
    print("CRNN Character Classifier")
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
    
    # Test with GRU
    print("\n" + "=" * 50)
    print("Testing GRU variant...")
    model_gru = create_crnn_model(
        num_classes=52,
        rnn_type="GRU",
        use_bidirectional=True
    )
    output_gru = model_gru(dummy_input)
    print(f"GRU output shape: {output_gru.shape}")
    num_params_gru = count_parameters(model_gru)
    print(f"GRU parameters: {num_params_gru:,}")
    print("GRU test passed!")

