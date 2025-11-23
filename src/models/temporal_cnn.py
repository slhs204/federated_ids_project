"""
Temporal CNN Model for Network Intrusion Detection

Architecture optimized for RTX 5070 Ti (12GB VRAM)
- Input: (batch, 78 features)
- Output: (batch, 16 classes)
- Parameters: ~2.1M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCNN(nn.Module):
    """
    Temporal Convolutional Network with Self-Attention
    
    Args:
        input_size (int): Number of input features (default: 78)
        num_classes (int): Number of output classes (default: 16)
        dropout (float): Dropout rate (default: 0.3)
    """
    
    def __init__(self, input_size=78, num_classes=16, dropout=0.3):
        super(TemporalCNN, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # 1D Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm1d(256)
        
        # Self-Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=dropout,
            batch_first=False
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Reshape for 1D convolution: (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Convolutional layers with batch normalization
        x = self.relu(self.bn1(self.conv1(x)))  # (batch, 64, features)
        x = self.relu(self.bn2(self.conv2(x)))  # (batch, 128, features)
        x = self.relu(self.bn3(self.conv3(x)))  # (batch, 256, features)
        
        # Self-Attention
        # Reshape to (seq_len, batch, channels) for attention
        x = x.permute(2, 0, 1)  # (features, batch, 256)
        attn_output, attn_weights = self.attention(x, x, x)
        
        # Global average pooling over sequence dimension
        x = attn_output.mean(dim=0)  # (batch, 256)
        
        # Fully connected layers
        x = self.dropout(x)
        x = self.relu(self.bn_fc1(self.fc1(x)))  # (batch, 128)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, num_classes)
        
        return x
    
    def get_num_params(self):
        """Return the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for comparison
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): LSTM hidden size
        num_layers (int): Number of LSTM layers
        num_classes (int): Number of output classes
    """
    
    def __init__(self, input_size=78, hidden_size=128, num_layers=2, num_classes=16, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input of shape (batch, features)
        
        Returns:
            torch.Tensor: Output logits of shape (batch, num_classes)
        """
        # Reshape to (batch, seq_len=1, features)
        x = x.unsqueeze(1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        # h_n: (num_layers * num_directions, batch, hidden_size)
        # We want the last layer's forward and backward hidden states
        forward_hidden = h_n[-2, :, :]  # Last layer forward
        backward_hidden = h_n[-1, :, :]  # Last layer backward
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Dropout and classification
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        
        return out
    
    def get_num_params(self):
        """Return the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Model factory
def get_model(model_name='temporal_cnn', **kwargs):
    """
    Factory function to get model by name
    
    Args:
        model_name (str): Model name ('temporal_cnn' or 'lstm')
        **kwargs: Additional model arguments
    
    Returns:
        nn.Module: Model instance
    """
    models = {
        'temporal_cnn': TemporalCNN,
        'lstm': LSTMClassifier
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)


if __name__ == "__main__":
    # Test model instantiation
    print("=" * 60)
    print("Testing Temporal CNN Model")
    print("=" * 60)
    
    model = TemporalCNN(input_size=78, num_classes=16)
    print(f"Model Parameters: {model.get_num_params():,}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 78)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (first 3 samples):\n{output[:3]}")
    
    print("\n" + "=" * 60)
    print("Testing LSTM Model")
    print("=" * 60)
    
    lstm_model = LSTMClassifier(input_size=78, num_classes=16)
    print(f"Model Parameters: {lstm_model.get_num_params():,}")
    
    with torch.no_grad():
        lstm_output = lstm_model(x)
    
    print(f"Output shape: {lstm_output.shape}")
