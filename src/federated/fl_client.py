"""
Flower Federated Learning Client Implementation

Optimized for RTX 5070 Ti - supports parallel client simulation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import flwr as fl
from collections import OrderedDict
import numpy as np
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))


class FlowerClient(fl.client.NumPyClient):
    """
    Flower client for federated IDS training
    
    Args:
        model (nn.Module): PyTorch model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (str): Device ('cuda' or 'cpu')
        client_id (int): Client identifier
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        client_id: int = 0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.client_id = client_id
        
        # Move model to device
        self.model.to(self.device)
        
        # Criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def get_parameters(self, config: Dict[str, any]) -> List[np.ndarray]:
        """
        Return model parameters as a list of NumPy arrays
        """
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from a list of NumPy arrays
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, any]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on local data
        
        Args:
            parameters: Global model parameters
            config: Training configuration
        
        Returns:
            Updated parameters, number of samples, metrics
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Get training config
        epochs = config.get("local_epochs", 3)
        batch_size = config.get("batch_size", 128)
        
        # Train
        train_loss, train_acc = self.train(epochs)
        
        # Return updated model parameters and metrics
        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "client_id": self.client_id
            }
        )
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, any]
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local validation data
        
        Args:
            parameters: Model parameters
            config: Evaluation configuration
        
        Returns:
            Loss, number of samples, metrics
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate
        loss, accuracy, f1 = self.test()
        
        return (
            float(loss),
            len(self.val_loader.dataset),
            {
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "client_id": self.client_id
            }
        )
    
    def train(self, epochs: int) -> Tuple[float, float]:
        """
        Local training loop
        
        Args:
            epochs: Number of training epochs
        
        Returns:
            Average loss and accuracy
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Metrics
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                epoch_total += target.size(0)
                epoch_correct += predicted.eq(target).sum().item()
            
            # Epoch metrics
            avg_loss = epoch_loss / len(self.train_loader)
            accuracy = 100.0 * epoch_correct / epoch_total
            
            total_loss += avg_loss
            correct += epoch_correct
            total += epoch_total
            
            if epoch == epochs - 1:  # Log last epoch
                print(f"Client {self.client_id} | Epoch {epoch+1}/{epochs} | "
                      f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")
        
        # Return average metrics over all epochs
        avg_loss = total_loss / epochs
        avg_accuracy = 100.0 * correct / total
        
        return avg_loss, avg_accuracy
    
    def test(self) -> Tuple[float, float, float]:
        """
        Evaluate model on validation set
        
        Returns:
            Loss, accuracy, F1-score
        """
        self.model.eval()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        avg_loss = test_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        # F1-score (macro average)
        from sklearn.metrics import f1_score
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        return avg_loss, accuracy, f1


def get_client_fn(
    model_fn,
    train_loaders: List[DataLoader],
    val_loaders: List[DataLoader],
    device: str = "cuda"
):
    """
    Factory function to create client_fn for Flower simulation
    
    Args:
        model_fn: Function that returns a model instance
        train_loaders: List of training data loaders (one per client)
        val_loaders: List of validation data loaders
        device: Device to use
    
    Returns:
        client_fn: Function that returns a FlowerClient
    """
    def client_fn(cid: str) -> FlowerClient:
        """
        Create a FlowerClient instance
        
        Args:
            cid: Client ID (string)
        
        Returns:
            FlowerClient instance
        """
        client_id = int(cid)
        
        # Create fresh model for this client
        model = model_fn()
        
        # Get this client's data loaders
        train_loader = train_loaders[client_id]
        val_loader = val_loaders[client_id]
        
        # Create and return client
        return FlowerClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            client_id=client_id
        )
    
    return client_fn


if __name__ == "__main__":
    # Test client creation
    print("Testing Flower Client")
    print("=" * 60)
    
    from torch.utils.data import TensorDataset
    from src.models.temporal_cnn import TemporalCNN, LSTMClassifier
    
    # Create dummy data
    X_train = torch.randn(1000, 78)
    y_train = torch.randint(0, 16, (1000,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    X_val = torch.randn(200, 78)
    y_val = torch.randint(0, 16, (200,))
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create client
    model = TemporalCNN()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = FlowerClient(model, train_loader, val_loader, device=device, client_id=0)
    
    # Test get_parameters
    params = client.get_parameters(config={})
    print(f"Number of parameter arrays: {len(params)}")
    
    # Test fit
    print("\nTesting local training...")
    updated_params, num_samples, metrics = client.fit(
        parameters=params,
        config={"local_epochs": 2}
    )
    print(f"Trained on {num_samples} samples")
    print(f"Metrics: {metrics}")
    
    # Test evaluate
    print("\nTesting evaluation...")
    loss, num_samples, eval_metrics = client.evaluate(
        parameters=updated_params,
        config={}
    )
    print(f"Evaluated on {num_samples} samples")
    print(f"Loss: {loss:.4f}")
    print(f"Metrics: {eval_metrics}")
    
    print("\n" + "=" * 60)
    print("Client test completed successfully!")
