"""
Flower Federated Learning Server Implementation

Supports FedAvg, FedProx strategies with evaluation
"""

import flwr as fl
from flwr.server.strategy import FedAvg, FedProx
from flwr.common import Parameters, Scalar
from typing import Dict, Optional, Tuple, List
import torch
import numpy as np
from pathlib import Path


def get_evaluate_fn(model, test_loader, device="cuda"):
    """
    Return an evaluation function for server-side evaluation
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to use
    
    Returns:
        Evaluation function
    """
    def evaluate(
        server_round: int,
        parameters: Parameters,
        config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate global model on centralized test set
        
        Args:
            server_round: Current round number
            parameters: Global model parameters
            config: Configuration dictionary
        
        Returns:
            Loss and metrics dictionary
        """
        # Convert parameters to model weights
        params_dict = zip(model.state_dict().keys(), parameters.tensors)
        state_dict = {k: torch.tensor(np.frombuffer(v, dtype=np.float32).copy()).reshape(model.state_dict()[k].shape) 
                     for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        # Move model to device
        model.to(device)
        model.eval()
        
        # Evaluation
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        
        # F1-score
        from sklearn.metrics import f1_score
        f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        print(f"\n[Round {server_round}] Server Evaluation:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")
        print(f"  F1-Macro: {f1_macro:.4f}")
        print(f"  F1-Weighted: {f1_weighted:.4f}")
        
        return avg_loss, {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted
        }
    
    return evaluate


def get_on_fit_config_fn(
    local_epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 0.001
):
    """
    Return function that configures training on client
    
    Args:
        local_epochs: Number of local training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
    
    Returns:
        Configuration function
    """
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """
        Return training configuration for each round
        """
        config = {
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "server_round": server_round
        }
        
        # Optional: Adjust learning rate over rounds
        if server_round > 30:
            config["learning_rate"] = learning_rate * 0.5
        if server_round > 40:
            config["learning_rate"] = learning_rate * 0.1
        
        return config
    
    return fit_config


def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """
    Aggregate evaluation metrics from multiple clients
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples
    
    Returns:
        Aggregated metrics dictionary
    """
    # Multiply accuracy by number of examples for each client
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return
    aggregated = {
        "accuracy": sum(accuracies) / sum(examples),
    }
    
    # Add F1-score if available
    if "f1_score" in metrics[0][1]:
        f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
        aggregated["f1_score"] = sum(f1_scores) / sum(examples)
    
    return aggregated


def create_strategy(
    strategy_name: str = "fedavg",
    fraction_fit: float = 0.8,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 6,
    min_evaluate_clients: int = 8,
    min_available_clients: int = 8,
    initial_parameters: Optional[Parameters] = None,
    evaluate_fn = None,
    on_fit_config_fn = None,
    proximal_mu: float = 0.01  # For FedProx only
) -> fl.server.strategy.Strategy:
    """
    Create FL strategy
    
    Args:
        strategy_name: Strategy name ('fedavg' or 'fedprox')
        fraction_fit: Fraction of clients to sample for training
        fraction_evaluate: Fraction of clients to sample for evaluation
        min_fit_clients: Minimum number of clients for training
        min_evaluate_clients: Minimum number of clients for evaluation
        min_available_clients: Minimum number of available clients
        initial_parameters: Initial model parameters
        evaluate_fn: Server-side evaluation function
        on_fit_config_fn: Training configuration function
        proximal_mu: Proximal term coefficient (FedProx only)
    
    Returns:
        Strategy instance
    """
    common_args = {
        "fraction_fit": fraction_fit,
        "fraction_evaluate": fraction_evaluate,
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "min_available_clients": min_available_clients,
        "initial_parameters": initial_parameters,
        "evaluate_fn": evaluate_fn,
        "on_fit_config": on_fit_config_fn,
        "evaluate_metrics_aggregation_fn": weighted_average,
    }
    
    if strategy_name.lower() == "fedavg":
        return FedAvg(**common_args)
    elif strategy_name.lower() == "fedprox":
        return FedProx(**common_args, proximal_mu=proximal_mu)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def start_server(
    strategy: fl.server.strategy.Strategy,
    num_rounds: int = 50,
    server_address: str = "0.0.0.0:8080",
    save_path: Optional[Path] = None
):
    """
    Start Flower server
    
    Args:
        strategy: FL strategy
        num_rounds: Number of training rounds
        server_address: Server address
        save_path: Path to save final model
    """
    print("=" * 60)
    print("Starting Flower Server")
    print("=" * 60)
    print(f"Server address: {server_address}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Strategy: {strategy.__class__.__name__}")
    print("=" * 60)
    
    # Start server
    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # Save final model if path provided
    if save_path is not None:
        print(f"Saving final model to {save_path}")
        # TODO: Implement model saving
        # This requires extracting parameters from strategy
    
    return history


if __name__ == "__main__":
    # Test server setup
    print("Testing Flower Server Setup")
    print("=" * 60)
    
    from src.models.temporal_cnn import TemporalCNN
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create dummy test data
    X_test = torch.randn(500, 78)
    y_test = torch.randint(0, 16, (500,))
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = TemporalCNN()
    
    # Get initial parameters
    initial_params = [val.cpu().numpy() for val in model.state_dict().values()]
    
    # Create strategy
    strategy = create_strategy(
        strategy_name="fedavg",
        fraction_fit=0.8,
        min_available_clients=8,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
        evaluate_fn=get_evaluate_fn(model, test_loader, device="cpu"),
        on_fit_config_fn=get_on_fit_config_fn()
    )
    
    print("Strategy created successfully!")
    print(f"Strategy type: {type(strategy).__name__}")
    print(f"Fraction fit: {strategy.fraction_fit}")
    print(f"Min available clients: {strategy.min_available_clients}")
    
    print("\n" + "=" * 60)
    print("Server setup test completed!")
    print("To start actual training, run train_federated.py")
