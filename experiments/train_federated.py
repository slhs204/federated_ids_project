"""
Federated Learning Training Script

Complete training pipeline using Flower simulation mode
Optimized for RTX 5070 Ti - can run 8 clients in parallel
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import flwr as fl
from pathlib import Path
import json
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.temporal_cnn import TemporalCNN, LSTMClassifier
from src.federated.fl_client import get_client_fn
from src.federated.fl_server import (
    create_strategy,
    get_evaluate_fn,
    get_on_fit_config_fn
)


def load_data(data_path: Path):
    """
    Load preprocessed data
    
    Args:
        data_path: Path to processed data directory
    
    Returns:
        train_loaders, val_loaders, test_loader
    """
    print(f"Loading data from {data_path}")
    
    # For now, create dummy data (replace with actual data loading)
    # TODO: Replace with actual data loading from pkl files
    
    # Create dummy datasets (REPLACE THIS)
    num_clients = 8
    samples_per_client = 50000
    
    train_loaders = []
    val_loaders = []
    
    for i in range(num_clients):
        # Generate dummy Non-IID data
        X_train = torch.randn(samples_per_client, 78)
        # Simulate Non-IID: each client has different class distribution
        y_train = torch.randint(0, 16, (samples_per_client,))
        
        # Add class imbalance for Non-IID simulation
        if i < 3:  # First 3 clients: more DDoS attacks (class 0)
            mask = torch.rand(samples_per_client) < 0.7
            y_train[mask] = 0
        elif i < 6:  # Next 3 clients: more Botnet (class 1)
            mask = torch.rand(samples_per_client) < 0.6
            y_train[mask] = 1
        # Rest: balanced
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        train_loaders.append(train_loader)
        
        # Validation data
        X_val = torch.randn(10000, 78)
        y_val = torch.randint(0, 16, (10000,))
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        val_loaders.append(val_loader)
    
    # Global test set
    X_test = torch.randn(20000, 78)
    y_test = torch.randint(0, 16, (20000,))
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ Loaded {num_clients} clients")
    print(f"✓ Train samples per client: {samples_per_client}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    return train_loaders, val_loaders, test_loader


def main(args):
    """Main training function"""
    
    print("=" * 70)
    print("Federated Learning - Network Intrusion Detection")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Num Clients: {args.num_clients}")
    print(f"  Num Rounds: {args.num_rounds}")
    print(f"  Local Epochs: {args.local_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print("=" * 70)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data
    data_path = Path(args.data_path)
    train_loaders, val_loaders, test_loader = load_data(data_path)
    
    # Create model
    print(f"\n🧠 Creating model: {args.model}")
    if args.model == "temporal_cnn":
        model = TemporalCNN(input_size=78, num_classes=16)
    elif args.model == "lstm":
        model = LSTMClassifier(input_size=78, num_classes=16)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get initial parameters
    initial_params = [val.cpu().numpy() for val in model.state_dict().values()]
    
    # Create strategy
    print(f"\n📊 Creating FL strategy: {args.strategy}")
    strategy = create_strategy(
        strategy_name=args.strategy,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=max(2, int(args.num_clients * args.fraction_fit)),
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
        evaluate_fn=get_evaluate_fn(model, test_loader, device=str(device)),
        on_fit_config_fn=get_on_fit_config_fn(
            local_epochs=args.local_epochs,
            batch_size=args.batch_size
        ),
        proximal_mu=args.proximal_mu
    )
    
    # Create client function
    def model_fn():
        """Return a fresh model instance"""
        if args.model == "temporal_cnn":
            return TemporalCNN(input_size=78, num_classes=16)
        elif args.model == "lstm":
            return LSTMClassifier(input_size=78, num_classes=16)
    
    client_fn = get_client_fn(
        model_fn=model_fn,
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        device=str(device)
    )
    
    # Start simulation
    print(f"\n🚀 Starting Federated Learning ({args.num_rounds} rounds)")
    print("=" * 70)
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args={
            "num_gpus": 1 if device.type == "cuda" else 0,
            "num_cpus": 4,  # Reduced from 8 to prevent Raylet errors
            "include_dashboard": False,
            "_temp_dir": str(Path.home() / ".ray_tmp"),  # Use local temp dir
        }
    )
    
    print("\n" + "=" * 70)
    print("✅ Training completed!")
    print("=" * 70)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        # Convert history to JSON-serializable format
        history_dict = {
            "losses_distributed": [(round_num, loss) for round_num, loss in history.losses_distributed],
            "losses_centralized": [(round_num, loss) for round_num, loss in history.losses_centralized],
            "metrics_distributed": history.metrics_distributed,
            "metrics_centralized": history.metrics_centralized
        }
        json.dump(history_dict, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"   - Training history: {history_path}")
    
    # Print final metrics
    if history.losses_centralized:
        final_loss = history.losses_centralized[-1][1]
        print(f"\n📈 Final Test Loss: {final_loss:.4f}")
    
    if history.metrics_centralized and "accuracy" in history.metrics_centralized:
        final_acc = history.metrics_centralized["accuracy"][-1][1]
        print(f"📈 Final Test Accuracy: {final_acc:.4f} ({100*final_acc:.2f}%)")
    
    return history


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Federated Learning for Network Intrusion Detection"
    )
    
    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed",
        help="Path to processed data directory"
    )
    
    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="temporal_cnn",
        choices=["temporal_cnn", "lstm"],
        help="Model architecture"
    )
    
    # FL Configuration
    parser.add_argument(
        "--strategy",
        type=str,
        default="fedavg",
        choices=["fedavg", "fedprox"],
        help="FL strategy"
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=8,
        help="Number of clients"
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=50,
        help="Number of FL rounds"
    )
    parser.add_argument(
        "--fraction_fit",
        type=float,
        default=0.8,
        help="Fraction of clients to sample per round"
    )
    
    # Training
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=3,
        help="Number of local epochs per round"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size"
    )
    parser.add_argument(
        "--proximal_mu",
        type=float,
        default=0.01,
        help="Proximal term coefficient (FedProx only)"
    )
    
    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=8,
        help="Number of CPUs for Ray"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/federated",
        help="Output directory for results"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
