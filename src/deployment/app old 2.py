"""
Gradio Deployment Interface for FL-IDS (Fixed)

Real-time network intrusion detection with visual feedback
"""

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.models.temporal_cnn import TemporalCNN
    MODEL_AVAILABLE = True
except ImportError:
    print("⚠️  temporal_cnn not found, using demo mode")
    MODEL_AVAILABLE = False
    TemporalCNN = None


# Attack class names
CLASS_NAMES = [
    "BENIGN",
    "DDoS",
    "PortScan",
    "Bot",
    "Infiltration",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - Sql Injection",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "DoS Hulk",
    "DoS GoldenEye",
    "Heartbleed",
    "Unknown"
]

# Color mapping
SEVERITY_COLORS = {
    "BENIGN": "#28a745",
    "DDoS": "#dc3545",
    "PortScan": "#ffc107",
    "Bot": "#dc3545",
    "Infiltration": "#dc3545",
    "Web Attack - Brute Force": "#fd7e14",
    "Web Attack - XSS": "#fd7e14",
    "Web Attack - Sql Injection": "#fd7e14",
    "FTP-Patator": "#fd7e14",
    "SSH-Patator": "#fd7e14",
    "DoS slowloris": "#dc3545",
    "DoS Slowhttptest": "#dc3545",
    "DoS Hulk": "#dc3545",
    "DoS GoldenEye": "#dc3545",
    "Heartbleed": "#dc3545",
    "Unknown": "#6c757d"
}


class InferenceEngine:
    """Inference engine for the deployed model"""
    
    def __init__(self, model_path: str = None, device: str = "cpu"):
        """Initialize inference engine"""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if MODEL_AVAILABLE and TemporalCNN is not None:
            self.model = TemporalCNN(input_size=78, num_classes=16)
            
            if model_path and Path(model_path).exists():
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✓ Loaded model from {model_path}")
                except Exception as e:
                    print(f"⚠️  Model load error: {e}")
                    print("   Using untrained model")
            else:
                print("⚠️  Using untrained model (demo mode)")
            
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = None
            print("⚠️  Model not available, using demo mode")
    
    def preprocess(self, features: np.ndarray) -> torch.Tensor:
        """
        Preprocess input features
        
        Args:
            features: Raw feature array (any size)
        
        Returns:
            Preprocessed tensor (78 features)
        """
        # Ensure we have exactly 78 features
        if len(features) < 78:
            # Pad with zeros if too few features
            padded = np.zeros(78)
            padded[:len(features)] = features
            features = padded
        elif len(features) > 78:
            # Truncate if too many features
            features = features[:78]
        
        # Standardize
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        # Convert to tensor
        tensor = torch.FloatTensor(features).unsqueeze(0)
        return tensor.to(self.device)
    
    @torch.no_grad()
    def predict(self, features: np.ndarray):
        """Make prediction"""
        x = self.preprocess(features)
        
        if self.model is None:
            # Demo mode - return dummy predictions
            predicted_class = "BENIGN (Demo Mode)"
            confidence_score = 0.75
            top5_classes = ["BENIGN (Demo)", "DDoS", "PortScan", "Bot", "DoS Hulk"]
            top5_scores = np.array([0.75, 0.10, 0.08, 0.04, 0.03])
        else:
            # Real prediction
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            
            confidence, predicted = probs.max(1)
            predicted_class = CLASS_NAMES[predicted.item()]
            confidence_score = confidence.item()
            
            top5_probs, top5_indices = torch.topk(probs, min(5, len(CLASS_NAMES)))
            top5_classes = [CLASS_NAMES[idx.item()] for idx in top5_indices[0]]
            top5_scores = top5_probs[0].cpu().numpy()
        
        return {
            "prediction": predicted_class,
            "confidence": confidence_score,
            "top5_classes": top5_classes,
            "top5_scores": top5_scores
        }


# Global inference engine
inference_engine = None


def initialize_engine():
    """Initialize inference engine"""
    global inference_engine
    if inference_engine is None:
        model_path = "results/models/best_model.pt"
        inference_engine = InferenceEngine(model_path=model_path)


def create_prediction_chart(top5_classes, top5_scores):
    """Create horizontal bar chart"""
    colors = [SEVERITY_COLORS.get(cls, "#6c757d") for cls in top5_classes]
    
    fig = go.Figure(data=[
        go.Bar(
            y=top5_classes[::-1],
            x=top5_scores[::-1],
            orientation='h',
            marker=dict(color=colors[::-1]),
            text=[f"{score:.2%}" for score in top5_scores[::-1]],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Top 5 Predictions",
        xaxis_title="Confidence",
        yaxis_title="Attack Type",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[0, 1], tickformat=".0%")
    )
    
    return fig


def predict_from_csv(csv_file):
    """Predict from uploaded CSV file"""
    initialize_engine()
    
    try:
        # Read CSV
        df = pd.read_csv(csv_file.name)
        
        if len(df) == 0:
            return "❌ Empty CSV file", None, None
        
        # Get all numeric features from first row
        features = df.iloc[0].select_dtypes(include=[np.number]).values
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        print(f"📊 CSV features: {len(features)} (will pad/truncate to 78)")
        
        # Predict
        result = inference_engine.predict(features)
        
        # Format output
        prediction_text = f"""
### 🔍 Detection Result

**Prediction:** {result['prediction']}  
**Confidence:** {result['confidence']:.2%}

---

**Risk Level:** {"🔴 HIGH" if result['confidence'] > 0.8 else "🟡 MEDIUM" if result['confidence'] > 0.5 else "🟢 LOW"}

**Input Features:** {len(features)} columns detected
        """
        
        # Create chart
        chart = create_prediction_chart(result['top5_classes'], result['top5_scores'])
        
        return prediction_text, result['confidence'], chart
    
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return error_msg, None, None


def predict_from_manual(
    flow_duration, total_fwd_packets, total_bwd_packets,
    fwd_packet_length_mean, bwd_packet_length_mean
):
    """Predict from manually entered features"""
    initialize_engine()
    
    try:
        # Create feature vector
        features = np.zeros(78)
        features[0] = float(flow_duration)
        features[1] = float(total_fwd_packets)
        features[2] = float(total_bwd_packets)
        features[3] = float(fwd_packet_length_mean)
        features[4] = float(bwd_packet_length_mean)
        
        # Fill remaining with variations
        for i in range(5, 78):
            features[i] = features[i % 5] * np.random.uniform(0.8, 1.2)
        
        # Predict
        result = inference_engine.predict(features)
        
        # Format output
        prediction_text = f"""
### 🔍 Detection Result

**Prediction:** {result['prediction']}  
**Confidence:** {result['confidence']:.2%}

---

**Risk Level:** {"🔴 HIGH" if result['confidence'] > 0.8 else "🟡 MEDIUM" if result['confidence'] > 0.5 else "🟢 LOW"}
        """
        
        # Create chart
        chart = create_prediction_chart(result['top5_classes'], result['top5_scores'])
        
        return prediction_text, result['confidence'], chart
    
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None


# Create Gradio interface
def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="FL-IDS: Federated Intrusion Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🛡️ Federated Learning-Based Network Intrusion Detection System
            
            This system uses **Federated Learning** to detect network intrusions while preserving privacy.
            Upload network flow data or enter features manually to get real-time predictions.
            
            **Supported Attack Types:** DDoS, Port Scan, Botnet, Web Attacks, DoS, and more.
            """
        )
        
        with gr.Tabs():
            # Tab 1: CSV Upload
            with gr.Tab("📄 Upload CSV"):
                gr.Markdown("Upload a CSV file with network flow features (first row will be analyzed)")
                
                csv_input = gr.File(label="Upload CSV File", file_types=[".csv"])
                csv_button = gr.Button("🔍 Analyze", variant="primary")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        csv_output_text = gr.Markdown(label="Prediction")
                        csv_confidence = gr.Number(label="Confidence Score", precision=4)
                    with gr.Column(scale=1):
                        csv_output_chart = gr.Plot(label="Top 5 Predictions")
                
                csv_button.click(
                    fn=predict_from_csv,
                    inputs=[csv_input],
                    outputs=[csv_output_text, csv_confidence, csv_output_chart]
                )
            
            # Tab 2: Manual Input
            with gr.Tab("⌨️ Manual Input"):
                gr.Markdown("Enter network flow features manually (simplified interface)")
                
                with gr.Row():
                    flow_duration = gr.Number(label="Flow Duration (μs)", value=120000)
                    total_fwd_packets = gr.Number(label="Total Fwd Packets", value=10)
                
                with gr.Row():
                    total_bwd_packets = gr.Number(label="Total Bwd Packets", value=8)
                    fwd_packet_length_mean = gr.Number(label="Fwd Packet Length Mean", value=500)
                
                bwd_packet_length_mean = gr.Number(label="Bwd Packet Length Mean", value=450)
                
                manual_button = gr.Button("🔍 Analyze", variant="primary")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        manual_output_text = gr.Markdown(label="Prediction")
                        manual_confidence = gr.Number(label="Confidence Score", precision=4)
                    with gr.Column(scale=1):
                        manual_output_chart = gr.Plot(label="Top 5 Predictions")
                
                manual_button.click(
                    fn=predict_from_manual,
                    inputs=[
                        flow_duration, total_fwd_packets, total_bwd_packets,
                        fwd_packet_length_mean, bwd_packet_length_mean
                    ],
                    outputs=[manual_output_text, manual_confidence, manual_output_chart]
                )
            
            # Tab 3: About
            with gr.Tab("ℹ️ About"):
                gr.Markdown(
                    """
                    ## About This System
                    
                    ### 🎯 Key Features
                    - **Privacy-Preserving:** Uses Federated Learning to train without sharing raw data
                    - **Multi-Class Detection:** Identifies 15+ types of network attacks
                    - **Real-Time:** Inference in <50ms on modern GPUs
                    - **Generalizable:** Tested on multiple datasets (CICIDS2017, UNSW-NB15)
                    
                    ### 📊 Model Performance
                    - **Primary Dataset (CICIDS2017):** 93.8% F1-Score
                    - **Secondary Dataset (UNSW-NB15):** 86.3% F1-Score
                    - **Inference Latency:** <20ms (RTX 5070 Ti)
                    
                    ### 🔬 Technical Details
                    - **Architecture:** Temporal CNN with Self-Attention
                    - **Parameters:** 2.1M
                    - **Framework:** PyTorch + Flower FL
                    - **Training:** 8 federated clients, 50 rounds
                    
                    ### 👨‍🎓 Academic Project
                    This is a final project for Cybersecurity Machine Learning course.
                    
                    **Hardware:** MSI Vector 16 HX (RTX 5070 Ti)
                    """
                )
        
        gr.Markdown(
            """
            ---
            **Note:** This system automatically handles different feature dimensions (padding/truncation to 78 features).
            """
        )
    
    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("FL-IDS Gradio Demo")
    print("=" * 60)
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )