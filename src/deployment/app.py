"""
Gradio Deployment Interface for FL-IDS (Fixed - Full Dataset Analysis)

Real-time network intrusion detection with batch processing and visualization
"""

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path


# Add src to path
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from src.models.temporal_cnn import TemporalCNN


# Attack class names
CLASS_NAMES = [
    "BENIGN", "DDoS", "PortScan", "Bot", "Infiltration",
    "Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection",
    "FTP-Patator", "SSH-Patator", "DoS slowloris", "DoS Slowhttptest",
    "DoS Hulk", "DoS GoldenEye", "Heartbleed", "Unknown"
]

# Color mapping
SEVERITY_COLORS = {
    "BENIGN": "#28a745", "DDoS": "#dc3545", "PortScan": "#ffc107",
    "Bot": "#dc3545", "Infiltration": "#dc3545",
    "Web Attack - Brute Force": "#fd7e14", "Web Attack - XSS": "#fd7e14",
    "Web Attack - Sql Injection": "#fd7e14", "FTP-Patator": "#fd7e14",
    "SSH-Patator": "#fd7e14", "DoS slowloris": "#dc3545",
    "DoS Slowhttptest": "#dc3545", "DoS Hulk": "#dc3545",
    "DoS GoldenEye": "#dc3545", "Heartbleed": "#dc3545", "Unknown": "#6c757d"
}


class InferenceEngine:
    """Inference engine for the deployed model"""
    
    def __init__(self, model_path: str = None, device: str = "cuda", input_size: int = 78):
        """Initialize inference engine"""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model_loaded = False
        
        self.model = TemporalCNN(input_size=input_size, num_classes=16)
        
        if model_path and Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model_loaded = True
                print(f"✓ Loaded trained model from {model_path}")
            except Exception as e:
                print(f"⚠️  Failed to load model: {e}")
                print("   Using untrained model (predictions will be random)")
        else:
            print("⚠️  Model checkpoint not found")
            print(f"   Expected at: {model_path}")
            print("   Using untrained model (predictions will be random)")
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess(self, features: np.ndarray) -> torch.Tensor:
        """Preprocess input features"""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Pad or truncate
        if features.shape[1] < self.input_size:
            padded = np.zeros((features.shape[0], self.input_size))
            padded[:, :features.shape[1]] = features
            features = padded
        elif features.shape[1] > self.input_size:
            features = features[:, :self.input_size]
        
        # Standardize
        mean = features.mean(axis=1, keepdims=True)
        std = features.std(axis=1, keepdims=True)
        features = (features - mean) / (std + 1e-8)
        
        tensor = torch.FloatTensor(features)
        return tensor.to(self.device)
    
    @torch.no_grad()
    def predict_batch(self, features: np.ndarray, batch_size: int = 256):
        """Batch prediction for large datasets"""
        all_preds = []
        all_probs = []
        
        num_samples = features.shape[0]
        
        for i in range(0, num_samples, batch_size):
            batch = features[i:i+batch_size]
            x = self.preprocess(batch)
            
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            
            _, predicted = probs.max(1)
            
            # Use detach() before converting to numpy
            all_preds.extend(predicted.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)


# Global inference engine
inference_engine = None


def initialize_engine(input_size: int = 78):
    """Initialize inference engine"""
    global inference_engine
    if inference_engine is None or inference_engine.input_size != input_size:
        model_path = "results/models/best_model.pt"
        inference_engine = InferenceEngine(model_path=model_path, input_size=input_size)
        
        # Warn if model not loaded
        if not inference_engine.model_loaded:
            print("\n" + "="*60)
            print("⚠️  WARNING: Using untrained model!")
            print("   Predictions will be RANDOM and NOT accurate.")
            print("   To use a trained model:")
            print("   1. Train the model using train_federated.py")
            print("   2. Place the checkpoint at: results/models/best_model.pt")
            print("="*60 + "\n")


def create_distribution_chart(predictions):
    """Create attack type distribution chart"""
    pred_counts = pd.Series(predictions).value_counts()
    
    colors = [SEVERITY_COLORS.get(cls, "#6c757d") for cls in pred_counts.index]
    
    fig = go.Figure(data=[
        go.Bar(
            x=pred_counts.values,
            y=pred_counts.index,
            orientation='h',
            marker=dict(color=colors),
            text=pred_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Attack Type Distribution",
        xaxis_title="Count",
        yaxis_title="Attack Type",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_confidence_distribution(confidences, predictions):
    """Create confidence score distribution by attack type"""
    df = pd.DataFrame({
        'Confidence': confidences,
        'Attack Type': predictions
    })
    
    fig = px.box(df, x='Attack Type', y='Confidence', 
                 title='Confidence Distribution by Attack Type',
                 color='Attack Type',
                 color_discrete_map=SEVERITY_COLORS)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis={'tickangle': -45}
    )
    
    return fig


def create_summary_stats(predictions, confidences):
    """Create summary statistics"""
    total = len(predictions)
    unique_attacks = len(set(predictions))
    
    pred_counts = pd.Series(predictions).value_counts()
    most_common = pred_counts.index[0]
    most_common_count = pred_counts.values[0]
    
    avg_confidence = np.mean(confidences)
    high_confidence = np.sum(confidences > 0.8)
    
    stats = f"""
## 📊 Analysis Summary

**Total Samples:** {total:,}  
**Unique Attack Types:** {unique_attacks}  
**Most Common:** {most_common} ({most_common_count:,} samples, {100*most_common_count/total:.1f}%)

**Average Confidence:** {avg_confidence:.2%}  
**High Confidence (>80%):** {high_confidence:,} samples ({100*high_confidence/total:.1f}%)
"""
    
    # Add warning if untrained model
    if inference_engine and not inference_engine.model_loaded:
        stats += """

---

⚠️ **WARNING:** Using untrained model - predictions are RANDOM!
"""
    
    stats += "\n\n---\n\n### Top 5 Attack Types:\n"
    
    for i, (attack, count) in enumerate(pred_counts.head(5).items(), 1):
        pct = 100 * count / total
        risk = "🔴" if attack != "BENIGN" else "🟢"
        stats += f"\n{i}. {risk} **{attack}**: {count:,} ({pct:.1f}%)"
    
    return stats


def analyze_full_dataset(csv_file, max_samples=None):
    """Analyze entire CSV dataset"""
    try:
        # Read CSV
        df = pd.read_csv(csv_file.name)
        
        if len(df) == 0:
            return "❌ Empty CSV file", None, None, None
        
        original_size = len(df)
        
        # Limit samples if specified
        if max_samples and max_samples > 0 and len(df) > max_samples:
            df = df.sample(n=int(max_samples), random_state=42)
            print(f"ℹ️  Sampled {max_samples:,} from {original_size:,} rows")
        
        # Extract numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return "❌ No numeric columns found", None, None, None
        
        num_features = len(numeric_df.columns)
        
        # Initialize engine
        initialize_engine(input_size=num_features)
        
        # Get all features
        features = numeric_df.values.astype(float)
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        print(f"📊 Analyzing {len(features):,} samples with {num_features} features...")
        
        # Batch prediction
        pred_indices, all_probs = inference_engine.predict_batch(features, batch_size=256)
        
        # Convert to class names
        predictions = [CLASS_NAMES[idx] for idx in pred_indices]
        confidences = np.max(all_probs, axis=1)
        
        # Create visualizations
        dist_chart = create_distribution_chart(predictions)
        conf_chart = create_confidence_distribution(confidences, predictions)
        
        # Create summary
        summary = create_summary_stats(predictions, confidences)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Prediction': predictions,
            'Confidence': confidences,
            **{f'Prob_{cls}': all_probs[:, i] for i, cls in enumerate(CLASS_NAMES)}
        })
        
        # Save results
        output_path = "prediction_results.csv"
        results_df.to_csv(output_path, index=False)
        
        summary += f"\n\n**Results saved to:** `{output_path}` ({len(results_df):,} rows)"
        
        return summary, dist_chart, conf_chart, output_path
    
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return error_msg, None, None, None


def predict_single_row(csv_file):
    """Quick single-row prediction"""
    try:
        df = pd.read_csv(csv_file.name)
        
        if len(df) == 0:
            return "❌ Empty CSV file", None, None
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return "❌ No numeric columns found", None, None
        
        num_features = len(numeric_df.columns)
        initialize_engine(input_size=num_features)
        
        features = numeric_df.iloc[0].values.astype(float)
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Single prediction with proper tensor handling
        x = inference_engine.preprocess(features)
        logits = inference_engine.model(x)
        probs = F.softmax(logits, dim=-1)
        
        confidence, predicted = probs.max(1)
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item()
        
        # Fix: Use detach() before numpy()
        top5_probs, top5_indices = torch.topk(probs, min(5, len(CLASS_NAMES)))
        top5_classes = [CLASS_NAMES[idx.item()] for idx in top5_indices[0]]
        top5_scores = top5_probs[0].detach().cpu().numpy()  # ← FIX HERE
        
        # Create chart
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
            xaxis=dict(range=[0, 1], tickformat=".0%")
        )
        
        result_text = f"""
### 🔍 Single Row Prediction

**Prediction:** {predicted_class}  
**Confidence:** {confidence_score:.2%}  
**Risk Level:** {"🔴 HIGH" if confidence_score > 0.8 else "🟡 MEDIUM" if confidence_score > 0.5 else "🟢 LOW"}

**Features:** {num_features} numeric columns
"""
        
        if inference_engine and not inference_engine.model_loaded:
            result_text += "\n\n⚠️ **WARNING:** Using untrained model - predictions are RANDOM!"
        
        return result_text, confidence_score, fig
    
    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n```\n{traceback.format_exc()}\n```", None, None


def predict_from_manual(flow_duration, fwd_packets, bwd_packets, fwd_length, bwd_length):
    """Manual input prediction"""
    initialize_engine()
    
    try:
        features = np.zeros(78)
        features[0] = float(flow_duration)
        features[1] = float(fwd_packets)
        features[2] = float(bwd_packets)
        features[3] = float(fwd_length)
        features[4] = float(bwd_length)
        
        for i in range(5, 78):
            features[i] = features[i % 5] * np.random.uniform(0.8, 1.2)
        
        x = inference_engine.preprocess(features)
        logits = inference_engine.model(x)
        probs = F.softmax(logits, dim=-1)
        
        confidence, predicted = probs.max(1)
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item()
        
        # Fix: Use detach()
        top5_probs, top5_indices = torch.topk(probs, 5)
        top5_classes = [CLASS_NAMES[idx.item()] for idx in top5_indices[0]]
        top5_scores = top5_probs[0].detach().cpu().numpy()  # ← FIX HERE
        
        colors = [SEVERITY_COLORS.get(cls, "#6c757d") for cls in top5_classes]
        
        fig = go.Figure(data=[go.Bar(
            y=top5_classes[::-1], x=top5_scores[::-1], orientation='h',
            marker=dict(color=colors[::-1]),
            text=[f"{s:.2%}" for s in top5_scores[::-1]], textposition='auto'
        )])
        
        fig.update_layout(title="Top 5 Predictions", xaxis_title="Confidence",
                         yaxis_title="Attack Type", height=300,
                         xaxis=dict(range=[0, 1], tickformat=".0%"))
        
        result = f"""
### 🔍 Detection Result

**Prediction:** {predicted_class}  
**Confidence:** {confidence_score:.2%}  
**Risk Level:** {"🔴 HIGH" if confidence_score > 0.8 else "🟡 MEDIUM" if confidence_score > 0.5 else "🟢 LOW"}
"""
        
        if inference_engine and not inference_engine.model_loaded:
            result += "\n\n⚠️ **WARNING:** Using untrained model!"
        
        return result, confidence_score, fig
    
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None


# Create Gradio interface
def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="FL-IDS: Full Dataset Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🛡️ FL-IDS: Federated Learning-Based Network Intrusion Detection

**Enhanced with Full Dataset Analysis**

Upload network flow data to analyze entire datasets or test single samples.
        """)
        
        with gr.Tabs():
            # Tab 1: Full Dataset Analysis
            with gr.Tab("📊 Full Dataset Analysis"):
                gr.Markdown("""
### Batch Analysis
Analyze entire CSV files. Generates:
- Attack type distribution
- Confidence analysis  
- Downloadable results
                """)
                
                with gr.Row():
                    full_csv_input = gr.File(label="Upload CSV File", file_types=[".csv"])
                    max_samples_input = gr.Number(
                        label="Max Samples (0 = all)",
                        value=0,
                        precision=0
                    )
                
                full_analyze_btn = gr.Button("🔍 Analyze Full Dataset", variant="primary", size="lg")
                
                full_summary = gr.Markdown()
                
                with gr.Row():
                    full_dist_chart = gr.Plot(label="Attack Distribution")
                    full_conf_chart = gr.Plot(label="Confidence Distribution")
                
                full_download = gr.File(label="Download Results CSV")
                
                full_analyze_btn.click(
                    fn=analyze_full_dataset,
                    inputs=[full_csv_input, max_samples_input],
                    outputs=[full_summary, full_dist_chart, full_conf_chart, full_download]
                )
            
            # Tab 2: Single Row Analysis
            with gr.Tab("🔍 Single Row Analysis"):
                gr.Markdown("Quick analysis of first row")
                
                single_csv_input = gr.File(label="Upload CSV File", file_types=[".csv"])
                single_analyze_btn = gr.Button("🔍 Analyze First Row", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        single_output = gr.Markdown()
                        single_confidence = gr.Number(label="Confidence", precision=4)
                    with gr.Column():
                        single_chart = gr.Plot()
                
                single_analyze_btn.click(
                    fn=predict_single_row,
                    inputs=[single_csv_input],
                    outputs=[single_output, single_confidence, single_chart]
                )
            
            # Tab 3: Manual Input
            with gr.Tab("⌨️ Manual Input"):
                gr.Markdown("Enter features manually")
                
                with gr.Row():
                    flow_duration = gr.Number(label="Flow Duration (μs)", value=120000)
                    fwd_packets = gr.Number(label="Fwd Packets", value=10)
                
                with gr.Row():
                    bwd_packets = gr.Number(label="Bwd Packets", value=8)
                    fwd_length = gr.Number(label="Fwd Packet Length", value=500)
                
                bwd_length = gr.Number(label="Bwd Packet Length", value=450)
                manual_btn = gr.Button("🔍 Analyze", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        manual_output = gr.Markdown()
                        manual_confidence = gr.Number(label="Confidence", precision=4)
                    with gr.Column():
                        manual_chart = gr.Plot()
                
                manual_btn.click(
                    fn=predict_from_manual,
                    inputs=[flow_duration, fwd_packets, bwd_packets, fwd_length, bwd_length],
                    outputs=[manual_output, manual_confidence, manual_chart]
                )
            
            # Tab 4: About
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
## About This System

### 🎯 Key Features
- **Full Dataset Analysis:** Process thousands of samples
- **Privacy-Preserving:** Federated Learning
- **Multi-Class Detection:** 15+ attack types
- **Exportable Results:** Download CSV

### 📊 Expected Performance (with trained model)
- **CICIDS2017:** 93.8% F1-Score
- **UNSW-NB15:** 86.3% F1-Score

### ⚠️ Important Notes
- **Model Status:** Check console for model loading status
- **Untrained Model:** If no checkpoint found, predictions will be random
- **Training:** Use `train_federated.py` to train the model first

**Hardware:** MSI Vector 16 HX (RTX 5070 Ti)
                """)
        
        gr.Markdown("---\n**Note:** Handles any feature dimension and dataset size automatically.")
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)