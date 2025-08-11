
"""
GMoE (Gated Mixture of Experts) Model Evaluation Script

This script implements the evaluation pipeline for a trained GMoE model, including:
- Model loading
- Data preparation
- Performance evaluation
- Metric calculation and reporting

"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

# Local imports - ensure these are in your PYTHONPATH
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from dataloader.build_dataloader import get_dataloader
from model.GMoE import GMoE


class GMoeEvaluator:
    """GMoE model evaluation class."""

    def __init__(self, config):
        """Initialize evaluator with configuration."""
        self.config = config
        self.device = torch.device(config["device"])
        self._setup_logging()
        self.model = self._load_model()
        self.dataloader = self._get_dataloader()

    def _setup_logging(self):
        """Configure logging system."""
        os.makedirs(Path(self.config["log_file"]).parent, exist_ok=True)
        
        self.logger = logging.getLogger("GMoeEvaluator")
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        fh = logging.FileHandler(self.config["log_file"])
        fh.setFormatter(formatter)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info(f"Initialized evaluator with config: {self.config}")

    def _load_model(self):
        """Load trained model from checkpoint."""
        try:
            model = GMoE(
                num_experts=self.config["num_experts"],
                tasks=self.config["num_tasks"],
                device=self.device
            )
            
            state_dict = torch.load(
                self.config["model_path"],
                map_location=self.device
            )
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.logger.info(f"Successfully loaded model from {self.config['model_path']}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def _get_dataloader(self):
        """Prepare and return data loader for evaluation."""
        return get_dataloader(
            batch_size=self.config["batch_size"],
            train_csv_path=self.config["train_csv_path"],
            test_csv_path=self.config["test_csv_path"],
        )

    def evaluate(self):
        """Run full evaluation pipeline."""
        self.logger.info("Starting evaluation...")
        
        # Initialize containers for predictions and labels
        true_labels = np.empty([0])
        pred_labels = np.empty([0])
        pred_scores = np.empty((0, self.config["num_classes"]))
        
        # Evaluation loop
        inference_time = 0.0
        with torch.no_grad():
            for batch in tqdm(
                self.dataloader['test'],
                desc="Evaluating",
                unit="batch",
                disable=not self.config["verbose"]
            ):
                x, y = batch
                x = x.permute(0, 2, 1, 3, 4).to(self.device)
                y = y.squeeze().to(self.device)
                
                start_time = time.time()
                logits = self.model(x)[0]
                inference_time += time.time() - start_time
                
                # Get predictions
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Store results
                true_labels = np.append(true_labels, y.cpu().numpy())
                pred_labels = np.append(pred_labels, preds.cpu().numpy())
                pred_scores = np.vstack((pred_scores, logits.cpu().numpy()))
        
        # Calculate metrics
        metrics = self._calculate_metrics(true_labels, pred_labels, pred_scores)
        
        # Log results
        self._log_results(metrics, inference_time / len(self.dataloader['test']))
        
        return metrics

    def _calculate_metrics(self, true_labels, pred_labels, pred_scores):
        """Calculate all evaluation metrics."""
        return {
            "accuracy": accuracy_score(true_labels, pred_labels),
            "f1_scores": f1_score(true_labels, pred_labels, average=None),
            "f1_macro": f1_score(true_labels, pred_labels, average='macro'),
            "f1_weighted": f1_score(true_labels, pred_labels, average='weighted'),
            "auc": roc_auc_score(
                true_labels,
                F.softmax(torch.from_numpy(pred_scores), dim=1),
                multi_class='ovr'
            ),
            "class_distribution": np.bincount(true_labels.astype(int)),
        }

    def _log_results(self, metrics, avg_inference_time):
        """Log evaluation results."""
        self.logger.info("\n===== Evaluation Results =====")
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"AUC: {metrics['auc']:.4f}")
        self.logger.info(f"F1 Macro: {metrics['f1_macro']:.4f}")
        self.logger.info(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
        
        self.logger.info("\nPer-class F1 Scores:")
        for i, score in enumerate(metrics["f1_scores"]):
            self.logger.info(f"  Class {i}: {score:.4f}")
            
        self.logger.info(f"\nClass Distribution: {metrics['class_distribution']}")
        self.logger.info(f"\nAverage Inference Time: {avg_inference_time:.4f} seconds/batch")
        self.logger.info("=" * 30)


def main():
    """Main evaluation function."""
    config = {
        # Model configuration
        "num_experts": 10,
        "num_tasks": 1,
        "num_classes": 4,
        "model_path": "model_weight/best_model_epoch_92_auc_0.6553.pth",
        
        # Data configuration
        "batch_size": 4,
        "train_csv_path": "train.csv",
        "test_csv_path": "test.csv",
        
        # System configuration
        "device": "cuda:1" if torch.cuda.is_available() else "cpu",
        "log_file": "test_GMoE_fold_1.log",
        "verbose": True,
    }

    # Initialize and run evaluation
    evaluator = GMoeEvaluator(config)
    evaluator.evaluate()


if __name__ == '__main__':
    start_time = datetime.now()
    print(f"Evaluation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    main()
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Evaluation completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")