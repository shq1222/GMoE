
"""
GMoE (Gating Mixture of Experts) Model Training Script

This script implements the training pipeline for the GMoE model, including:
- Data loading and preprocessing
- Model initialization
- Training loop with validation
- Performance metrics tracking
- Model checkpointing

"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Local imports
from dataloader.build_dataloader import get_dataloader
from losses.ice_loss_KL import ice_loss_KL
from model.GMoE import GMoE

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("GMoE_training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class GMoeTrainer:
    """GMoE model training and evaluation class."""

    def __init__(self, config):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = torch.device(f"cuda:{config['device_num']}")
        self._setup_directories()
        self._init_model()
        self._init_optimizer()
        self.dataloader = self._get_dataloader()
        self.dataset_sizes = {
            x: len(self.dataloader[x].dataset) for x in ["train", "test"]
        }

    def _setup_directories(self):
        """Create necessary directories for saving models and logs."""
        os.makedirs(self.config["save_dir"], exist_ok=True)
        os.makedirs(os.path.dirname(self.config["log_file"]), exist_ok=True)

    def _init_model(self):
        """Initialize model and move to appropriate device."""
        self.model = GMoE(
            num_experts=self.config["num_experts"],
            num_tasks=self.config["num_tasks"],
            device=self.device,
        ).to(self.device)
        logger.info(f"Model initialized on {self.device}")

    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config["lr"],
            momentum=0.9,
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.config["scheduler_step_size"],
            gamma=self.config["scheduler_gamma"],
        )
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def _get_dataloader(self):
        """Initialize and return data loaders."""
        return get_dataloader(
            batch_size=self.config["batch_size"],
            train_csv_path=self.config["train_csv_path"],
            test_csv_path=self.config["test_csv_path"],
        )

    def train_epoch(self, phase):
        """Train or evaluate for one epoch."""
        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        t1_loss = 0.0
        t1_trues = np.empty([0])
        t1_preds = np.empty([0])
        y_scores = np.empty((0, self.config["num_classes"]))
        inference_time = 0.0
        inference_batches = 0

        for batch in tqdm(
            self.dataloader[phase],
            desc=f"{phase.capitalize()} Epoch",
            disable=not self.config["verbose"],
        ):
            if batch is None:
                continue

            x, y1 = batch
            x = x.permute(0, 2, 1, 3, 4).to(self.device)
            y1 = y1.squeeze().to(self.device)

            start_time = time.time()
            with torch.set_grad_enabled(phase == "train"):
                logits = self.model(x)[0]
                loss = ice_loss_KL(
                    x,
                    logits,
                    y1,
                    self.config["num_classes"],
                    self.device,
                    annealing_coef=self.config["lamb"],
                )

                if phase == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if phase == "test":
                inference_time += time.time() - start_time
                inference_batches += 1

            # Update metrics
            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            t1_loss += loss.item() * batch_size

            y_pred = torch.max(nn.functional.softmax(logits, dim=1), dim=1)[1]
            t1_trues = np.append(t1_trues, y1.data.cpu())
            t1_preds = np.append(t1_preds, y_pred.detach().cpu().numpy())
            y_scores = np.vstack((y_scores, logits.detach().cpu().numpy()))

        # Calculate epoch metrics
        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_t1_loss = t1_loss / self.dataset_sizes[phase]

        metrics = {
            "loss": epoch_loss,
            "t1_loss": epoch_t1_loss,
            "accuracy": accuracy_score(t1_trues, t1_preds),
            "f1_macro": f1_score(t1_trues, t1_preds, average="macro"),
            "f1_weighted": f1_score(t1_trues, t1_preds, average="weighted"),
            "auc": roc_auc_score(
                t1_trues,
                nn.functional.softmax(torch.from_numpy(y_scores), dim=1),
                multi_class="ovr",
            ),
        }

        if phase == "test" and inference_batches > 0:
            metrics["avg_inference_time"] = inference_time / inference_batches

        return metrics

    def save_checkpoint(self, epoch, auc, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_auc": auc,
        }

        filename = f"checkpoint_epoch_{epoch}.pth"
        if is_best:
            filename = f"best_model_auc_{auc:.4f}.pth"

        torch.save(checkpoint, os.path.join(self.config["save_dir"], filename))
        logger.info(f"Saved checkpoint: {filename}")

    def train(self):
        """Run full training pipeline."""
        logger.info("Starting training...")
        logger.info(f"Configuration: {self.config}")

        best_auc = 0.0
        training_metrics = {"train": [], "test": []}

        for epoch in range(self.config["num_epochs"]):
            epoch_start_time = time.time()
            logger.info(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")

            for phase in ["train", "test"]:
                metrics = self.train_epoch(phase)

                # Log metrics
                logger.info(
                    f"[{phase.upper()}] Loss: {metrics['loss']:.4f}, "
                    f"Accuracy: {metrics['accuracy']:.4f}, "
                    f"AUC: {metrics['auc']:.4f}"
                )

                if phase == "test":
                    if metrics["auc"] > best_auc:
                        best_auc = metrics["auc"]
                        self.save_checkpoint(epoch + 1, best_auc, is_best=True)
                    training_metrics["test"].append(metrics)

                    if "avg_inference_time" in metrics:
                        logger.info(
                            f"Inference time: {metrics['avg_inference_time']:.4f}s/batch"
                        )

            self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch time: {epoch_time:.2f} seconds")

        logger.info(f"Training complete. Best AUC: {best_auc:.4f}")
        return training_metrics


def main():
    """Main training function."""
    config = {
        "device_num": 0,
        "save_dir": "fold1",
        "log_file": "GMoE_fold1.log",
        "num_experts": 10,
        "lamb": 0.01,
        "num_epochs": 100,
        "num_classes": 4,
        "batch_size": 4,
        "num_tasks": 1,
        "lr": 0.001,
        "scheduler_step_size": 20,
        "scheduler_gamma": 0.1,
        "train_csv_path": "train.csv",
        "test_csv_path": "test.csv",
        "verbose": True,
    }

    # Initialize and run training
    trainer = GMoeTrainer(config)
    trainer.train()


if __name__ == "__main__":
    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    logger.info(f"Training started at {start_time}")
    main()
    end_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    logger.info(f"Training completed at {end_time}")