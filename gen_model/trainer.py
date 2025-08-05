import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import config

class Trainer:
    """
    A class to encapsulate the training and evaluation loop.
    """
    def __init__(self, model, train_loader, eval_loader, optimizer=None, device=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.optimizer = optimizer if optimizer else optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)

    def _train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for i, batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            
            # Update progress bar
            if (i + 1) % config.LOG_INTERVAL == 0:
                progress_bar.set_postfix({"training_loss": f"{total_loss / (i + 1):.4f}"})
    
    def _init_metrics(self):
        self.metrics = {}
        token_levels = len(self.model.token_level_vocab_sizes)
        for i in range(token_levels):
            self.metrics[f"level_{i}_accuracy"] = 0
        self.metrics[f"total"] = 0
        self.metrics["item_accuracy"] = 0

    def _update_metrics(self, predictions, labels):
        token_levels = len(self.model.token_level_vocab_sizes)
        level_offsets = torch.tensor(self.model.level_offsets, device=self.model.device)
        valid_mask = labels != self.model.pad_token_id
        self.metrics["total"] += valid_mask.sum().item() / token_levels

        for i in range(token_levels):
            level_predictions = predictions[:, i::token_levels]
            level_labels = labels[:, i::token_levels] - level_offsets[i]
            level_mask = valid_mask[:, i::token_levels]
            self.metrics[f"level_{i}_accuracy"] += ((level_predictions == level_labels) & level_mask).sum().item()

        batch_size = labels.size(0)
        labels_item = labels.view(batch_size, -1, token_levels)
        labels_item = labels_item - level_offsets
        predictions_item = predictions.view(batch_size, -1, token_levels)
        valid_item_mask = torch.all(labels_item != self.model.pad_token_id, dim=-1)
        correct_items_mask = torch.all(
            predictions_item[:, :, :token_levels - 1] == labels_item[:, :, :token_levels - 1],
            dim=2
        )
        correct_and_valid_items = (correct_items_mask & valid_item_mask).sum().item()
        self.metrics["item_accuracy"] += correct_and_valid_items

        return self.metrics

    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0
        self._init_metrics()
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()

                predictions = outputs.logits.argmax(dim=-1)
                self._update_metrics(predictions, labels)

        avg_loss = total_loss / len(self.eval_loader)
        print(f"Evaluation Loss: {avg_loss:.4f}")
        
        print(f"===Evaluation Metrics===")
        for key, value in self.metrics.items():
            if "accuracy" in key:
                value /= self.metrics["total"]
            print(f"{key}: {value:.4f}")

        return avg_loss

    def train(self):
        print(f"Starting training on {self.device}...")
        for epoch in range(config.NUM_EPOCHS):
            self._train_epoch(epoch)
            self.evaluate()
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch: int):
        if not os.path.exists(config.OUTPUT_DIR):
            os.makedirs(config.OUTPUT_DIR)
        
        path = os.path.join(config.OUTPUT_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
        }, path)
        print(f"Checkpoint saved to {path}")