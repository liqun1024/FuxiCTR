import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import config

from reward import RewardCalculator

class GRPOTrainer:
    """
    A class to encapsulate the training and evaluation loop for GPO (Generative Policy Optimization).
    This trainer uses a policy gradient approach to fine-tune the generative model.
    """
    def __init__(self, model, train_loader, eval_loader, 
                 tokenizer,
                 K_SAMPLES=5, GENERATE_MAX_LENGTH=80,
                 optimizer=None, device=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.K_SAMPLES = K_SAMPLES
        self.GENERATE_MAX_LENGTH = GENERATE_MAX_LENGTH
        self.optimizer = optimizer if optimizer else optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)

        self.RewardCalculator = RewardCalculator(tokenizer, config.CONFIG_DIR, config.MODEL_PATH, self.device)

    def _get_action_log_probs(self, logits, labels):
        batch_size = logits.shape[0]
        log_probs = torch.zeros(batch_size, device=self.device)

        token_levels = len(self.model.token_level_vocab_sizes)

        t_logits = logits / self.model.loss_temperature
        log_probs_dist = torch.nn.functional.log_softmax(t_logits, dim=-1)

        for i in range(token_levels):
            head_labels = labels[:, i::token_levels].clone() # (batch_size * K, group_seq_len)
            head_log_probs_dist = log_probs_dist[:, i::token_levels, :] # (batch_size * K, group_seq_len, max_vocab_size)

            if head_labels.numel() == 0:
                continue

            valid_token_mask = head_labels != self.model.pad_token_id
            head_labels[valid_token_mask] -= self.model.level_offsets[i]

            action_log_probs = head_log_probs_dist.gather(dim=-1, index=head_labels.unsqueeze(-1)).squeeze(-1) # (batch_size * K, group_seq_len)

            action_log_probs = action_log_probs.masked_fill(~valid_token_mask, 0.0) # Set invalid tokens to 0 log probability
            log_probs += action_log_probs.sum(dim=1) # Sum over the sequence length

        return log_probs

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for i, batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            candidate_items = batch["input_items"]
            target_label = batch["target_label"].to(self.device)
            batch_size = input_ids.shape[0]

            expanded_input_ids = input_ids.repeat_interleave(self.K_SAMPLES, dim=0) # (batch_size * K, seq_len)
            expanded_attention_mask = attention_mask.repeat_interleave(self.K_SAMPLES, dim=0)
            expanded_target_label = target_label.repeat_interleave(self.K_SAMPLES, dim=0)

            with torch.no_grad():
                sampled_sequences = self.model.generate(
                    input_ids=expanded_input_ids,
                    attention_mask=expanded_attention_mask,
                    max_length=self.GENERATE_MAX_LENGTH,
                    strategy='sampling',
                    top_p=0.9,
                    temperature=0.7
                )

            rewards = self.RewardCalculator(sampled_sequences, candidate_items, expanded_target_label, self.K_SAMPLES) # (batch_size * K)

            outputs = self.model(input_ids=expanded_input_ids, attention_mask=expanded_attention_mask, labels=sampled_sequences)
            logits = outputs.logits # (batch_size * K, seq_len, max_vocab_size)
            log_probs = self._get_action_log_probs(logits, sampled_sequences) # (batch_size * K)

            rewards_2d = rewards['total_rewards'].view(batch_size, self.K_SAMPLES).detach()
            log_probs_2d = log_probs.view(batch_size, self.K_SAMPLES)
            advantages = rewards_2d - rewards_2d.mean(dim=1, keepdim=True).detach() # (batch_size, K)
            policy_loss = -(advantages * log_probs_2d).mean()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            total_loss += policy_loss.item()
            
            if (i + 1) % config.LOG_INTERVAL == 0:
                progress_bar.set_postfix({"policy_loss": f"{total_loss / (i + 1):.4f}"})

    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0
        self._init_metrics()
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.GENERATE_MAX_LENGTH,
                    strategy='greedy'
                )
                rewards = self.RewardCalculator(output, input_ids)
        
        print(f"===Evaluation===")
        for key, values in rewards.items():
            print(f"{key}: {(sum(values) / len(values)):.4f}")

        return rewards['total_rewards']

    def train(self):
        print(f"Starting GPO training on {self.device}...")
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
