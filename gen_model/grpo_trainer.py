import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import config
import copy

from reward import RewardCalculator

class GRPOTrainer:
    """
    A class to encapsulate the training and evaluation loop for GPO (Generative Policy Optimization).
    This trainer uses a policy gradient approach to fine-tune the generative model.
    """
    def __init__(self, model, train_loader, eval_loader, 
                 tokenizer,
                 K_SAMPLES=5, GENERATE_MAX_LENGTH=80,
                 BETA = 0.01, CLIP_PARAM=0.2,
                 optimizer=None, device=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.K_SAMPLES = K_SAMPLES
        self.GENERATE_MAX_LENGTH = GENERATE_MAX_LENGTH
        self.optimizer = optimizer if optimizer else optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)

        self.beta = BETA
        self.clip_param = CLIP_PARAM
        self.ref_update_per_iter = 4
        self.mu = 3


        self.RewardCalculator = RewardCalculator(tokenizer, config.CONFIG_DIR, config.MODEL_PATH, self.device)

    def _get_action_log_probs(self, logits, labels):
        batch_size, seq_len = logits.shape[0], logits.shape[1]
        
        t_logits = logits / self.model.loss_temperature
        log_probs_dist = torch.nn.functional.log_softmax(t_logits, dim=-1)

        token_levels = len(self.model.token_level_vocab_sizes)
        level_indices = torch.arange(seq_len, device=self.device) % token_levels
        level_indices = level_indices.unsqueeze(0).expand(batch_size, -1)

        level_offsets_tensor = torch.tensor(self.model.level_offsets, device=self.device, dtype=labels.dtype)
        offset_tensor = level_offsets_tensor[level_indices]

        local_labels = labels - offset_tensor
        local_labels = torch.clamp(local_labels, min=0)
        
        action_log_probs = log_probs_dist.gather(dim=-1, index=local_labels.unsqueeze(-1)).squeeze(-1)
        
        valid_token_mask = labels != self.model.pad_token_id
        action_log_probs = action_log_probs.masked_fill(~valid_token_mask, 0.0)

        return action_log_probs
    
    def check_tensor(self, x):
        if not torch.isfinite(x).all():
            print(f"Warning: NaN or Inf!")
            print("NaN count:", torch.isnan(x).sum().item())
            print("Inf count:", torch.isinf(x).sum().item())
            print(x)
            raise ValueError(f"NaN or Inf")
        return x

    def _train_epoch(self):
        self.model.train()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        
        for iter, batch in progress_bar:
            if iter % self.ref_update_per_iter == 0:
                ref_model = copy.deepcopy(self.model)
                ref_model.to(self.device)
                ref_model.eval()
                for param in ref_model.parameters():
                    param.requires_grad = False

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

            outputs = self.model(input_ids=expanded_input_ids, attention_mask=expanded_attention_mask, labels=sampled_sequences)
            logits = outputs.logits # (batch_size * K, seq_len, max_vocab_size)
            log_probs = self._get_action_log_probs(logits, sampled_sequences) # (batch_size * K, seq_len)
            old_log_probs = log_probs.detach() # (batch_size * K, seq_len)

            with torch.no_grad():
                ref_outputs = ref_model(input_ids=expanded_input_ids, attention_mask=expanded_attention_mask, labels=sampled_sequences)
                ref_logits = ref_outputs.logits
            ref_log_probs = self._get_action_log_probs(ref_logits, sampled_sequences) # (batch_size * K)

            token_mask = sampled_sequences != self.model.pad_token_id
            score_mask = (torch.arange(sampled_sequences.shape[1]) % len(self.model.token_level_vocab_sizes)) != len(self.model.token_level_vocab_sizes) - 1
            score_mask = score_mask.unsqueeze(0).to(token_mask.device)
            mask = token_mask & score_mask
            
            with torch.autograd.set_detect_anomaly(True):
                for grpo_iter in range(self.mu):
                    outputs = self.model(input_ids=expanded_input_ids, attention_mask=expanded_attention_mask, labels=sampled_sequences)
                    logits = outputs.logits # (batch_size * K, seq_len, max_vocab_size)
                    log_probs = self._get_action_log_probs(logits, sampled_sequences) # (batch_size * K, seq_len)
                    rewards = self.RewardCalculator(sampled_sequences, candidate_items, expanded_target_label, self.K_SAMPLES) # (batch_size * K)
                    rewards_2d = rewards['total_rewards'].view(batch_size, self.K_SAMPLES).detach()
                    rewards_mean = rewards_2d.mean(dim=1, keepdim=True).detach()
                    rewards_std = rewards_2d.std(dim=1, keepdim=True).detach()
                    advantages = (rewards_2d - rewards_mean) / (rewards_std + 1e-4) # (batch_size, K)
                    advantages = advantages.view(-1, 1)

                    ratio = torch.exp(log_probs - old_log_probs) # (batch_size * K, seq_len)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
                    surrogate_loss = torch.min(surr1, surr2)

                    kl = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1 #(batch_size * K, seq_len)

                    pre_token_loss = surrogate_loss - self.beta * kl

                    valid_mask = token_mask.sum(dim=1) > 0

                    if valid_mask.sum() == 0:
                        continue
                    else:
                        pre_token_loss = pre_token_loss[valid_mask]
                        token_mask = token_mask[valid_mask]
                        seq_loss = (pre_token_loss * token_mask).sum(dim=1) / token_mask.sum(dim=1)
                        loss = -seq_loss.mean()

                    loss = self.check_tensor(loss)

                    with open("log.log", "a") as f:
                        f.write(f"Iter: {iter},\t GRPO_Iter: {grpo_iter},\t loss: {loss:.4f}\n "
                                f"total_rewards:{rewards["total_rewards"].mean()}\t integrity_rewards:{rewards["integrity_rewards"].mean()}\t "
                                f"count_rewards:{rewards["count_rewards"].mean()}\t sim_loss:{rewards["sim_loss"].mean()}\n")

                    self.optimizer.zero_grad()
                    loss.backward()    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

            if iter % 200 == 0:
                # self.evaluate()
                self.save_checkpoint(iter)
            
            


    def evaluate(self) -> float:
        self.model.eval()
        sim_loss = 0
        sim_rewards = 0
        integrity_rewards = 0
        count_rewards = 0
        total_rewards = 0
        cnt = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                candidate_items = batch["input_items"]
                target_label = batch["target_label"].to(self.device)

                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.GENERATE_MAX_LENGTH,
                    strategy='greedy'
                )
                rewards = self.RewardCalculator(output, candidate_items, target_label, 1)
                sim_loss += rewards["sim_loss"].sum()
                integrity_rewards += rewards["integrity_rewards"].sum()
                count_rewards += rewards["count_rewards"].sum()
                total_rewards += rewards["total_rewards"].sum()
                cnt += len(rewards["total_rewards"])

        with open("log.log", "a") as f:
            f.write(f"===Evaluation===\n")
            f.write(f"sim_loss: {sim_loss / cnt}\n")
            f.write(f"integrity_rewards: {integrity_rewards / cnt}\n")
            f.write(f"count_rewards: {count_rewards / cnt}\n")
            f.write(f"total_rewards: {total_rewards / cnt}\n")

        return total_rewards / cnt

    def train(self):
        print(f"Starting GPO training on {self.device}...")
        self._train_epoch()

    def save_checkpoint(self, iter: int):
        if not os.path.exists(config.GRPO_OUTPUT_DIR):
            os.makedirs(config.GRPO_OUTPUT_DIR)
        
        path = os.path.join(config.GRPO_OUTPUT_DIR, f"model_iter_{iter+1}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iter': iter,
        }, path)
        print(f"Checkpoint saved to {path}")
