import os
import torch

from SIM import SIM
from ..fuxictr.utils import load_config
from ..fuxictr.features import FeatureMap

class RewardCalculator:
    def __init__(self, tokenizer, config_dir, model_path, device):
        self.tokenizer = tokenizer
        self.device = device
        self.SIM = self._init_SIM(config_dir, model_path, device)

    def _init_SIM(self, config_dir, model_path, device):
        params = load_config(config_dir, "SIM")
        params['gpu'] = device.index if device.type == 'cuda' else -1
        
        # Build feature_map
        data_dir = os.path.join(params['data_root'], params['dataset_id'])
        feature_map_json = os.path.join(data_dir, "feature_map.json")
        feature_map = FeatureMap(params['dataset_id'], data_dir)
        feature_map.load(feature_map_json, params)

        model = SIM(feature_map, **params)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        return model
    
    def _process_seq(self, seq, select_seq=False):
        processed_sequences = []
        for s in seq:
            if not select_seq:
                processed_seq = s[:-2] + [s[-1]] # delete <special_token> (sperate history and target)
            processed_sequences.append(processed_seq)

        max_len = max(len(s) for s in processed_sequences) if processed_sequences else 0
        
        inputs = []
        masks = []
        for s in processed_sequences:
            # 左填充：在前面补0
            padding = [0] * (max_len - len(s))
            padded_seq = padding + s
            inputs.append(padded_seq)
            # mask: 原始位置为1，填充位置为0
            mask = [0] * (max_len - len(s)) + [1] * len(s)
            masks.append(mask)
        return torch.tensor(inputs, dtype=torch.long, device=self.device), torch.tensor(masks, dtype=torch.bool, device=self.device)

    def __call__(self, generated_ids, candidate_ids):
        batch_size = generated_ids.shape[0]
        generated_items, missing_items = self.tokenizer.decode_batch(generated_ids, has_similarity=True)
        candidate_items, _ = self.tokenizer.decode_batch(candidate_ids)

        rewards_item = ['total_rewards', 'integrity_rewards', 'count_rewards', 'sim_rewards']
        rewards = {key: [] for key in rewards_item}

        for i, (gen_items, cand_items) in enumerate(zip(generated_items, candidate_items)):
            if missing_items[i] > 0:
                integrity_rewards = len(gen_items) / (len(gen_items) + missing_items[i])
            else:
                integrity_rewards = 1.0

            gen_items_unique = set(gen_items)
            valid_items = len(gen_items)
            for item in gen_items_unique:
                diff = abs(gen_items.count(item) - cand_items.count(item))
                valid_items -= diff
            count_rewards = max(valid_items, 0) / len(gen_items) if gen_items else 0

            if not gen_items:
                rewards.append(0)
                continue
            item, item_mask = self._process_seq(cand_items, select_seq=False)
            topk_item, topk_mask = self._process_seq(cand_items, select_seq=True)
            item_dict = {"item_hist": item}
            topk_item_dict = {"item_hist": topk_item}
            sim_rewards = 5 - self.SIM.get_loss(item_dict, item_mask, topk_item_dict, topk_mask).item()

            total_rewards = 2 * integrity_rewards + 2 * count_rewards + 10 * sim_rewards

            rewards["total_rewards"].append(total_rewards)
            rewards["integrity_rewards"].append(integrity_rewards)
            rewards["count_rewards"].append(count_rewards)
            rewards["sim_rewards"].append(sim_rewards)

        return rewards
