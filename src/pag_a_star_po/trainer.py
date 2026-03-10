import os
import json
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from .helpers import score_solution


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_length: int = 2048
    temperature: float = 1.0
    kl_coef: float = 0.1
    batch_size: int = 1
    learning_rate: float = 1e-6
    num_epochs: int = 5
    gradient_accumulation_steps: int = 32
    max_turns: int = 2
    reward_shaping_alpha: float = 1.0
    use_role_adv_norm: bool = True
    max_new_tokens: int = 512
    offline_data_path: str = "training_data_with_vstar.json"
    checkpoint_dir: str = "./checkpoints"
    clear_cache_every: int = 4


class PAGAPOTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        print(f"Loading model: {config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Loading reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto")
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.policy_adv_stats = {'mean': 0.0, 'std': 1.0}
        self.verifier_adv_stats = {'mean': 0.0, 'std': 1.0}

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, path)
        print(f"Checkpoint saved to {path}")

    # NOTE: The full trainer implementation is large; this module provides the trainer skeleton
    # and key helper methods were ported to this package. For full training loop, call train().

    def train(self, offline_data_path: str = None):
        if offline_data_path is None:
            offline_data_path = self.config.offline_data_path
        print("Starting training loop (skeleton). Load data and call train_epoch as implemented in notebook.")
        # Loading and running a full training loop requires large resources; keep this as a starter.
        return
