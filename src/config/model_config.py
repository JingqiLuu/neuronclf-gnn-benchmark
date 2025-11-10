#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass
class ModelConfig:
    epochs: int = 200
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    
    hidden_dim: int = 64
    dropout: float = 0.5
    
    num_layers: int = 2
    heads: int = 8
    mlp_layers: int = 2
    
    early_stopping: bool = True
    early_stopping_patience: int = 30
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    use_kfold: bool = False
    kfold_splits: int = 5
    kfold_random_state: int = 42
    
    embedding_dim: int = 64
    num_trials_per_neuron: int = 5
    trial_noise_std: float = 0.01
    reconstruction_type: str = 'linear'
    time_dim: int = 100
    recon_weight: float = 0.5
    temporal_noise_std: float = 0.01
    use_population: bool = False
    use_neighbor: bool = False
    nhead: int = 2