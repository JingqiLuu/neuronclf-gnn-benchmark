#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-graph models implementation.

Includes MLP and Logistic Regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class MLP(nn.Module):
    """Multi-layer Perceptron"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, dropout: float = 0.5):
        super(MLP, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class NonGraphTrainer:
    """Non-graph model trainer"""
    
    def __init__(self, model_type: str, data, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_type = model_type
        self.data = data
        self.config = config
        self.device = device
        
        self.node_features = data.x.cpu().numpy()
        self.labels = data.y.cpu().numpy()
        self.num_classes = len(np.unique(self.labels))
        
        self.scaler = StandardScaler()
        self.node_features = self.scaler.fit_transform(self.node_features)
        
        if config.use_kfold:
            self.create_kfold_splits()
        else:
            self.create_splits()
    
    def create_splits(self):
        """Create data splits by class"""
        n_nodes = len(self.labels)
        
        np.random.seed(42)
        
        self.train_mask = np.zeros(n_nodes, dtype=bool)
        self.val_mask = np.zeros(n_nodes, dtype=bool)
        self.test_mask = np.zeros(n_nodes, dtype=bool)
        
        unique_classes = np.unique(self.labels)
        
        for class_label in unique_classes:
            class_indices = np.where(self.labels == class_label)[0]
            n_class_nodes = len(class_indices)
            
            if n_class_nodes == 0:
                continue
            
            np.random.shuffle(class_indices)
            
            train_size_class = int(self.config.train_ratio * n_class_nodes)
            val_size_class = int(self.config.val_ratio * n_class_nodes)
            
            train_indices_class = class_indices[:train_size_class]
            val_indices_class = class_indices[train_size_class:train_size_class + val_size_class]
            test_indices_class = class_indices[train_size_class + val_size_class:]
            
            self.train_mask[train_indices_class] = True
            self.val_mask[val_indices_class] = True
            self.test_mask[test_indices_class] = True
    
    def create_kfold_splits(self):
        """Create K-Fold cross-validation splits"""
        from sklearn.model_selection import train_test_split
        
        n_nodes = len(self.labels)
        
        self.kfold_splits = []
        
        for fold_idx in range(self.config.kfold_splits):
            random_seed = self.config.kfold_random_state + fold_idx
            
            train_indices, test_indices, train_labels, test_labels = train_test_split(
                np.arange(n_nodes), self.labels, 
                test_size=0.2, 
                random_state=random_seed,
                stratify=self.labels
            )
            
            inner_train_idx, inner_val_idx = train_test_split(
                np.arange(len(train_indices)),
                test_size=0.2,
                random_state=random_seed,
                stratify=train_labels
            )
            
            train_mask = np.zeros(n_nodes, dtype=bool)
            val_mask = np.zeros(n_nodes, dtype=bool)
            test_mask = np.zeros(n_nodes, dtype=bool)
            
            train_mask[train_indices[inner_train_idx]] = True
            val_mask[train_indices[inner_val_idx]] = True
            test_mask[test_indices] = True
            
            self.kfold_splits.append({
                'train_mask': train_mask,
                'val_mask': val_mask,
                'test_mask': test_mask,
                'fold_idx': fold_idx,
                'random_seed': random_seed
            })
        
        self.train_mask = self.kfold_splits[0]['train_mask']
        self.val_mask = self.kfold_splits[0]['val_mask']
        self.test_mask = self.kfold_splits[0]['test_mask']
    
    def train_mlp(self, train_mask, val_mask, test_mask):
        """Train MLP model"""
        X_train = self.node_features[train_mask]
        y_train = self.labels[train_mask]
        X_val = self.node_features[val_mask]
        y_val = self.labels[val_mask]
        X_test = self.node_features[test_mask]
        y_test = self.labels[test_mask]
        
        model = MLP(
            input_dim=self.node_features.shape[1],
            hidden_dim=self.config.hidden_dim,
            output_dim=self.num_classes,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=self.config.scheduler_factor, 
            patience=self.config.scheduler_patience
        )
        
        best_val_acc = 0
        train_losses = []
        val_accuracies = []
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_pred = val_outputs.argmax(dim=1)
                val_acc = (val_pred == y_val_tensor).float().mean().item()
                val_accuracies.append(val_acc)
            
            scheduler.step(loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if self.config.early_stopping and patience_counter >= self.config.early_stopping_patience:
                break
        
        model.load_state_dict(best_model_state)
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_pred = test_outputs.argmax(dim=1)
            test_acc = (test_pred == y_test_tensor).float().mean().item()
        
        return {
            'test_accuracy': test_acc,
            'val_accuracy': best_val_acc,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'predictions': test_pred.cpu().numpy(),
            'true_labels': y_test
        }
    
    def train_logistic_regression(self, train_mask, val_mask, test_mask, lr_params=None):
        """Train Logistic Regression model"""
        X_train = self.node_features[train_mask]
        y_train = self.labels[train_mask]
        X_val = self.node_features[val_mask]
        y_val = self.labels[val_mask]
        X_test = self.node_features[test_mask]
        y_test = self.labels[test_mask]
        
        if lr_params is None:
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0,
                solver='lbfgs'
            )
        else:
            model = LogisticRegression(
                random_state=42,
                C=lr_params.get('C', 1.0),
                max_iter=lr_params.get('max_iter', 1000),
                solver=lr_params.get('solver', 'lbfgs'),
                penalty=lr_params.get('penalty', 'l2')
            )
        
        model.fit(X_train, y_train)
        
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        
        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        
        return {
            'test_accuracy': test_acc,
            'val_accuracy': val_acc,
            'predictions': test_pred,
            'true_labels': y_test
        }
    
    def train_kfold(self):
        """K-Fold cross-validation training"""
        fold_results = []
        
        for fold_idx, fold_data in enumerate(self.kfold_splits):
            fold_seed = fold_data.get('random_seed', self.config.kfold_random_state + fold_idx)
            np.random.seed(fold_seed)
            import torch
            torch.manual_seed(fold_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(fold_seed)
                torch.cuda.manual_seed_all(fold_seed)
            import random
            random.seed(fold_seed)
            
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            train_mask = fold_data['train_mask']
            val_mask = fold_data['val_mask']
            test_mask = fold_data['test_mask']
            
            if self.model_type == 'MLP':
                result = self.train_mlp(train_mask, val_mask, test_mask)
            elif self.model_type == 'LogisticRegression':
                result = self.train_logistic_regression(train_mask, val_mask, test_mask, getattr(self, 'lr_params', None))
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            avg_f1 = f1_score(result['true_labels'], result['predictions'], average='macro')
            weighted_avg_f1 = f1_score(result['true_labels'], result['predictions'], average='weighted')
            cm = confusion_matrix(result['true_labels'], result['predictions'])
            
            fold_results.append({
                'fold_idx': fold_idx,
                'test_accuracy': result['test_accuracy'],
                'val_accuracy': result['val_accuracy'],
                'avg_f1': avg_f1,
                'weighted_avg_f1': weighted_avg_f1,
                'confusion_matrix': cm,
                'predictions': result['predictions'],
                'true_labels': result['true_labels'],
                'train_losses': result.get('train_losses', []),
                'val_accuracies': result.get('val_accuracies', [])
            })
        
        val_accuracies = [result['val_accuracy'] for result in fold_results]
        mean_val_accuracy = np.mean(val_accuracies)
        std_val_accuracy = np.std(val_accuracies)
        
        test_accuracies = [result['test_accuracy'] for result in fold_results]
        mean_test_accuracy = np.mean(test_accuracies)
        std_test_accuracy = np.std(test_accuracies)
        
        avg_f1_scores = [result['avg_f1'] for result in fold_results]
        mean_avg_f1 = np.mean(avg_f1_scores)
        std_avg_f1 = np.std(avg_f1_scores)
        
        weighted_avg_f1_scores = [result['weighted_avg_f1'] for result in fold_results]
        mean_weighted_avg_f1 = np.mean(weighted_avg_f1_scores)
        std_weighted_avg_f1 = np.std(weighted_avg_f1_scores)
        
        return fold_results, mean_val_accuracy, std_val_accuracy, mean_test_accuracy, std_test_accuracy