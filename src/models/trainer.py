#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

from ..config.model_config import ModelConfig


class GNNTrainer:

    
    def __init__(self, model, data, config: ModelConfig, device='cuda' if torch.cuda.is_available() else 'cpu', save_checkpoints=True):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.config = config
        self.save_checkpoints = save_checkpoints
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=config.scheduler_factor, 
            patience=config.scheduler_patience
        )
        
        # 分割训练/验证/测试集
        if config.use_kfold:
            self.create_kfold_splits()
        else:
            self.create_splits()
    
    def create_splits(self):
        n_nodes = self.data.num_nodes
        labels = self.data.y.cpu().numpy()
        
        torch.manual_seed(42)
        np.random.seed(42)
        import random
        random.seed(42)
        
        self.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        unique_classes = np.unique(labels)
        
        for class_label in unique_classes:
            class_indices = np.where(labels == class_label)[0]
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
        n_nodes = self.data.num_nodes
        labels = self.data.y.cpu().numpy()
        
        self.kfold_splits = []
        
        for fold_idx in range(self.config.kfold_splits):
            random_seed = self.config.kfold_random_state + fold_idx
            
            train_indices, test_indices, train_labels, test_labels = train_test_split(
                np.arange(n_nodes), labels, 
                test_size=0.2, 
                random_state=random_seed,
                stratify=labels
            )
            
            inner_train_idx, inner_val_idx = train_test_split(
                np.arange(len(train_indices)),
                test_size=0.2,
                random_state=random_seed,
                stratify=train_labels
            )
            
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            
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
    
    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        
        if hasattr(self.data, 'edge_attr') and self.data.edge_attr is not None:
            out = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
        else:
            out = self.model(self.data.x, self.data.edge_index)
        
        loss = F.cross_entropy(out[self.train_mask], self.data.y[self.train_mask])
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, mask):
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.data, 'edge_attr') and self.data.edge_attr is not None:
                out = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
            else:
                out = self.model(self.data.x, self.data.edge_index)
            
            pred = out[mask].argmax(dim=1)
            correct = (pred == self.data.y[mask]).sum().item()
            accuracy = correct / mask.sum().item()
        return accuracy, pred.cpu().numpy(), self.data.y[mask].cpu().numpy()
    
    def train(self, epochs=None):
        if epochs is None:
            epochs = self.config.epochs
            
        best_val_acc = 0
        train_losses = []
        val_accuracies = []
        patience_counter = 0
        
        for epoch in range(epochs):
            loss = self.train_epoch()
            train_losses.append(loss)
            
            val_acc, _, _ = self.evaluate(self.val_mask)
            val_accuracies.append(val_acc)
            
            self.scheduler.step(loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if self.save_checkpoints:
                    torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if self.config.early_stopping and patience_counter >= self.config.early_stopping_patience:
                break
        
        if self.save_checkpoints and os.path.exists('best_model.pth'):
            self.model.load_state_dict(torch.load('best_model.pth'))
        
        test_acc, test_pred, test_true = self.evaluate(self.test_mask)
        
        return train_losses, val_accuracies, test_acc, test_pred, test_true
    
    def train_kfold(self, epochs=None):
        if epochs is None:
            epochs = self.config.epochs
        
        fold_results = []
        
        for fold_idx, fold_data in enumerate(self.kfold_splits):
            self.train_mask = fold_data['train_mask']
            self.val_mask = fold_data['val_mask']
            self.test_mask = fold_data['test_mask']
            
            fold_seed = fold_data.get('random_seed', self.config.kfold_random_state + fold_idx)
            torch.manual_seed(fold_seed)
            np.random.seed(fold_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(fold_seed)
                torch.cuda.manual_seed_all(fold_seed)
            
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            self.model.apply(self._reset_parameters)
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.config.learning_rate, 
                weight_decay=self.config.weight_decay
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=self.config.scheduler_factor, 
                patience=self.config.scheduler_patience
            )
            
            best_val_acc = 0
            train_losses = []
            val_accuracies = []
            patience_counter = 0
            
            for epoch in range(epochs):
                loss = self.train_epoch()
                train_losses.append(loss)
                
                val_acc, _, _ = self.evaluate(self.val_mask)
                val_accuracies.append(val_acc)
                
                self.scheduler.step(loss)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    if self.save_checkpoints:
                        torch.save(self.model.state_dict(), f'best_model_fold_{fold_idx}.pth')
                else:
                    patience_counter += 1
                
                if self.config.early_stopping and patience_counter >= self.config.early_stopping_patience:
                    break
            
            if self.save_checkpoints and os.path.exists(f'best_model_fold_{fold_idx}.pth'):
                self.model.load_state_dict(torch.load(f'best_model_fold_{fold_idx}.pth'))
            
            test_acc, test_pred, test_true = self.evaluate(self.test_mask)
            
            cm = confusion_matrix(test_true, test_pred)
            
            avg_f1 = f1_score(test_true, test_pred, average='macro')
            weighted_avg_f1 = f1_score(test_true, test_pred, average='weighted')
            
            fold_results.append({
                'fold_idx': fold_idx,
                'test_accuracy': test_acc,
                'avg_f1': avg_f1,
                'weighted_avg_f1': weighted_avg_f1,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'confusion_matrix': cm,
                'predictions': test_pred,
                'true_labels': test_true,
                'best_val_acc': best_val_acc
            })
        
        val_accuracies = [result['best_val_acc'] for result in fold_results]
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
    
    def _reset_parameters(self, module):
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()