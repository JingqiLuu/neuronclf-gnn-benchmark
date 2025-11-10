#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN Models Bayesian Optimization

Optimizes GNN models (GCN, GraphSAGE, GAT, GraphTransformer) on three feature sets:
- Position features (pos_only)
- Connection profile features (connection_profile_only)
- ISI features (isi_only)
"""

import os
import sys
import logging
import numpy as np
import torch
import json
import optuna
from typing import Dict, Any


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.logging_config import get_global_logger, reset_global_log
from src.data.data_cache import initialize_global_cache
from src.models.gnn_models import GCN, GraphSAGE, GAT, GraphTransformer
from src.models.trainer import GNNTrainer
from src.optimization.bayesian_optimizer import BayesianOptimizer, OptimizationConfig, HyperparameterSpaces


def create_optimized_model_config(params: Dict[str, Any], model_type: str):
    from src.config.model_config import ModelConfig
    
    config = ModelConfig(
        epochs=200,
        learning_rate=params['learning_rate'],
        weight_decay=params['weight_decay'],
        hidden_dim=params['hidden_dim'],
        dropout=params['dropout'],
        num_layers=params['num_layers'],
        early_stopping_patience=params['early_stopping_patience'],
        scheduler_patience=params['scheduler_patience'],
        scheduler_factor=params['scheduler_factor'],
        use_kfold=True,
        kfold_splits=5,
        kfold_random_state=42
    )
    
    if model_type == 'GAT':
        config.heads = params['heads']
    elif model_type == 'GraphTransformer':
        config.heads = params['heads']
    
    return config


def gnn_objective(trial, model_type: str, data, processor, feature_name: str):
    try:
        params = HyperparameterSpaces.suggest_hyperparameters(trial, model_type, feature_name)
        
        config = create_optimized_model_config(params, model_type)
        
        if model_type == 'GCN':
            model = GCN(
                data.num_node_features, 
                config.hidden_dim, 
                len(processor.label_encoder.classes_),
                num_layers=config.num_layers,
                dropout=config.dropout,
                use_edge_attr=True,
                batch_norm=getattr(config, 'batch_norm', True),
                residual_connection=getattr(config, 'residual_connection', False),
                activation=getattr(config, 'activation', 'relu'),
                aggr=getattr(config, 'aggr', 'mean'),
                bias=getattr(config, 'bias', True),
                normalize=getattr(config, 'normalize', True)
            )
        elif model_type == 'GraphSAGE':
            model = GraphSAGE(
                data.num_node_features, 
                config.hidden_dim, 
                len(processor.label_encoder.classes_),
                num_layers=config.num_layers,
                dropout=config.dropout,
                use_edge_attr=True,
                batch_norm=getattr(config, 'batch_norm', True),
                residual_connection=getattr(config, 'residual_connection', False),
                activation=getattr(config, 'activation', 'relu'),
                aggr=getattr(config, 'aggr', 'mean'),
                bias=getattr(config, 'bias', True),
                normalize=getattr(config, 'normalize', True)
            )
        elif model_type == 'GAT':
            model = GAT(
                data.num_node_features, 
                config.hidden_dim, 
                len(processor.label_encoder.classes_),
                num_layers=config.num_layers,
                heads=getattr(config, 'heads', 1),
                dropout=config.dropout,
                use_edge_attr=True,
                batch_norm=getattr(config, 'batch_norm', True),
                residual_connection=getattr(config, 'residual_connection', False),
                activation=getattr(config, 'activation', 'relu'),
                aggr=getattr(config, 'aggr', 'mean'),
                bias=getattr(config, 'bias', True),
                normalize=getattr(config, 'normalize', True),
                concat=getattr(config, 'concat', True),
                negative_slope=getattr(config, 'negative_slope', 0.2),
                add_self_loops=getattr(config, 'add_self_loops', True),
                edge_dim=getattr(config, 'edge_dim', 1)
            )
        elif model_type == 'GraphTransformer':
            model = GraphTransformer(
                data.num_node_features, 
                config.hidden_dim, 
                len(processor.label_encoder.classes_),
                num_layers=config.num_layers,
                heads=getattr(config, 'heads', 1),
                dropout=config.dropout,
                use_edge_attr=True,
                batch_norm=getattr(config, 'batch_norm', True),
                residual_connection=getattr(config, 'residual_connection', False),
                activation=getattr(config, 'activation', 'relu'),
                aggr=getattr(config, 'aggr', 'mean'),
                bias=getattr(config, 'bias', True),
                normalize=getattr(config, 'normalize', True),
                concat=getattr(config, 'concat', True),
                beta=getattr(config, 'beta', 0.0),
                edge_dim=getattr(config, 'edge_dim', 1),
                root_weight=getattr(config, 'root_weight', True)
            )
        else:
            raise ValueError(f"Unknown GNN model type: {model_type}")
        
        trainer = GNNTrainer(model, data, config, save_checkpoints=False)
        fold_results, mean_val_acc, std_val_acc, mean_test_acc, std_test_acc = trainer.train_kfold()
        
        avg_f1_scores = [result['avg_f1'] for result in fold_results]
        mean_avg_f1 = np.mean(avg_f1_scores)
        std_avg_f1 = np.std(avg_f1_scores)
        
        trial.set_user_attr('val_accuracy', mean_val_acc)
        trial.set_user_attr('val_accuracy_std', std_val_acc)
        trial.set_user_attr('test_accuracy', mean_test_acc)
        trial.set_user_attr('test_accuracy_std', std_test_acc)
        trial.set_user_attr('test_f1_score', mean_avg_f1)
        trial.set_user_attr('test_f1_std', std_avg_f1)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return mean_val_acc
        
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.warning(f"Trial failed: {e}")
        return 0.0


def gnn_bayesian_optimization():
    """Run Bayesian optimization for GNN models on three feature sets"""
    reset_global_log()
    logger = get_global_logger("all_gnn_three_features_optimization")
    logger.info("Starting Bayesian optimization for all GNN models with three feature sets")
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    import random
    random.seed(42)
    import os
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    feature_configs = {
        'pos_only': 'Position features (3D coordinates)',
        'connection_profile_only': 'Connection features (202D)',
        'isi_only': 'neuronal activity features (90D)'
    }
    
    logger.info(f"Optimizing {len(feature_configs)} feature combinations")
    
    gnn_models = ['GCN', 'GraphSAGE', 'GAT', 'GraphTransformer']
    non_graph_models = []
    
    logger.info(f"Testing {len(gnn_models)} GNN models for each feature combination")

    optimization_config = OptimizationConfig(
        concise_logging=True,
        n_trials=50,
        timeout=10800,
        direction='maximize',
        sampler='TPE',
        pruner='MedianPruner',
        n_startup_trials=10,
        n_warmup_steps=5,
        interval_steps=3
    )
    
    all_optimization_results = {}
    
    data_cache = initialize_global_cache(merge_bilateral=False)
    processor = data_cache.get_processor()
    
    for i, (feature_name, description) in enumerate(feature_configs.items(), 1):
        logger.info(f"Feature combination {i}/{len(feature_configs)}: {feature_name} ({description})")
        
        data = data_cache.get_data(feature_name)
        
        feature_results = {}
        
        for model_type in gnn_models:
            optimizer = BayesianOptimizer(optimization_config)
            study_name = f"three_features_{feature_name}_{model_type}_optimization"
            optimizer.create_study(study_name)
            
            best_params, best_value = optimizer.optimize(
                gnn_objective, 
                model_type, 
                data, 
                processor, 
                feature_name
            )
            
            best_trial = optimizer.study.best_trial
            best_test_acc = best_trial.user_attrs.get('test_accuracy', 0.0)
            best_test_std = best_trial.user_attrs.get('test_accuracy_std', 0.0)
            best_val_acc = best_trial.user_attrs.get('val_accuracy', 0.0)
            best_val_std = best_trial.user_attrs.get('val_accuracy_std', 0.0)
            best_mean_f1 = best_trial.user_attrs.get('test_f1_score', 0.0)
            best_std_f1 = best_trial.user_attrs.get('test_f1_std', 0.0)
            
            feature_results[f"{model_type}_optimized"] = {
                'best_params': best_params,
                'best_test_accuracy': best_test_acc,
                'test_accuracy_std': best_test_std,
                'best_val_accuracy': best_val_acc,
                'val_accuracy_std': best_val_std,
                'test_f1_score': best_mean_f1,
                'test_f1_std': best_std_f1,
                'model_type': model_type,
                'optimization_type': 'bayesian'
            }
            
            logger.info(f"  {model_type}: Test {best_test_acc:.4f}±{best_test_std:.4f}, Val {best_val_acc:.4f}±{best_val_std:.4f}, F1 {best_mean_f1:.4f}±{best_std_f1:.4f} (Trial {best_trial.number})")
            
            results_dir = 'results'
            os.makedirs(results_dir, exist_ok=True)
            plot_path = f"{results_dir}/three_features_bayesian_optimization_{feature_name}_{model_type}.png"
            optimizer.plot_optimization_history(plot_path)
            
            study_path = f"{results_dir}/three_features_bayesian_study_{feature_name}_{model_type}.pkl"
            optimizer.save_study(study_path)
        
        
        all_optimization_results[feature_name] = {
            'description': description,
            'optimization_results': feature_results,
            'data_info': {
                'num_nodes': data.num_nodes,
                'num_edges': data.num_edges,
                'num_features': data.num_node_features,
                'num_classes': len(processor.label_encoder.classes_)
            }
        }
        
        del data
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    logger.info(f"\n{'Feature Combination':<25} {'Best Model':<20} {'Test Accuracy':<18} {'Val Accuracy':<15}")
    logger.info("-" * 80)
    
    for feature_name, experiment_data in all_optimization_results.items():
        optimization_results = experiment_data['optimization_results']
        
        best_model = max(optimization_results.items(), key=lambda x: x[1]['best_test_accuracy'])
        best_model_name = best_model[0]
        best_test_acc = best_model[1]['best_test_accuracy']
        test_std = best_model[1]['test_accuracy_std']
        best_val_acc = best_model[1]['best_val_accuracy']
        
        logger.info(f"{feature_name:<25} {best_model_name:<20} Test:{best_test_acc:.4f}±{test_std:.4f} Val:{best_val_acc:.4f}")
    
    save_gnn_bayesian_results(all_optimization_results)
    logger.info(f"\nResults saved to results/ folder")
    logger.info("Bayesian optimization experiment completed")
    return all_optimization_results


def save_gnn_bayesian_results(all_optimization_results):
    import json
    import os
    logger = logging.getLogger(__name__)
    
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    summary_results = {}
    for feature_name, experiment_data in all_optimization_results.items():
        optimization_results = experiment_data['optimization_results']
        
        best_model = max(optimization_results.items(), key=lambda x: x[1]['best_test_accuracy'])
        
        summary_results[feature_name] = {
            'description': experiment_data['description'],
            'data_info': experiment_data['data_info'],
            'best_model': best_model[0],
            'best_val_accuracy': best_model[1]['best_val_accuracy'],
            'val_accuracy_std': best_model[1]['val_accuracy_std'],
            'best_test_accuracy': best_model[1]['best_test_accuracy'],
            'test_accuracy_std': best_model[1]['test_accuracy_std'],
            'test_f1_score': best_model[1]['test_f1_score'],
            'test_f1_std': best_model[1]['test_f1_std'],
            'best_params': best_model[1]['best_params'],
            'all_model_results': {
                name: {
                    'best_val_accuracy': result['best_val_accuracy'],
                    'val_accuracy_std': result['val_accuracy_std'],
                    'best_test_accuracy': result['best_test_accuracy'],
                    'test_accuracy_std': result['test_accuracy_std'],
                    'test_f1_score': result['test_f1_score'],
                    'test_f1_std': result['test_f1_std'],
                    'best_params': result['best_params'],
                    'model_type': result['model_type'],
                    'optimization_type': result['optimization_type']
                }
                for name, result in optimization_results.items()
            }
        }
    
    json_file_path = f'{results_dir}/three_features_bayesian_optimization_results.json'
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {json_file_path}")

if __name__ == "__main__":
    gnn_bayesian_optimization()
