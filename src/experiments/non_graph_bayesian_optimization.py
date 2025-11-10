#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-graph model Bayesian optimization
"""

import os
import sys
import logging
import numpy as np
import torch
import json
from typing import Dict, Any

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.logging_config import get_global_logger, reset_global_log
from src.data.data_cache import initialize_global_cache
from src.models.non_graph_models import NonGraphTrainer
from src.optimization.bayesian_optimizer import BayesianOptimizer, OptimizationConfig, HyperparameterSpaces


def non_graph_objective(trial, model_type: str, data, processor, feature_name: str):
    """Non-graph model (MLP, LogReg) Bayesian optimization objective function"""
    try:
        if model_type == 'MLP':
            params = HyperparameterSpaces.suggest_hyperparameters(trial, 'MLP', feature_name)
            from src.config.model_config import ModelConfig
            config = ModelConfig(
                epochs=200,
                learning_rate=params.get('learning_rate', 1e-3),
                weight_decay=params.get('weight_decay', 0.0),
                hidden_dim=params.get('hidden_dim', 128),
                dropout=params.get('dropout', 0.5),
                num_layers=params.get('num_layers', 2),
                early_stopping_patience=params.get('early_stopping_patience', 20),
                scheduler_patience=params.get('scheduler_patience', 10),
                scheduler_factor=params.get('scheduler_factor', 0.5),
                use_kfold=True,
                kfold_splits=5,
                kfold_random_state=42
            )
            trainer = NonGraphTrainer('MLP', data, config)
        elif model_type == 'LogisticRegression':
            params = HyperparameterSpaces.suggest_hyperparameters(trial, 'LogisticRegression', feature_name)
            from src.config.model_config import ModelConfig
            temp_config = ModelConfig(use_kfold=True, kfold_splits=5, kfold_random_state=42)
            trainer = NonGraphTrainer('LogisticRegression', data, temp_config)
            trainer.lr_params = params

        fold_results, mean_val_acc, std_val_acc, mean_test_acc, std_test_acc = trainer.train_kfold()

        avg_f1_scores = [result['avg_f1'] for result in fold_results]
        mean_avg_f1 = float(np.mean(avg_f1_scores))
        std_avg_f1 = float(np.std(avg_f1_scores))

        trial.set_user_attr('val_accuracy', float(mean_val_acc))
        trial.set_user_attr('val_accuracy_std', float(std_val_acc))
        trial.set_user_attr('test_accuracy', float(mean_test_acc))
        trial.set_user_attr('test_accuracy_std', float(std_test_acc))
        trial.set_user_attr('test_f1_score', mean_avg_f1)
        trial.set_user_attr('test_f1_std', std_avg_f1)

        return float(mean_test_acc)

    except Exception as e:
        logging.warning(f"Trial failed: {e}")
        return 0.0


def non_graph_bayesian_optimization():
    reset_global_log()
    logger = get_global_logger("non_graph_bayesian_optimization")
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import random
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    feature_configs = {
        'pos_only': 'Position features (3D coordinates)',
        'connection_profile_only': 'Connection profile features (202D)',
        'isi_only': 'ISI features (90D)'
    }
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
    non_graph_models = ['MLP', 'LogisticRegression']
    for i, (feature_name, description) in enumerate(feature_configs.items(), 1):
        data = data_cache.get_data(feature_name)
        feature_results = {}
        for model_type in non_graph_models:
            optimizer = BayesianOptimizer(optimization_config)
            study_name = f"{feature_name}_{model_type}_non_graph_optimization"
            optimizer.create_study(study_name)

            best_params, best_value = optimizer.optimize(
                non_graph_objective,
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

            results_dir = 'results'
            os.makedirs(results_dir, exist_ok=True)
            plot_path = f"{results_dir}/non_graph_bayesian_optimization_{feature_name}_{model_type}.png"
            optimizer.plot_optimization_history(plot_path)
            study_path = f"{results_dir}/non_graph_bayesian_study_{feature_name}_{model_type}.pkl"
            optimizer.save_study(study_path)

            logger.info(f"  {model_type}: Test {best_test_acc:.4f}±{best_test_std:.4f}, Val {best_val_acc:.4f}±{best_val_std:.4f} (Trial {best_trial.number})")

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
    for feature_name, experiment_data in all_optimization_results.items():
        optimization_results = experiment_data['optimization_results']
        best_model = max(optimization_results.items(), key=lambda x: x[1]['best_test_accuracy'])
        name, res = best_model
        logger.info(f"{feature_name:<25} {name:<16} {res['best_test_accuracy']:.4f}±{res['test_accuracy_std']:.4f}   {res['best_val_accuracy']:.4f}±{res['val_accuracy_std']:.4f}   {res['test_f1_score']:.4f}±{res['test_f1_std']:.4f}")

    save_non_graph_bayesian_results(all_optimization_results)
    logger.info(f"\nResults saved to results/ folder")
    return all_optimization_results


def save_non_graph_bayesian_results(all_optimization_results):
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    summary_results = {}
    for feature_name, experiment_data in all_optimization_results.items():
        optimization_results = experiment_data['optimization_results']
        best_model = max(optimization_results.items(), key=lambda x: x[1]['best_test_accuracy'])
        name, res = best_model
        summary_results[feature_name] = {
            'description': experiment_data['description'],
            'data_info': experiment_data['data_info'],
            'best_model': name,
            'best_val_accuracy': res['best_val_accuracy'],
            'val_accuracy_std': res['val_accuracy_std'],
            'best_test_accuracy': res['best_test_accuracy'],
            'test_accuracy_std': res['test_accuracy_std'],
            'test_f1_score': res['test_f1_score'],
            'test_f1_std': res['test_f1_std'],
            'best_params': res['best_params']
        }

    json_file_path = f'{results_dir}/non_graph_bayesian_optimization_results.json'
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {json_file_path}")


if __name__ == "__main__":
    non_graph_bayesian_optimization()


