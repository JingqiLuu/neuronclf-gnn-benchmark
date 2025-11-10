
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import optuna
import numpy as np
import torch
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    n_trials: int = 50
    timeout: Optional[int] = None
    direction: str = 'maximize'
    sampler: str = 'TPE'
    pruner: str = 'MedianPruner'
    n_startup_trials: int = 10
    n_warmup_steps: int = 5
    interval_steps: int = 3
    concise_logging: bool = False
    trial_logging: bool = True


class BayesianOptimizer:
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.study = None
        self.best_params = None
        self.best_value = None
        
    def create_study(self, study_name: str = "bayesian_optimization"):
        if self.config.sampler == 'TPE':
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_ei_candidates=24,
                seed=42
            )
        elif self.config.sampler == 'Random':
            sampler = optuna.samplers.RandomSampler(seed=42)
        elif self.config.sampler == 'CMA-ES':
            sampler = optuna.samplers.CmaEsSampler(seed=42)
        else:
            sampler = optuna.samplers.TPESampler(seed=42)
        
        if self.config.pruner == 'MedianPruner':
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.config.n_startup_trials,
                n_warmup_steps=self.config.n_warmup_steps,
                interval_steps=self.config.interval_steps
            )
        elif self.config.pruner == 'PercentilePruner':
            pruner = optuna.pruners.PercentilePruner(25.0)
        else:
            pruner = optuna.pruners.MedianPruner()
        
        self.study = optuna.create_study(
            study_name=study_name,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner
        )
        
        logger.info(f"Bayesian optimization study created: {study_name}")
        
    def optimize(self, objective_func, *args, **kwargs):
        if self.study is None:
            raise ValueError("Please call create_study() first")
        
        logger.info("Starting Bayesian optimization...")
        
        import optuna
        verbosity = optuna.logging.INFO if self.config.trial_logging else optuna.logging.WARNING
        optuna.logging.set_verbosity(verbosity)
        
        if self.config.concise_logging:
            try:
                logging.getLogger().setLevel(logging.WARNING)
                logging.getLogger('src').setLevel(logging.WARNING)
                logging.getLogger('torch').setLevel(logging.WARNING)
            except Exception:
                pass
        
        def _trial_callback(study, trial):
            if not self.config.trial_logging:
                return
            import optuna
            if trial.state == optuna.trial.TrialState.COMPLETE:
                test_acc = trial.user_attrs.get('test_accuracy', trial.value)
                f1 = trial.user_attrs.get('test_f1_score', 0.0)
                best_value = study.best_value if study.best_value is not None else trial.value
                trial_value = trial.value if trial.value is not None else float('nan')
                logger.info(
                    f"Trial {trial.number}: acc={test_acc:.4f} f1={f1:.4f} value={trial_value:.4f} best={best_value:.4f} params={trial.params}"
                )
            elif trial.state == optuna.trial.TrialState.PRUNED:
                step_info = trial.last_step if trial.last_step is not None else 'unknown'
                logger.info(f"Trial {trial.number}: pruned at step {step_info}")
            elif trial.state == optuna.trial.TrialState.FAIL:
                fail_reason = trial.system_attrs.get('fail_reason') if hasattr(trial, 'system_attrs') else None
                logger.info(f"Trial {trial.number}: failed with error {fail_reason or 'unknown'}")
        
        self.study.optimize(
            lambda trial: objective_func(trial, *args, **kwargs),
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            callbacks=[_trial_callback]
        )
        
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        logger.info("Bayesian optimization finished")
        logger.info(f"Best objective: {self.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        return self.best_params, self.best_value
    
    def get_trial_results(self):
        if self.study is None:
            return None
        
        results = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                results.append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                })
        
        return results
    
    def plot_optimization_history(self, save_path: str = None):
        if self.study is None:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            try:
                trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                if trials:
                    values = [t.value for t in trials]
                    best_values = []
                    best_so_far = float('-inf')
                    for value in values:
                        if value > best_so_far:
                            best_so_far = value
                        best_values.append(best_so_far)
                    
                    ax1.plot(range(1, len(values) + 1), values, 'o-', alpha=0.7, label='Trial Values')
                    ax1.plot(range(1, len(best_values) + 1), best_values, 'r-', linewidth=2, label='Best So Far')
                    ax1.set_xlabel('Trial Number')
                    ax1.set_ylabel('Objective Value')
                    ax1.set_title('Optimization History')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                else:
                    ax1.text(0.5, 0.5, 'Optimization History\n(No data)', 
                           ha='center', va='center', transform=ax1.transAxes)
            except Exception as e:
                logger.warning(f"Failed to plot optimization history: {e}")
                ax1.text(0.5, 0.5, 'Optimization History\n(Plot failed)', 
                       ha='center', va='center', transform=ax1.transAxes)
            
            try:
                param_importances = optuna.importance.get_param_importances(self.study)
                if param_importances:
                    params = list(param_importances.keys())
                    importances = list(param_importances.values())
                    
                    ax2.barh(params, importances)
                    ax2.set_xlabel('Importance')
                    ax2.set_title('Parameter Importances')
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'Parameter Importances\n(No data)', 
                           ha='center', va='center', transform=ax2.transAxes)
            except Exception as e:
                logger.warning(f"Failed to plot parameter importance: {e}")
                ax2.text(0.5, 0.5, 'Parameter Importances\n(Plot failed)', 
                       ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Optimization history saved to: {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not installed, skip plotting optimization history")
    
    def save_study(self, filepath: str):
        if self.study is None:
            return
        
        import joblib
        joblib.dump(self.study, filepath)
        logger.info(f"Study saved to: {filepath}")
    
    def load_study(self, filepath: str):
        import joblib
        self.study = joblib.load(filepath)
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        logger.info(f"Study loaded from: {filepath}")


class HyperparameterSpaces:

    @staticmethod
    def get_gnn_spaces():
        return {
            'learning_rate': (1e-5, 1e-2, 'log'),
            'weight_decay': (1e-5, 1e-2, 'log'),
            'hidden_dim': (32, 256, 'int'),
            'dropout': (0.0, 0.7, 'float'),
            'num_layers': (2, 7, 'int'),
            'early_stopping_patience': (20, 100, 'int'),
            'scheduler_patience': (5, 20, 'int'),
            'scheduler_factor': (0.1, 0.8, 'float'),
            'batch_norm': [True, False],
            'residual_connection': [True, False],
            'activation': ['relu', 'elu', 'gelu'],
            'aggr': ['mean', 'max'],
            'bias': [True, False],
            'normalize': [True, False]
        }

    @staticmethod
    def get_gat_spaces():   
        base_spaces = HyperparameterSpaces.get_gnn_spaces()
        gat_spaces = {
            'heads': (1, 8, 'int'),
            'concat': [True, False],
            'negative_slope': (0.1, 0.3, 'float'),
            'add_self_loops': [True, False],
            'edge_dim': (1, 4, 'int'),
        }
        return {**base_spaces, **gat_spaces}
    
    @staticmethod
    def get_graph_transformer_spaces():
        base_spaces = HyperparameterSpaces.get_gnn_spaces()
        transformer_spaces = {
            'heads': (1, 8, 'int'),
            'concat': [True, False],
            'beta': (0.0, 0.5, 'float'),
            'edge_dim': (1, 4, 'int'),
            'root_weight': [True, False],
            'bias': [True, False],
        }
        return {**base_spaces, **transformer_spaces}
    
    @staticmethod
    def get_mlp_spaces():
        return {
            'learning_rate': (1e-4, 1e-1, 'log'),
            'weight_decay': (1e-5, 1e-2, 'log'),
            'hidden_dim': (32, 512, 'int'),
            'dropout': (0.1, 0.8, 'float'),
            'num_layers': (2, 6, 'int'),
            'early_stopping_patience': (20, 100, 'int'),
            'scheduler_patience': (5, 30, 'int'),
            'scheduler_factor': (0.1, 0.8, 'float')
        }
    
    @staticmethod
    def get_logistic_regression_spaces():
        return {
            'C': (1e-3, 1e3, 'log'),
            'max_iter': (100, 2000, 'int'),
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'penalty': ['l1', 'l2', 'elasticnet', 'none']
        }
    
    @staticmethod
    def get_lolcat_spaces():
        return {
            'learning_rate': (1e-5, 1e-2, 'log'),
            'weight_decay': (1e-5, 1e-2, 'log'),
            'hidden_dim': (32, 256, 'int'),
            'dropout': (0.0, 0.7, 'float'),
            'num_layers': (2, 6, 'int'),
            'early_stopping_patience': (20, 100, 'int'),
            'scheduler_patience': (5, 20, 'int'),
            'scheduler_factor': (0.1, 0.8, 'float'),
            'embedding_dim': (32, 128, 'int'),
            'num_trials_per_neuron': (3, 10, 'int'),
            'trial_noise_std': (0.001, 0.1, 'log')
        }
    
    @staticmethod
    def get_neuprint_spaces():
        return {
            'learning_rate': (1e-5, 1e-2, 'log'),
            'weight_decay': (1e-5, 1e-2, 'log'),
            'hidden_dim': (32, 256, 'int'),
            'dropout': (0.0, 0.7, 'float'),
            'num_layers': (2, 6, 'int'),
            'nhead': (1, 8, 'int'),
            'early_stopping_patience': (20, 100, 'int'),
            'scheduler_patience': (5, 20, 'int'),
            'scheduler_factor': (0.1, 0.8, 'float'),
            'embedding_dim': (32, 128, 'int'),
            'time_dim': (50, 200, 'int'),
            'use_population': [True, False],
            'use_neighbor': [True, False],
            'temporal_noise_std': (0.001, 0.1, 'log')
        }
    
    @staticmethod
    def suggest_hyperparameters(trial, model_type: str, feature_name: str = None):
        if model_type == 'GCN':
            spaces = HyperparameterSpaces.get_gnn_spaces()
        elif model_type == 'GraphSAGE':
            spaces = HyperparameterSpaces.get_gnn_spaces()
        elif model_type == 'GAT':
            spaces = HyperparameterSpaces.get_gat_spaces()
        elif model_type == 'GraphTransformer':
            spaces = HyperparameterSpaces.get_graph_transformer_spaces()
        elif model_type == 'MLP':
            spaces = HyperparameterSpaces.get_mlp_spaces()
        elif model_type == 'LogisticRegression':
            spaces = HyperparameterSpaces.get_logistic_regression_spaces()
        elif model_type == 'LOLCAT':
            spaces = HyperparameterSpaces.get_lolcat_spaces()
        elif model_type == 'NeuPRINT':
            spaces = HyperparameterSpaces.get_neuprint_spaces()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        params = {}
        for param_name, param_config in spaces.items():
            if isinstance(param_config, tuple):
                if len(param_config) == 3:
                    low, high, param_type = param_config
                    if param_type == 'log':
                        params[param_name] = trial.suggest_float(param_name, low, high, log=True)
                    elif param_type == 'int':
                        params[param_name] = trial.suggest_int(param_name, low, high)
                    elif param_type == 'float':
                        params[param_name] = trial.suggest_float(param_name, low, high)
                else:
                    low, high = param_config
                    params[param_name] = trial.suggest_float(param_name, low, high)
            elif isinstance(param_config, list):
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            else:
                raise ValueError(f"Invalid parameter configuration: {param_config}")
        
        if feature_name and model_type in ['GCN', 'GraphSAGE', 'GAT', 'GraphTransformer', 'MLP']:
            params = HyperparameterSpaces._adjust_params_for_feature(params, feature_name)
        
        return params
    
    @staticmethod
    def suggest_lolcat_hyperparameters(trial, feature_name: str = None):
        spaces = HyperparameterSpaces.get_lolcat_spaces()
        
        params = {}
        for param_name, param_config in spaces.items():
            if isinstance(param_config, tuple):
                if len(param_config) == 3:
                    low, high, param_type = param_config
                    if param_type == 'log':
                        params[param_name] = trial.suggest_float(param_name, low, high, log=True)
                    elif param_type == 'int':
                        params[param_name] = trial.suggest_int(param_name, low, high)
                    elif param_type == 'float':
                        params[param_name] = trial.suggest_float(param_name, low, high)
                else:
                    low, high = param_config
                    params[param_name] = trial.suggest_float(param_name, low, high)
            elif isinstance(param_config, list):
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            else:
                raise ValueError(f"Invalid parameter configuration: {param_config}")
        
        if feature_name:
            params = HyperparameterSpaces._adjust_lolcat_params_for_feature(params, feature_name)
        
        return params
    
    @staticmethod
    def _adjust_lolcat_params_for_feature(params: Dict, feature_name: str) -> Dict:
        if feature_name == 'connection_profile_only':
            params['dropout'] = max(params['dropout'], 0.3)
            params['weight_decay'] = max(params['weight_decay'], 1e-4)
            params['embedding_dim'] = max(params['embedding_dim'], 64)
        elif feature_name == 'pos_only':
            params['hidden_dim'] = min(params['hidden_dim'], 128)
            params['num_layers'] = min(params['num_layers'], 4)
            params['embedding_dim'] = min(params['embedding_dim'], 64)
        
        return params
    
    @staticmethod
    def suggest_neuprint_hyperparameters(trial, feature_name: str = None):
        spaces = HyperparameterSpaces.get_neuprint_spaces()
        
        params = {}
        for param_name, param_config in spaces.items():
            if isinstance(param_config, tuple):
                if len(param_config) == 3:
                    low, high, param_type = param_config
                    if param_type == 'log':
                        params[param_name] = trial.suggest_float(param_name, low, high, log=True)
                    elif param_type == 'int':
                        params[param_name] = trial.suggest_int(param_name, low, high)
                    elif param_type == 'float':
                        params[param_name] = trial.suggest_float(param_name, low, high)
                else:
                    low, high = param_config
                    params[param_name] = trial.suggest_float(param_name, low, high)
            elif isinstance(param_config, list):
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            else:
                raise ValueError(f"Invalid parameter configuration: {param_config}")
        
        if feature_name:
            params = HyperparameterSpaces._adjust_neuprint_params_for_feature(params, feature_name)
        
        return params
    
    @staticmethod
    def _adjust_neuprint_params_for_feature(params: Dict, feature_name: str) -> Dict:
        if feature_name == 'connection_profile_only':
            params['dropout'] = max(params['dropout'], 0.3)
            params['weight_decay'] = max(params['weight_decay'], 1e-4)
            params['embedding_dim'] = max(params['embedding_dim'], 64)
        elif feature_name == 'pos_only':
            params['hidden_dim'] = min(params['hidden_dim'], 128)
            params['num_layers'] = min(params['num_layers'], 4)
            params['embedding_dim'] = min(params['embedding_dim'], 64)
        elif feature_name == 'isi_only':
            params['embedding_dim'] = max(params['embedding_dim'], 48)
            params['time_dim'] = min(params['time_dim'], 150)
        
        return params
    
    @staticmethod
    def _adjust_params_for_feature(params: Dict, feature_name: str) -> Dict:
        if feature_name == 'gene_only':
            params['hidden_dim'] = max(params['hidden_dim'], 128)
            params['num_layers'] = max(params['num_layers'], 3)
        elif feature_name == 'connection_profile_only':
            params['dropout'] = max(params['dropout'], 0.3)
            params['weight_decay'] = max(params['weight_decay'], 1e-4)
        elif feature_name == 'pos_only':
            params['hidden_dim'] = min(params['hidden_dim'], 128)
            params['num_layers'] = min(params['num_layers'], 4)
        
        return params