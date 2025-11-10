#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch_geometric.data import Data

from .processor import NeuronDataProcessor


class DataCache:
    
    def __init__(self, merge_bilateral: bool = False):
        self.merge_bilateral = merge_bilateral
        self.processor = None
        self._data_cache = {}
        self._is_initialized = False
        self._edge_indices = None
        self._edge_weights = None
        self._labels = None
        
    def initialize(self):
        from ..config.logging_config import get_global_logger
        logger = get_global_logger()
        logger.info("Initializing data cache...")
        
        self.processor = NeuronDataProcessor(merge_bilateral=self.merge_bilateral)
        from ..data.feature_extractors import (
            extract_position_features, 
            extract_connection_profile_features
        )
        from ..data.isi_extractor import extract_isi_trials
        
        self._position_features = extract_position_features(self.processor.funatlas, self.processor.filtered_indices)
        self._connection_profile_features = extract_connection_profile_features(self.processor.funatlas, self.processor.filtered_indices)
        
        trials_features, valid_mask, neuron_ids, num_trials_per_neuron = extract_isi_trials(
            self.processor.funatlas,
            self.processor.filtered_indices,
            n_isi_bins=90,
            window_duration=20,
            max_datasets=127,
            min_isi_count=3,
            return_valid_mask=True
        )
        
        if trials_features is None:
            raise ValueError("Failed to extract ISI features")
        
        isi_features_valid = np.mean(trials_features, axis=1)
        n_neurons = self.processor.funatlas.n_neurons
        self._isi_features = np.zeros((n_neurons, 90))
        self._isi_valid_mask = np.zeros(n_neurons, dtype=bool)
        
        for i, neuron_id in enumerate(neuron_ids):
            if neuron_id < n_neurons:
                self._isi_features[neuron_id] = isi_features_valid[i]
                self._isi_valid_mask[neuron_id] = True
        from ..data.feature_extractors import extract_graph_structure
        self._edge_indices, self._edge_weights = extract_graph_structure(
            self.processor.funatlas, 
            self.processor.filtered_indices, 
            use_only_functional=True
        )
        
        self._labels = self.processor.create_labels()
        
        for feature_name in ['pos_only', 'connection_profile_only', 'isi_only']:
            data = self._create_pyg_data_from_cached_features(feature_name)
            self._data_cache[feature_name] = data
        
        self._is_initialized = True
        logger.info(f"Data cache initialized: {len(self._data_cache)} feature combinations cached")
    
    def _create_pyg_data_from_cached_features(self, feature_selection: str) -> Data:
        if feature_selection == 'pos_only':
            node_features = self._position_features
            filter_mask = None
        elif feature_selection == 'connection_profile_only':
            node_features = self._connection_profile_features
            filter_mask = None
        elif feature_selection == 'isi_only':
            node_features = self._isi_features
            filter_mask = self._isi_valid_mask
        
        if filter_mask is not None:
            node_features = node_features[filter_mask]
            labels = self._labels[filter_mask]
            
            old_to_new = {}
            new_idx = 0
            for old_idx in range(len(filter_mask)):
                if filter_mask[old_idx]:
                    old_to_new[old_idx] = new_idx
                    new_idx += 1
            
            valid_edges = []
            valid_weights = []
            for i in range(self._edge_indices.shape[1]):
                src = self._edge_indices[0, i]
                dst = self._edge_indices[1, i]
                if src in old_to_new and dst in old_to_new:
                    valid_edges.append([old_to_new[src], old_to_new[dst]])
                    valid_weights.append(self._edge_weights[i])
            
            if len(valid_edges) > 0:
                edge_indices = np.array(valid_edges).T
                edge_weights = np.array(valid_weights)
            else:
                edge_indices = np.array([[], []], dtype=int)
                edge_weights = np.array([])
            
        else:
            labels = self._labels
            edge_indices = self._edge_indices
            edge_weights = self._edge_weights
        
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1) if len(edge_weights) > 0 else torch.zeros((0, 1), dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        return data
    
    def get_data(self, feature_selection: str) -> Data:
        return self._data_cache[feature_selection]
    
    def get_processor(self) -> NeuronDataProcessor:
        return self.processor
    
_global_cache = None


def initialize_global_cache(merge_bilateral: bool = False) -> DataCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = DataCache(merge_bilateral=merge_bilateral)
    
    _global_cache.initialize()
    return _global_cache
