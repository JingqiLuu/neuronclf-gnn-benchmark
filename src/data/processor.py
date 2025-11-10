#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pumpprobe_path = os.path.join(project_root, 'pumpprobe')
if pumpprobe_path not in sys.path:
    sys.path.insert(0, pumpprobe_path)
from Funatlas import Funatlas

from .feature_extractors import (
    extract_position_features,
    extract_connection_profile_features,
    extract_graph_structure
)
from ..config.data_paths import get_data_paths


class NeuronDataProcessor:
    
    def __init__(self, merge_bilateral=False, merge_dorsoventral=False, 
                 merge_numbered=False, merge_AWC=False):
        np.random.seed(42)
        import random
        random.seed(42)
        
        from ..config.logging_config import get_global_logger
        logger = get_global_logger()
        logger.info("Initializing dataset...")
        
        data_paths = get_data_paths()
        dataset_list_file = data_paths.get_dataset_list_file()
        
        self.funatlas = Funatlas.from_datasets(
            dataset_list_file, 
            merge_bilateral=merge_bilateral, 
            merge_dorsoventral=merge_dorsoventral,
            merge_numbered=merge_numbered,
            merge_AWC=merge_AWC,
            load_signal=True, 
            verbose=True
        )
        
        self.load_neuron_labels()
        
        logger.info(f"Dataset initialized: {self.funatlas.n_neurons} neurons")
    
    def load_neuron_labels(self):
        label_type = os.environ.get('NEURON_LABEL_TYPE', 'functional')
        
        if label_type == 'neurotransmitter':
            data_paths = get_data_paths()
            nt_file = data_paths.get_neurotransmitter_labels_file()
            with open(nt_file, 'r') as f:
                nt_data = json.load(f)
            
            self.neuron_types = {}
            for neuron_type, neuron_list in nt_data.items():
                for neuron in neuron_list:
                    self.neuron_types[neuron] = neuron_type
            
        elif hasattr(self.funatlas, 'sim') and self.funatlas.sim is not None:
            self.neuron_types = {}
            for neuron_type, neuron_list in self.funatlas.sim.items():
                for neuron in neuron_list:
                    self.neuron_types[neuron] = neuron_type
        else:
            data_paths = get_data_paths()
            sim_file = data_paths.get_functional_labels_file()
            with open(sim_file, 'r') as f:
                sim_data = json.load(f)
            
            self.neuron_types = {}
            for neuron_type, neuron_list in sim_data.items():
                for neuron in neuron_list:
                    self.neuron_types[neuron] = neuron_type
        
        if self.funatlas.merge_bilateral:
            for neuron_id in self.funatlas.neuron_ids:
                if neuron_id not in self.neuron_types:
                    found_label = None
                    
                    if neuron_id.endswith('_'):
                        base_name = neuron_id[:-1]
                        left_neuron = base_name + 'L'
                        right_neuron = base_name + 'R'
                    else:
                        left_neuron = neuron_id + 'L'
                        right_neuron = neuron_id + 'R'
                    
                    if left_neuron in self.neuron_types:
                        found_label = self.neuron_types[left_neuron]
                    elif right_neuron in self.neuron_types:
                        found_label = self.neuron_types[right_neuron]
                    
                    if found_label:
                        self.neuron_types[neuron_id] = found_label
        
        
        filtered_neuron_ids = []
        filtered_indices = []
        
        for i, neuron_id in enumerate(self.funatlas.neuron_ids):
            if neuron_id in self.neuron_types:
                filtered_neuron_ids.append(neuron_id)
                filtered_indices.append(i)
        
        self.funatlas.neuron_ids = filtered_neuron_ids
        self.funatlas.n_neurons = len(filtered_neuron_ids)
        
        self.filtered_indices = np.array(filtered_indices)
    
    def extract_node_features(self, feature_selection=None) -> np.ndarray:
        if feature_selection == 'pos_only':
            node_features = extract_position_features(self.funatlas, self.filtered_indices)
        elif feature_selection == 'connection_profile_only':
            node_features = extract_connection_profile_features(self.funatlas, self.filtered_indices)
        elif feature_selection == 'isi_only':
            return None
        else:
            raise ValueError(f"Unknown feature_selection: {feature_selection}")
        
        return node_features
    
    def create_labels(self) -> np.ndarray:
        labels = []
        for neuron_id in self.funatlas.neuron_ids:
            neuron_type = self.neuron_types[neuron_id]
            labels.append(neuron_type)
        
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        return encoded_labels
    
    def create_pyg_data(self, feature_selection=None) -> Data:
        node_features = self.extract_node_features(feature_selection)
        edge_indices, edge_weights = extract_graph_structure(self.funatlas, self.filtered_indices, use_only_functional=True)
        labels = self.create_labels()
        
        self.edge_indices = edge_indices
        self.edge_weights = edge_weights
        
        scaler = StandardScaler()
        node_features = scaler.fit_transform(node_features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(labels, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        return data
