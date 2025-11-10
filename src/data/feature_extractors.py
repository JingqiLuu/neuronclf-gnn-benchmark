#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Tuple


def extract_position_features(funatlas, filtered_indices):
    from ..config.data_paths import get_data_paths
    
    n_neurons = len(funatlas.neuron_ids)
    
    data_paths = get_data_paths()
    coord_file = data_paths.get_coord_file()
    
    with open(coord_file, 'r') as f:
        first_line = f.readline().strip()
        coord_neuron_ids = first_line[1:-1].split(' ')
    
    coords = np.loadtxt(coord_file)[:, ::-1]
    position_features = np.zeros((n_neurons, 3))
    
    for i, neuron_id in enumerate(funatlas.neuron_ids):
        if neuron_id in coord_neuron_ids:
            j = coord_neuron_ids.index(neuron_id)
            position_features[i] = coords[j]
        elif neuron_id.endswith('_'):
            base_name = neuron_id[:-1]
            left_neuron = base_name + 'L'
            right_neuron = base_name + 'R'
            
            left_coord = None
            right_coord = None
            
            if left_neuron in coord_neuron_ids:
                left_idx = coord_neuron_ids.index(left_neuron)
                left_coord = coords[left_idx]
            
            if right_neuron in coord_neuron_ids:
                right_idx = coord_neuron_ids.index(right_neuron)
                right_coord = coords[right_idx]
            
            if left_coord is not None and right_coord is not None:
                position_features[i] = (left_coord + right_coord) / 2
            elif left_coord is not None:
                position_features[i] = left_coord
            elif right_coord is not None:
                position_features[i] = right_coord
            else:
                raise ValueError(f"Neuron {neuron_id} not found in coordinate data")
        else:
            raise ValueError(f"Neuron {neuron_id} not found in coordinate data")
    
    return position_features


def extract_connection_profile_features(funatlas, filtered_indices):
    n_neurons = funatlas.n_neurons
    
    edge_indices, edge_weights = extract_graph_structure(funatlas, filtered_indices, use_only_functional=True)
    
    if edge_indices.shape[1] == 0:
        raise ValueError("No edges found in graph structure")
    
    adj_matrix = np.zeros((n_neurons, n_neurons))
    for i in range(edge_indices.shape[1]):
        src, dst = edge_indices[0, i], edge_indices[1, i]
        weight = edge_weights[i]
        adj_matrix[src, dst] = weight
        adj_matrix[dst, src] = weight
    
    return adj_matrix

def extract_graph_structure(funatlas, filtered_indices, use_only_functional: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    edge_dict = {}
    
    if not use_only_functional:
        if hasattr(funatlas, 'aconn_chem') and funatlas.aconn_chem is not None:
            filtered_chem = funatlas.aconn_chem[np.ix_(filtered_indices, filtered_indices)]
            chem_edges = np.where(filtered_chem > 0)
            for i, j in zip(chem_edges[0], chem_edges[1]):
                edge_key = (min(i, j), max(i, j))
                edge_dict[edge_key] = filtered_chem[i, j]
        
        if hasattr(funatlas, 'aconn_gap') and funatlas.aconn_gap is not None:
            filtered_gap = funatlas.aconn_gap[np.ix_(filtered_indices, filtered_indices)]
            gap_edges = np.where(filtered_gap > 0)
            for i, j in zip(gap_edges[0], gap_edges[1]):
                edge_key = (min(i, j), max(i, j))
                if edge_key in edge_dict:
                    edge_dict[edge_key] = max(edge_dict[edge_key], filtered_gap[i, j])
                else:
                    edge_dict[edge_key] = filtered_gap[i, j]
    
    if use_only_functional or not edge_dict:
        if not hasattr(funatlas, 'fconn') or len(funatlas.fconn) == 0:
            raise ValueError("No functional connectivity data available")
        
        signal_corr = funatlas.get_signal_correlations()
        if signal_corr is None or np.all(np.isnan(signal_corr)):
            raise ValueError("Signal correlations are None or all NaN")
        
        if signal_corr.shape[0] > funatlas.n_neurons:
            signal_corr = signal_corr[np.ix_(filtered_indices, filtered_indices)]
        
        func_edges = np.where((signal_corr > 0.3) & (~np.isnan(signal_corr)))
        for i, j in zip(func_edges[0], func_edges[1]):
            if i != j:
                edge_key = (min(i, j), max(i, j))
                if edge_key in edge_dict:
                    edge_dict[edge_key] = max(edge_dict[edge_key], signal_corr[i, j])
                else:
                    edge_dict[edge_key] = signal_corr[i, j]
    
    if not edge_dict:
        raise ValueError("No edges found in graph structure")
    
    edge_indices = []
    edge_weights = []
    for (i, j), weight in edge_dict.items():
        edge_indices.append([i, j])
        edge_weights.append(weight)
    
    edge_indices = np.array(edge_indices).T
    edge_weights = np.array(edge_weights)
    
    return edge_indices, edge_weights