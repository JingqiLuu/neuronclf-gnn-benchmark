#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
def detect_events(neural_signal: np.ndarray, 
                  method: str = 'threshold',
                  threshold: float = 2.0,
                  min_distance: int = 3) -> np.ndarray:
    if len(neural_signal) == 0:
        return np.array([])
    if method == 'threshold':
        mean = np.mean(neural_signal)
        std = np.std(neural_signal)
        threshold_value = mean + threshold * std
        above_threshold = neural_signal > threshold_value
        events = np.where(np.diff(above_threshold.astype(int)) > 0)[0] + 1
    elif method == 'peak':
        from scipy import signal
        mean = np.mean(neural_signal)
        std = np.std(neural_signal)
        height = mean + threshold * std
        events, _ = signal.find_peaks(neural_signal, 
                                      height=height, 
                                      distance=min_distance)
    return events


def extract_isi_trials(funatlas, filtered_indices, n_isi_bins=30, 
                               window_duration=30, max_datasets=127, 
                               min_isi_count=1, return_valid_mask=True):
    n_neurons = funatlas.n_neurons
    sampling_rate = 2  # Hz

    if not hasattr(funatlas, 'sig') or len(funatlas.sig) == 0:
        raise ValueError("No signal data found")
    
    neuron_trials = [[] for _ in range(n_neurons)]
    
    window_size = int(window_duration * sampling_rate)
    
    
    valid_datasets = 0
    total_windows = 0
    
    for dataset_idx, sig_data in enumerate(funatlas.sig):
        if valid_datasets >= max_datasets:
            break
        
        if not hasattr(sig_data, 'data') or sig_data.data is None:
            continue
        
        signal_matrix = sig_data.data  # (timepoints, original_neuron_count)
        atlas_mapping = funatlas.atlas_i[dataset_idx]
        total_timepoints = signal_matrix.shape[0]
        
        num_windows = int(total_timepoints // window_size)
        
        if num_windows == 0:
            continue
        
        for sig_neuron_idx in range(signal_matrix.shape[1]):
            atlas_neuron_idx = atlas_mapping[sig_neuron_idx]
            
            if 0 <= atlas_neuron_idx < n_neurons:
                neuron_trace = signal_matrix[:, sig_neuron_idx]
                for window_idx in range(num_windows):
                    start_idx = window_idx * window_size
                    end_idx = start_idx + window_size
                    
                    window_trace = neuron_trace[start_idx:end_idx]
                    
                    events = detect_events(window_trace, method='threshold', threshold=1.0)
                    
                    if len(events) >= 2:
                        isi_values = np.diff(events)
                        
                        if len(isi_values) >= min_isi_count:
                            isi_distribution = np.zeros(n_isi_bins)
                            
                            for isi_val in isi_values:
                                if 1 <= isi_val < n_isi_bins + 1:
                                    isi_distribution[int(isi_val) - 1] += 1
                            
                            if np.sum(isi_distribution) > 0:
                                isi_distribution = isi_distribution / np.sum(isi_distribution)
                            
                            neuron_trials[atlas_neuron_idx].append(isi_distribution)
                            total_windows += 1
        
        valid_datasets += 1
    neurons_with_trials = []
    for neuron_idx, trials in enumerate(neuron_trials):
        if len(trials) > 0:
            neurons_with_trials.append(neuron_idx)
    
    if len(neurons_with_trials) == 0:
        raise ValueError("No neurons have sufficient trials data")
    
    max_trials = max(len(neuron_trials[idx]) for idx in neurons_with_trials)
    valid_neurons = []
    trials_features_list = []
    neuron_ids_list = []
    num_trials_list = []
    for neuron_idx in neurons_with_trials:
        trials = neuron_trials[neuron_idx]
        num_real_trials = len(trials)
        trials_array = np.array(trials)  # (num_trials, n_isi_bins)
        if num_real_trials < max_trials:
            padding = np.repeat(trials_array[-1:], max_trials - num_real_trials, axis=0)
            trials_array = np.vstack([trials_array, padding])
        valid_neurons.append(neuron_idx)
        trials_features_list.append(trials_array)
        neuron_ids_list.append(neuron_idx)
        num_trials_list.append(num_real_trials)
    trials_features = np.array(trials_features_list)  # (num_valid_neurons, max_trials, n_isi_bins)
    neuron_ids = np.array(neuron_ids_list)
    num_trials_per_neuron = np.array(num_trials_list)
    if return_valid_mask:
        valid_mask = np.zeros(n_neurons, dtype=bool)
        valid_mask[valid_neurons] = True
        return trials_features, valid_mask, neuron_ids, num_trials_per_neuron
    else:
        return trials_features, neuron_ids, num_trials_per_neuron

