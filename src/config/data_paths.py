#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root


class DataPaths:
    def __init__(self):
        self._project_root = get_project_root()
        self._init_paths()
    
    def _init_paths(self):
        self.dataset_list_file = self._get_path(
            env_var='FUNATLAS_DATASET_LIST',
            default=Path('/your/path/funatlas_list.txt'), # change this to your own path
        )
        
        self.coord_file = self._get_path(
            env_var='NEURON_POSITIONS_FILE',
            default=self._project_root / 'pumpprobe' / 'anatlas_neuron_positions.txt'
        )
        
        self.neurotransmitter_labels_file = self._get_path(
            env_var='NEUROTRANSMITTER_LABELS_FILE',
            default=self._project_root / 'pumpprobe' / 'neurotransmitter_labels_simplified.json'
        )
        
        self.functional_labels_file = self._get_path(
            env_var='FUNCTIONAL_LABELS_FILE',
            default=self._project_root / 'pumpprobe' / 'sensoryintermotor_ids.json'
        )
    
    def _get_path(
        self, 
        env_var: str, 
        default: Path
    ) -> Path:
        env_path = os.environ.get(env_var)
        if env_path:
            path = Path(env_path).expanduser().resolve()
            if path.exists():
                return path
        return default.resolve()
    
    def get_dataset_list_file(self) -> str:
        return str(self.dataset_list_file)
    
    def get_coord_file(self) -> str:
        return str(self.coord_file)
    
    def get_neurotransmitter_labels_file(self) -> str:
        return str(self.neurotransmitter_labels_file)
    
    def get_functional_labels_file(self) -> str:
        return str(self.functional_labels_file)
    
    def validate_paths(self) -> dict:
        paths = {
            'dataset_list_file': self.dataset_list_file,
            'coord_file': self.coord_file,
            'neurotransmitter_labels_file': self.neurotransmitter_labels_file,
            'functional_labels_file': self.functional_labels_file,
        }
        
        results = {}
        for name, path in paths.items():
            exists = path.exists()
            results[name] = {
                'path': str(path),
                'exists': exists,
                'type': 'env_var' if name.upper() in os.environ else 'default'
            }
        
        return results
    


_global_data_paths: Optional[DataPaths] = None


def get_data_paths() -> DataPaths:
    global _global_data_paths
    if _global_data_paths is None:
        _global_data_paths = DataPaths()
    return _global_data_paths
