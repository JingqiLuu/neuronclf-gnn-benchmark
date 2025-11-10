#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import argparse
import os

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments.gnn_bayesian_optimization import gnn_bayesian_optimization
from src.experiments.non_graph_bayesian_optimization import non_graph_bayesian_optimization
def main():
    parser = argparse.ArgumentParser(
        description='Neuron type classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Run GNN model optimization
        python main.py --method gnn
        
        # Run non-graph model optimization
        python main.py --method non_graph
        
        # Run all experiments
        python main.py --method all
                """
    )
    parser.add_argument(
        '--method', 
        type=str, 
        choices=['gnn', 'non_graph', 'all'], 
        default='all', 
        help='Experiment method: gnn (GNN models), non_graph (non-graph models), all (all methods)'
    )
    args = parser.parse_args()
    print(f"Selected method: {args.method}\n")
    
    if args.method in ['gnn', 'all']:
        gnn_bayesian_optimization()
    if args.method in ['non_graph', 'all']:
        non_graph_bayesian_optimization()

if __name__ == "__main__":
    main()
