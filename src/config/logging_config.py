#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import logging
import datetime


def setup_logging(log_file=None, experiment_type="experiment"):
    if log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f'results/{experiment_type}_log_{timestamp}.log'
    
    os.makedirs('results', exist_ok=True)
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger, log_file


class GlobalLogManager:
    
    def __init__(self):
        self.logger = None
        self.log_file = None
        self.experiment_type = None
        self.is_initialized = False
    
    def initialize(self, experiment_type="experiment"):
        if not self.is_initialized:
            self.experiment_type = experiment_type
            self.logger, self.log_file = setup_logging(experiment_type=experiment_type)
            self.is_initialized = True
            self.logger.info(f"Starting {experiment_type} experiment")
        return self.logger, self.log_file
    
    def get_logger(self):
        if not self.is_initialized:
            self.initialize()
        return self.logger
    
    def get_log_file(self):
        if not self.is_initialized:
            self.initialize()
        return self.log_file
    
    def reset(self):
        self.logger = None
        self.log_file = None
        self.experiment_type = None
        self.is_initialized = False


_global_log_manager = GlobalLogManager()


def get_global_logger(experiment_type="experiment"):
    return _global_log_manager.initialize(experiment_type)[0]


def reset_global_log():
    _global_log_manager.reset()
