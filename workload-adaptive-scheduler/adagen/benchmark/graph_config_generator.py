import sys
sys.path.append("/home/shubhasu/shubhasu/sarathi-serve")
sys.path.append("/usr/local/lib/python3.10/dist-packages")
sys.path.append("/usr/lib/python3/dist-packages")

import logging
import os
import time

import ray
import ray.util.collective as col

import torch
import cupy

import wandb
from tqdm import tqdm

from sarathi import LLMEngine, SamplingParams
from sarathi.benchmark.config import BenchmarkConfig
from sarathi.benchmark.entities import Request
from sarathi.benchmark.request_generator import RequestGeneratorRegistry
from sarathi.benchmark.utils.random import set_seeds
from sarathi.config import ReplicaConfig
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.types import ReplicaResourceMapping, ResourceMapping
from sarathi.utils import get_ip

from sarathi.benchmark.global_variable_tester import MyClass

logger = logging.getLogger(__name__)




class GraphConfigGenerator:

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        if self.config.figure_name == "12a":
            self.config.request_rate_list = [20, 40, 60, 80, 100]
            self.config.model_list = ["Llama3-8B"]
        elif self.config.figure_name == "12b":
            self.config.request_rate_list = [10, 15, 20, 25, 30]
            self.config.model_list = ["Mixtral"]
        elif self.config.figure_name == "12c":
            self.config.request_rate_list = [4, 6, 8, 10, 12]
            self.config.model_list = ["Llama3-70B"]
        elif self.config.figure_name == "13":
            self.config.slo_scale_list = [0.5, 0.75, 1, 1.25, 1.5]
            self.config.model_list = ["Llama3-8B"]
        elif self.config.figure_name == "14":
            self.config.model_list = ["Llama3-8B", "Mixtral", "Llama3-70B"]
        elif self.config.figure_name == "17":
            self.config.model_list = ["Llama3-8B"]
            self.config.request_rate_list = [1000, 2000,3000, 4000, 5000]
        elif self.config.figure_name == "3a":
            self.config.model_list = ["Llama3-8B"]
        elif self.config.figure_name == "3b":
            self.config.model_list = ["Mixtral"]
        elif self.config.figure_name == "5a":
            self.config.model_list = ["Llama3-8B"]
        elif self.config.figure_name == "5b":
            self.config.model_list = ["Mixtral"]
        elif self.config.figure_name == "6a":
            self.config.model_list = ["Llama3-8B"]
        elif self.config.figure_name == "6b":
            self.config.model_list = ["Mixtral"]
        elif self.config.figure_name == "7a":
            self.config.request_rate_list = [20, 40, 60, 80, 100]
            self.config.model_list = ["Llama3-8B"]
        elif self.config.figure_name == "7b":
            self.config.request_rate_list = [10, 15, 20, 25, 30]
            self.config.model_list = ["Mixtral"]
        elif self.config.figure_name == "8a":
            self.config.request_rate_list = [4, 6, 8, 10, 12]
            self.config.model_list = ["Llama3-70B"]
        elif self.config.figure_name == "8b":
            self.config.slo_scale_list = [0.5, 0.75, 1, 1.25, 1.5]
            self.config.model_list = ["Llama3-8B"]
        elif self.config.figure_name == "15":
            self.config.model_list = ["Llama3-8B", "Mixtral", "Llama3-70B"]
        elif self.config.figure_name == "16":
            self.config.model_list = ["Llama3-8B"]
        elif self.config.figure_name == "19":
            self.config.request_rate_list = [10, 15, 20, 25, 30]
            self.config.model_list = ["Mixtral"]
        elif self.config.figure_name == "20":
            self.config.request_rate_list = [4, 6, 8, 10, 12]
            self.config.model_list = ["Llama3-70B"]
        elif self.config.figure_name == "21":
            self.config.slo_scale_list = [0.5, 0.75, 1, 1.25, 1.5]
            self.config.model_list = ["Llama3-8B"]
            


