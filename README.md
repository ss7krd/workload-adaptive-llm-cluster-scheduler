## Workload-Adaptive LLM Cluster Scheduler or Request Router (EuroSys '26)

In this project, we challenge the conventional request routing paradigm that focuses solely on load balancing across multiple instances of an LLM. Through systematic experiments, we discover that load balancing alone is insufficient for LLM workloads with diverse request characteristics. Unlike traditional deep learning inference where requests have uniform properties, LLM requests vary significantly in both prompt (prefill) and response (decode) lengths. This diversity means that even with perfectly balanced resource usage, the compute layout–organization of tokens across batches within each instance–ultimately determines the latency metrics. We design AdaGen as a workload-adaptive routing scheduler that progressively optimizes compute layouts across instances by leveraging the diversity pattern present in the workload. Further details can be found in our EuroSys '26 paper.

vLLM is used as the inference engine for this project.

The main folders are AdaGen and Graph. Under AdaGen folder, ```adagen/benchmark/benchmark_runner.py``` file consists of the scheduling logics following the paper.  Graph folder consists of the matplotlib codes to produce each figure.

The necessary libraries and dependencies of the virtual environment required to run the project are given in ```requirements.txt``` file. To setup a new virtual environment using ```mamba``` or any other package manager, one needs to install these libraries using pip.


After setting up virtual environment and downloading AdaGen, in order to run the code, please change the required figure_name and scheduler_name fields in the config.py file. Then, run the ```adagen/benchmark/main.py``` file using the following command:
```
sudo python3.10 main.py
```

After getting the outputs, please use the matplotlib files in the Graph folder to generate the final plots. The matplotlib files already have the necessary data embedded to produce each figure. For each figure, the matplotlib file name is the same as the figure name used in the latex file of the paper.

A few points to remember:

- The codes have been tested in H100 GPUs. They should also work on A100 GPUs, but not on V100 GPUs.
- The Nvidia cuda version needs to be at least 12.1
- The torch version needs to be at least 2.3
- The supported attention backends are FlashAttention and FlashInfer.
