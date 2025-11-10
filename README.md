Workload-Adaptive LLM Cluster Scheduler (EuroSys '26)

The main folders are AdaGen and Graph. Under AdaGen, ```adagen/benchmark/benchmark_runner.py``` file consists of the scheduling logics following the paper. Graph consists of the matplotlib codes to produce each figure.

The environment is already setup under the virtual environment named ```torchEnv``` in syrax-41. Command to enable it: 
```
mamba activate torchEnv
```
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
