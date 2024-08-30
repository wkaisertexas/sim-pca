# Simulation Principle Component Analysis

The goal of this repository is to use NVPerf to record commonly used GPU benchmarks and compare them see the similarities and differences between benchmarks and real-world applications.

## Selected Models

I want to select the most popular models on HuggingFace, run them on slurm with profilers and compare the results.

The selected models include:

- [Flux.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) which is a text to image model
- [Phi-3.5-MoE-instruct](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)
- [Jamba 1.5 Mini](https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini)
- [Llama 3.1 Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [Mistral NeMo Minitron](https://huggingface.co/nvidia/Mistral-NeMo-Minitron-8B-Base)

There were the top results (not counting duplicate results) on the day that I looked at the Hugging Face Leaderboards.

## Running this Project

This project uses [Poetry](https://python-poetry.org/) for the Python build system. To use poetry, you must first install it:

```bash
python3 -m pip install poetry
```

Then, alias the following into your `.bashrc`.

```bash
source /etc/profile.d/modules.sh

export TRANSFORMERS_CACHE=/p/archgpuprofiling/.cache
alias poetry="python3 -m poetry"
module load python3
BASE_DIR="/p/archgpuprofiling/.poetry/"

export POETRY_VIRTUALENVS_PATH="$BASE_DIR/virtualenvs"
export POETRY_CONFIG_DIR="$BASE_DIR/config"
export POETRY_DATA_DIR="$BASE_DIR/data"
export POETRY_CACHE_DIR="$BASE_DIR/cache"
export HF_DATASETS_CACHE="$BASE_DIR"
export HF_HOME="$BASE_DIR"
```

These environment variables will force HuggingFace and Poetry to use the project directory (`/p/archgpuprofiling`) instead of taking up space on your home directory which is limited to 100 gb.

After this, you can install the virtual environment by running:

```bash
poetry install
```

After which you can run any python script using `poetry run`.

```bash
poetry run python src/models/flux.py
```

## Collecting Traces

A standardized command was used to collect traces using `nvprof` instead of `ncu` because for some reason, `ncu` does not like the CS servers. For this proof of concept, however, this should not matter.

```bash
ncu --metrics sm__warps_active.avg.per_cycle_active,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__maximum_warps_per_active_cycle_pct,sm__maximum_warps_avg_per_active_cycle,sm__cycles_active.avg,lts__throughput.avg.pct_of_peak_sustained_elapsed,launch__waves_per_multiprocessor,launch__thread_count,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block_driver,launch__shared_mem_per_block,launch__shared_mem_config_size,launch__registers_per_thread,launch__occupancy_per_shared_mem_size,launch__occupancy_per_register_count,launch__occupancy_per_block_size,launch__occupancy_limit_warps,launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers,launch__occupancy_limit_blocks,launch__grid_size,launch__func_cache_config,launch__block_size,l1tex__throughput.avg.pct_of_peak_sustained_active,gpu__time_duration.sum,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,gpc__cycles_elapsed.max,gpc__cycles_elapsed.avg.per_second,breakdown:sm__throughput.avg.pct_of_peak_sustained_elapsed,breakdown:gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,launch__occupancy_per_cluster_size,launch__occupancy_cluster_pct,launch__occupancy_cluster_gpu_pct,launch__cluster_size,launch__cluster_scheduling_policy,launch__cluster_max_potential_size,launch__cluster_max_active,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,dram__cycles_elapsed.avg.per_second --target-processes all --csv -o out -f bash ./run 

ncu --import out.csv.ncu-rep --csv >> out.csv
```

This exports the metrics listed on a per-kernel basis. Figuring out how to merge these kernels and give weight is ambiguous.