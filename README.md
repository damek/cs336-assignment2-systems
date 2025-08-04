# Following along

Currently writing a twitter thread here: 

https://x.com/damekdavis/status/1949507725626347825


## Problems: 

- End to end profiling with Timeit
  - Run: [benchmark_table1.1.2.sh](cs336_systems/benchmarking_scripts/benchmark_table1.1.2.sh)
  - Results: [2025-07-28_table1.1.2.csv](cs336_systems/outputs/csv/2025-07-28_table1.1.2.csv)
- Nsys profiling
  - Run the profiles: [profile_nsight.sh](cs336_systems/benchmarking_scripts/profile_nsight.sh)
  - Analyze the reports: [analyze_nsys_reports.py](cs336_systems/benchmarking_scripts/analyze_nsys_reports.py)
  - Results: [nsys_analysis_report.md](cs336_systems/outputs/nsys_analysis_report.md)
- Mixed Precision
  - Different types of accumulation 
    - Run code [mixed_precision_accumulation.py](cs336_systems/mixed_precision_accumulation.py)
    - Results: [mixed_precision_accumulation.md](cs336_systems/outputs/mixed_precision_accumulation.md)
  - torch.autocast dtypes
    - Run code [autocast_dtypes.py](cs336_systems/autocast_dtypes.py)
    - Reults: [autocast_dtypes.md](cs336_systems/outputs/autocast_dtypes.md)

# Getting started with my code

## Running the standalone benchmarking script: 

```bash
uv run benchmarking_script.py --num_layers 12 --num_heads 12 --d_ff 3072 --d_model 76008 --context_length 1024 --rope_theta 10000 --vocab_size 10000 --output_csv "times.csv" --num_warmup 1 --num_benchmark 2
```

## Runai 
### Setting up the environment
Setting up the environment
```bash
runai submit cs336-dev \ -p <user> \  -i nvcr.io/nvidia/pytorch:25.06-py3 \  -g 1 --interactive --attach \  --command -- bash
git clone https://github.com/damek/cs336-assignment2-systems.git
pip install uv
cd cs336-assignment2-systems
export PATH="$HOME/.local/bin:$PATH"
uv sync
uv venv
source .venv/bin/activate
```

### Getting files out of the pod

If you've remote logged into the kubernetes server
```bash
# From within the pod, get the podname and the name space
echo "POD name    : $HOSTNAME"
cat /var/run/secrets/kubernetes.io/serviceaccount/namespace

# Outside the pod, set the pod and namespace variables 
POD=cs336-dev-0-0 # for example
NS=runai-?? # leaving this for myself to fill in.

# Outside the pod create a place to store things from the pod
mkdir -p ~/nsys_traces          # e.g., here is where i'll store the nsys_traces

# Now copy files from within the pod 
kubectl cp -n $NS  $POD:/workspace/cs336-assignment2-systems/cs336_systems/ouputs/nsys/nsys.tgz ~/nsys_traces/nsys.tgz

# Now you can copy and paste the file from the VS code gui (in my case). 
# Otherwise you can just use scp.
```


# CS336 Spring 2025 Assignment 2: Systems

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment2_systems.pdf](./cs336_spring2025_assignment2_systems.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. If you want to use your own 
  implementation, you can replace this directory with your own implementation.
- [`./cs336_systems`](./cs336_systems): This folder is basically empty! This is the
  module where you will implement your optimized Transformer language model. 
  Feel free to take whatever code you need from assignment 1 (in `cs336-basics`) and copy it 
  over as a starting point. In addition, you will implement distributed training and
  optimization in this module.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO(you): code that you'll write for assignment 2 
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

If you would like to use your own implementation of assignment 1, replace the `cs336-basics`
directory with your own implementation, or edit the outer `pyproject.toml` file to point to your
own implementation.

0. We use `uv` to manage dependencies. You can verify that the code from the `cs336-basics`
package is accessible by running:

```sh
$ uv run python
Using CPython 3.12.10
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 85 packages in 711ms
Python 3.12.10 (main, Apr  9 2025, 04:03:51) [Clang 20.1.0 ] on linux
...
>>> import cs336_basics
>>> 
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.
