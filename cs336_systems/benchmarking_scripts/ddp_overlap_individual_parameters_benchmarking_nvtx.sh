nsys profile --python-backtrace=cuda -o ../outputs/nsys/ddp_overlap_individual_parameters_benchmarking_nvtx/naive_ddp uv run ddp_overlap_individual_parameters_benchmarking_nvtx.py --naive_ddp

nsys profile --python-backtrace=cuda -o ../outputs/nsys/ddp_overlap_individual_parameters_benchmarking_nvtx/overlap uv run ddp_overlap_individual_parameters_benchmarking_nvtx.py 