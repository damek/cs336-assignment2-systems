OUT_DIR="../outputs/nsys/ddp_overlap_individual_parameters_benchmarking_nvtx"
mkdir -p "$OUT_DIR"

nsys profile --python-backtrace=cuda -o "$OUT_DIR/naive_ddp" uv run ddp_overlap_individual_parameters_benchmarking_nvtx.py --naive_ddp

nsys profile --python-backtrace=cuda -o "$OUT_DIR/overlap" uv run ddp_overlap_individual_parameters_benchmarking_nvtx.py 