import subprocess
import sys
import re

def analyze_file(nsys_file):
    print(f"\nAnalyzing {nsys_file}")
    
    # Get NVTX ranges
    nvtx_cmd = ["nsys", "stats", "--report", "nvtx_sum", "--format", "csv", nsys_file]
    try:
        nvtx_output = subprocess.check_output(nvtx_cmd, text=True, stderr=subprocess.DEVNULL)
        
        # Parse forward pass time
        for line in nvtx_output.split('\n'):
            if 'forward_pass' in line.lower():
                parts = line.split(',')
                if len(parts) > 2:
                    time_ns = float(parts[2])
                    print(f"Forward pass time: {time_ns/1e6:.2f} ms")
                    break
    except:
        print("Could not analyze NVTX ranges")
    
    # Get top kernel
    kernel_cmd = ["nsys", "stats", "--report", "cuda_gpu_sum", "--format", "csv", nsys_file]
    try:
        kernel_output = subprocess.check_output(kernel_cmd, text=True, stderr=subprocess.DEVNULL)
        lines = kernel_output.strip().split('\n')
        
        # Find header
        header_idx = None
        for i, line in enumerate(lines):
            if 'Time' in line or 'Duration' in line:
                header_idx = i
                break
        
        if header_idx and header_idx + 1 < len(lines):
            # Get first data line (top kernel)
            data_line = lines[header_idx + 1]
            parts = data_line.split(',')
            if len(parts) > 0:
                kernel_name = parts[0].strip('"')
                print(f"\nTop kernel: {kernel_name}")
                
                # Check if it's a GEMM
                if any(pattern in kernel_name.lower() for pattern in ['gemm', 'gemv', 'cublas']):
                    print("This is a matrix multiplication kernel")
                else:
                    print("This is NOT a matrix multiplication kernel")
    except:
        print("Could not analyze CUDA kernels")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_file(sys.argv[1])
