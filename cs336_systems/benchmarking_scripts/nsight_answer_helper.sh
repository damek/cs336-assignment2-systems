#!/bin/bash
# Helper script to analyze nsight profiles and generate answers

set -euo pipefail

NSYS_DIR="${1:-../outputs/nsys}"

echo "Nsight Profile Analysis Helper"
echo "=============================="
echo "This script will help you answer the nsight profiling questions"
echo ""

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "ERROR: nsys command not found. Make sure NVIDIA Nsight Systems is installed."
    exit 1
fi

# Check if directory exists
if [ ! -d "$NSYS_DIR" ]; then
    echo "ERROR: Directory $NSYS_DIR not found!"
    exit 1
fi

# Find .nsys files
NSYS_FILES=$(find "$NSYS_DIR" -name "*.nsys-rep" | head -5)
if [ -z "$NSYS_FILES" ]; then
    echo "ERROR: No .nsys files found in $NSYS_DIR"
    exit 1
fi

echo "Found nsys files:"
echo "$NSYS_FILES" | head -5
echo ""

# Create analysis directory
ANALYSIS_DIR="nsight_analysis_results"
mkdir -p "$ANALYSIS_DIR"
cd "$ANALYSIS_DIR"

# Copy Python scripts if they don't exist
if [ ! -f "analyze_nsight_profiles.py" ]; then
    echo "Creating analysis scripts..."
    # You would copy the scripts here or ensure they're in the path
fi

echo "Running analysis..."
echo ""

# Analyze one file in detail as an example
SAMPLE_FILE=$(echo "$NSYS_FILES" | head -1)
echo "Detailed analysis of: $(basename $SAMPLE_FILE)"
echo "================================================"

# Question (a): Forward pass timing
echo ""
echo "Question (a): Forward Pass Timing"
echo "---------------------------------"
nsys stats --report nvtx_sum --format table "$SAMPLE_FILE" 2>/dev/null | grep -E "(forward_pass|Forward)" || echo "No forward_pass NVTX range found"

# Question (b): Top CUDA kernels
echo ""
echo "Question (b): Most Time-Consuming CUDA Kernels"
echo "----------------------------------------------"
echo "Top 5 kernels by total time:"
nsys stats --report cuda_gpu_sum --format table "$SAMPLE_FILE" 2>/dev/null | head -15 || echo "Could not get CUDA kernel summary"

# Question (c): Non-matrix multiply kernels
echo ""
echo "Question (c): Kernel Types Analysis"
echo "-----------------------------------"
echo "Looking for non-GEMM kernels..."
nsys stats --report cuda_gpu_sum --format csv "$SAMPLE_FILE" 2>/dev/null | grep -v -i "gemm\|gemv\|cublas" | head -10 || echo "Analysis failed"

# Create a Python analysis script inline
cat > quick_analysis.py << 'EOF'
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
EOF

# Run the Python analysis
echo ""
echo "Running Python analysis..."
python3 quick_analysis.py "$SAMPLE_FILE"

# Generate answer template
echo ""
echo ""
echo "ANSWER TEMPLATE FOR YOUR WRITEUP"
echo "================================"
echo ""
echo "(a) Forward Pass Timing:"
echo "    The total forward pass time was XX.XX ms, which [matches/differs from] the Python"
echo "    timeit measurements by XX%. This [agreement/difference] is likely due to..."
echo ""
echo "(b) Most Time-Consuming CUDA Kernel:"
echo "    The most time-consuming kernel is [kernel_name], which is invoked XX times"
echo "    during a single forward pass. This kernel accounts for XX% of the total runtime."
echo "    [It is/It is not] the same kernel that dominates the forward+backward pass."
echo ""
echo "(c) Non-Matrix-Multiply Kernels:"
echo "    Besides matrix multiplies, significant time is spent on:"
echo "    - Softmax operations (XX ms)"
echo "    - Elementwise operations like activations (XX ms)"
echo "    - Reduction operations (XX ms)"
echo ""
echo "(d) Training Step Breakdown:"
echo "    When profiling a full training step, the fraction of time in matrix multiplication"
echo "    [increases/decreases] from XX% to XX%. Other kernels like gradient computation"
echo "    and optimizer updates account for XX% of the time."
echo ""
echo "(e) Attention Layer Analysis:"
echo "    Softmax operations in attention take XX ms while matrix multiplications take XX ms,"
echo "    giving a ratio of X:X. This differs from the FLOP ratio because softmax has"
echo "    lower arithmetic intensity and is memory-bandwidth bound."
echo ""
echo "Note: Run the full Python analysis scripts for complete results across all model sizes."