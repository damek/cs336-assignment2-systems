#!/usr/bin/env python3
import subprocess
import pandas as pd
from pathlib import Path
import re

def extract_forward_pass_timings():
    """Extract forward pass timings from nsys reports and save to markdown."""
    
    # Get all nsys report files
    nsys_dir = Path("../outputs/nsys")
    nsys_files = sorted(nsys_dir.glob("*.nsys-rep"))
    
    if not nsys_files:
        print("No .nsys-rep files found")
        return
    
    # Extract data
    data = []
    for f in nsys_files:
        # Run nsys stats command
        cmd = ["nsys", "stats", "--report", "nvtx_sum", str(f)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error processing {f.name}: {result.stderr}")
            continue
            
        # Extract forward_pass line
        for line in result.stdout.splitlines():
            if "forward_pass" in line.lower():
                # Parse the line (assuming format: percentage duration count avg min max sum stddev type name)
                parts = line.split()
                if len(parts) >= 10:
                    data.append({
                        'file': f.stem,
                        'duration_ns': int(parts[1]),
                        'percentage': float(parts[0])
                    })
                break
    
    if not data:
        print("No forward_pass timings found")
        return
        
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert nanoseconds to milliseconds for readability
    df['duration_ms'] = df['duration_ns'] / 1_000_000
    
    # Return the dataframe without saving or printing
    return df

def extract_kernels_by_nvtx_range(nvtx_range_name, top_n=1):
    """Extract CUDA kernels within a specific NVTX range from nsys reports."""
    
    # Get all nsys report files
    nsys_dir = Path("../outputs/nsys")
    nsys_files = sorted(nsys_dir.glob("*.nsys-rep"))
    
    if not nsys_files:
        print("No .nsys-rep files found")
        return None, None
    
    all_kernel_data = []
    
    for f in nsys_files:
        # Run nsys stats command for NVTX kernel summary
        cmd = ["nsys", "stats", "--report", "nvtx_kern_sum", str(f)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error processing {f.name}: {result.stderr}")
            continue
        
        # Parse output
        lines = result.stdout.splitlines()
        in_target_range = False
        
        for line in lines:
            # Check if we're in the target NVTX range section
            if f":{nvtx_range_name}" in line and "PushPop" in line:
                in_target_range = True
                # Parse the kernel line
                # Format: :nvtx_range PushPop start end count calls duration avg med min max stddev name
                parts = line.split(None, 11)  # Split into max 12 parts
                if len(parts) >= 12:
                    try:
                        kernel_data = {
                            'file': f.stem,
                            'count': int(parts[5]),
                            'duration_ns': int(parts[6]),
                            'avg_ns': float(parts[7]),
                            'kernel_name': parts[11]
                        }
                        all_kernel_data.append(kernel_data)
                    except (ValueError, IndexError):
                        continue
            elif in_target_range and line.strip() and not line.startswith(' :'):
                # We've moved to a different NVTX range
                in_target_range = False
    
    if not all_kernel_data:
        return None, None
    
    # Create DataFrame
    df = pd.DataFrame(all_kernel_data)
    
    # Find top kernels per file
    results = []
    for file in df['file'].unique():
        file_df = df[df['file'] == file]
        # Sort by total duration and get top N kernels
        top_kernels = file_df.nlargest(top_n, 'duration_ns')
        
        for idx, kernel in top_kernels.iterrows():
            results.append({
                'file': file,
                'top_kernel': kernel['kernel_name'][:80] + '...' if len(kernel['kernel_name']) > 80 else kernel['kernel_name'],
                'count': kernel['count'],
                'total_time_ms': kernel['duration_ns'] / 1_000_000,
                'avg_time_us': kernel['avg_ns'] / 1000
            })
    
    results_df = pd.DataFrame(results)
    return results_df, df

def create_nsys_analysis_report():
    """Create a comprehensive report answering all nsys profiling questions."""
    
    output_file = Path("../outputs/nsys_analysis_report.md")
    
    # Build the markdown content
    content = []
    content.append("# Nsight Systems Profiling Analysis Report\n\n")
    
    # Add kernel glossary
    content.append("## Kernel Type Glossary\n\n")
    content.append("- **cutlass::Kernel2**: Matrix multiplication (GEMM) operations\n")
    content.append("- **elementwise_kernel**: Element-wise operations (add, multiply, activation functions)\n")
    content.append("- **vectorized_elementwise_kernel**: Optimized element-wise operations\n")
    content.append("- **reduce_kernel**: Reduction operations (sum, mean, max)\n")
    content.append("- **sigmoid_kernel**: Sigmoid activation function\n")
    content.append("- **exp_kernel**: Exponential function\n")
    content.append("- **softmax**: Softmax normalization\n\n")
    
    # Question (a)
    content.append("## (a) What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?\n\n")
    content.append("**Deliverable:** A 1-2 sentence response.\n\n")
    
    # Extract forward pass timings
    timings_df = extract_forward_pass_timings()
    if timings_df is not None:
        content.append("### Forward Pass Timings\n\n")
        content.append(timings_df[['file', 'duration_ms', 'percentage']].to_markdown(index=False))
        content.append("\n\n**Answer:** [TO BE FILLED: Compare these timings with Python standard library measurements]\n\n")
    
    # Question (b)
    content.append("## (b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes?\n\n")
    content.append("**Deliverable:** A 1-2 sentence response.\n\n")
    
    # Extract kernel data for forward pass
    forward_top_kernels_df, forward_all_kernels_df = extract_kernels_by_nvtx_range("forward_pass")
    backward_top_kernels_df, backward_all_kernels_df = extract_kernels_by_nvtx_range("backward")
    
    if forward_top_kernels_df is not None:
        content.append("### Top CUDA Kernels in Forward Pass by Model Configuration\n\n")
        content.append(forward_top_kernels_df.to_markdown(index=False))
        content.append("\n\n")
        
        # Also show overall top kernel for forward pass
        if forward_all_kernels_df is not None:
            forward_top_overall = forward_all_kernels_df.nlargest(1, 'duration_ns').iloc[0]
            content.append(f"**Forward Pass Answer:** The CUDA kernel that takes the most cumulative GPU time is `{forward_top_overall['kernel_name'][:80]}...`, ")
            content.append(f"invoked {forward_top_overall['count']} times during a single forward pass.\n\n")
    
    if backward_top_kernels_df is not None:
        content.append("### Top CUDA Kernels in Backward Pass by Model Configuration\n\n")
        content.append(backward_top_kernels_df.to_markdown(index=False))
        content.append("\n\n")
        
        # Also show overall top kernel for backward pass
        if backward_all_kernels_df is not None:
            backward_top_overall = backward_all_kernels_df.nlargest(1, 'duration_ns').iloc[0]
            content.append(f"**Backward Pass Answer:** The CUDA kernel that takes the most cumulative GPU time is `{backward_top_overall['kernel_name'][:80]}...`, ")
            content.append(f"invoked {backward_top_overall['count']} times during a single backward pass.\n\n")
            
            # Compare forward and backward
            if forward_all_kernels_df is not None:
                same_kernel = forward_top_overall['kernel_name'] == backward_top_overall['kernel_name']
                content.append(f"**Comparison:** The top kernel is {'the same' if same_kernel else 'different'} for forward and backward passes.\n\n")
    
    # Question (c)
    content.append("## (c) What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?\n\n")
    content.append("**Deliverable:** A 1-2 sentence response.\n\n")
    
    # Get top 5 kernels for forward pass
    top5_forward_df, _ = extract_kernels_by_nvtx_range("forward_pass", top_n=5)
    if top5_forward_df is not None:
        content.append("### Top 5 CUDA Kernels in Forward Pass by Model Configuration\n\n")
        content.append(top5_forward_df.to_markdown(index=False))
        content.append("\n\n")
    
    content.append("## (d) Profile running one complete training step with your implementation of AdamW. How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?\n\n")
    content.append("**Deliverable:** A 1-2 sentence response.\n\n")
    content.append("**Answer:** [TO BE IMPLEMENTED]\n\n")
    
    content.append("## (e) Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?\n\n")
    content.append("**Deliverable:** A 1-2 sentence response.\n\n")
    content.append("**Answer:** [TO BE IMPLEMENTED]\n\n")
    
    # Join all content and write to file
    full_content = ''.join(content)
    with open(output_file, 'w') as f:
        f.write(full_content)
    
    # Print the entire markdown to terminal
    print(full_content)
    print(f"\nAnalysis also saved to {output_file}")

if __name__ == "__main__":
    create_nsys_analysis_report()