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

def extract_attention_breakdown():
    """Extract softmax vs matrix multiply breakdown within attention layers."""
    
    # Get all nsys report files
    nsys_dir = Path("../outputs/nsys")
    nsys_files = sorted(nsys_dir.glob("*.nsys-rep"))
    
    if not nsys_files:
        print("No .nsys-rep files found")
        return None
    
    attention_data = []
    
    for f in nsys_files:
        # Run nsys stats command for NVTX kernel summary
        cmd = ["nsys", "stats", "--report", "nvtx_kern_sum", str(f)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error processing {f.name}: {result.stderr}")
            continue
        
        # Parse output
        lines = result.stdout.splitlines()
        
        softmax_time_ns = 0
        matmul_time_ns = 0
        
        for line in lines:
            # Look for softmax within attention
            if ":softmax" in line and "PushPop" in line:
                parts = line.split(None, 11)
                if len(parts) >= 12:
                    try:
                        softmax_time_ns += int(parts[6])
                    except (ValueError, IndexError):
                        continue
            
            # Look for matrix multiply operations in attention
            elif (":attn_scores" in line or ":final_matmul" in line or ":output_proj" in line) and "PushPop" in line:
                parts = line.split(None, 11)
                if len(parts) >= 12:
                    try:
                        matmul_time_ns += int(parts[6])
                    except (ValueError, IndexError):
                        continue
        
        if softmax_time_ns > 0 or matmul_time_ns > 0:
            attention_data.append({
                'file': f.stem,
                'softmax_ms': softmax_time_ns / 1_000_000,
                'matmul_ms': matmul_time_ns / 1_000_000,
                'softmax_to_matmul_ratio': softmax_time_ns / matmul_time_ns if matmul_time_ns > 0 else 0
            })
    
    if not attention_data:
        return None
        
    return pd.DataFrame(attention_data)

def extract_kernel_breakdown_by_nvtx_range(nvtx_range_name):
    """Extract kernel breakdown (matrix multiply vs others) for a specific NVTX range."""
    
    # Get all nsys report files
    nsys_dir = Path("../outputs/nsys")
    nsys_files = sorted(nsys_dir.glob("*.nsys-rep"))
    
    if not nsys_files:
        print("No .nsys-rep files found")
        return None
    
    breakdown_data = []
    
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
        
        matmul_time_ns = 0
        other_time_ns = 0
        total_time_ns = 0
        
        for line in lines:
            # Check if we're in the target NVTX range section
            if f":{nvtx_range_name}" in line and "PushPop" in line:
                in_target_range = True
                # Parse the kernel line
                parts = line.split(None, 11)
                if len(parts) >= 12:
                    try:
                        duration_ns = int(parts[6])
                        kernel_name = parts[11]
                        total_time_ns += duration_ns
                        
                        # Check if it's a matrix multiply (CUTLASS GEMM)
                        if "cutlass" in kernel_name.lower() and "gemm" in kernel_name.lower():
                            matmul_time_ns += duration_ns
                        else:
                            other_time_ns += duration_ns
                            
                    except (ValueError, IndexError):
                        continue
            elif in_target_range and line.strip() and not line.startswith(' :'):
                # We've moved to a different NVTX range
                in_target_range = False
        
        if total_time_ns > 0:
            breakdown_data.append({
                'file': f.stem,
                'matmul_time_ms': matmul_time_ns / 1_000_000,
                'other_time_ms': other_time_ns / 1_000_000,
                'total_time_ms': total_time_ns / 1_000_000,
                'matmul_fraction': matmul_time_ns / total_time_ns,
                'other_fraction': other_time_ns / total_time_ns
            })
    
    if not breakdown_data:
        return None
        
    return pd.DataFrame(breakdown_data)

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
    
    # Question (d)
    content.append("## (d) Profile running one complete training step with your implementation of AdamW. How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?\n\n")
    content.append("**Deliverable:** A 1-2 sentence response.\n\n")
    
    # Get kernel breakdown for forward pass and train step
    forward_breakdown_df = extract_kernel_breakdown_by_nvtx_range("forward_pass")
    train_step_breakdown_df = extract_kernel_breakdown_by_nvtx_range("train_step")
    
    if forward_breakdown_df is not None and train_step_breakdown_df is not None:
        # Merge the dataframes
        comparison_df = pd.merge(
            forward_breakdown_df[['file', 'matmul_fraction', 'other_fraction']],
            train_step_breakdown_df[['file', 'matmul_fraction', 'other_fraction']], 
            on='file', 
            suffixes=('_forward', '_train')
        )
        
        # Format percentages
        comparison_df['matmul_forward_%'] = (comparison_df['matmul_fraction_forward'] * 100).round(1)
        comparison_df['matmul_train_%'] = (comparison_df['matmul_fraction_train'] * 100).round(1)
        comparison_df['other_forward_%'] = (comparison_df['other_fraction_forward'] * 100).round(1)
        comparison_df['other_train_%'] = (comparison_df['other_fraction_train'] * 100).round(1)
        
        content.append("### Matrix Multiplication vs Other Kernels: Forward Pass vs Complete Training Step\n\n")
        content.append(comparison_df[['file', 'matmul_forward_%', 'matmul_train_%', 'other_forward_%', 'other_train_%']].to_markdown(index=False))
        content.append("\n\n")
    
    # Question (e)
    content.append("## (e) Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?\n\n")
    content.append("**Deliverable:** A 1-2 sentence response.\n\n")
    
    # Get attention breakdown
    attention_df = extract_attention_breakdown()
    if attention_df is not None:
        content.append("### Softmax vs Matrix Multiplication in Self-Attention\n\n")
        # Format the ratio as percentage
        attention_df['runtime_ratio_%'] = (attention_df['softmax_to_matmul_ratio'] * 100).round(1)
        content.append(attention_df[['file', 'softmax_ms', 'matmul_ms', 'runtime_ratio_%']].to_markdown(index=False))
        content.append("\n\n")
        content.append("**Note:** runtime_ratio_% = (softmax_time / matmul_time) * 100\n\n")
        
        # Add theoretical FLOP ratio information
        content.append("**Theoretical FLOP Analysis:**\n")
        content.append("- **Attention matrix multiplies (per layer):**\n")
        content.append("  - Q*K^T: 2 * seq_len * seq_len * d_head FLOPs\n")
        content.append("  - Softmax(scores) * V: 2 * seq_len * seq_len * d_head FLOPs\n")
        content.append("  - Output projection: 2 * seq_len * d_model * d_model FLOPs\n")
        content.append("- **Softmax (per layer):** seq_len * seq_len FLOPs (for exp and normalization)\n\n")
        content.append("**FLOP Ratio:** For typical transformers, matrix multiplies dominate with O(seq_len² * d_model) vs softmax's O(seq_len²)\n\n")
    
    # Join all content and write to file
    full_content = ''.join(content)
    with open(output_file, 'w') as f:
        f.write(full_content)
    
    # Print the entire markdown to terminal
    print(full_content)
    print(f"\nAnalysis also saved to {output_file}")

if __name__ == "__main__":
    create_nsys_analysis_report()