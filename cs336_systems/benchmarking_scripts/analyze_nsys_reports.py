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
    
    # Save to markdown
    output_file = Path("../outputs/forward_pass_timings.md")
    markdown_content = df[['file', 'duration_ms', 'percentage']].to_markdown(index=False)
    
    with open(output_file, 'w') as f:
        f.write("# Forward Pass Timings\n\n")
        f.write(markdown_content)
        f.write("\n")
    
    # Echo to terminal
    print("\nForward Pass Timings:")
    print(markdown_content)
    print(f"\nSaved to {output_file}")
    return df

def extract_forward_pass_kernels():
    """Extract CUDA kernels within forward pass from nsys reports."""
    
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
        in_forward_pass = False
        
        for line in lines:
            # Check if we're in the forward_pass section
            if ":forward_pass" in line and "PushPop" in line:
                in_forward_pass = True
                # Parse the kernel line
                # Format: :forward_pass PushPop start end count calls duration avg med min max stddev name
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
            elif in_forward_pass and line.strip() and not line.startswith(' :'):
                # We've moved to a different NVTX range
                in_forward_pass = False
    
    if not all_kernel_data:
        print("No forward_pass kernel data found")
        return None, None
    
    # Create DataFrame
    df = pd.DataFrame(all_kernel_data)
    
    # Find top kernels per file
    results = []
    for file in df['file'].unique():
        file_df = df[df['file'] == file]
        # Sort by total duration and get top kernel
        top_kernel = file_df.nlargest(1, 'duration_ns').iloc[0]
        
        results.append({
            'file': file,
            'top_kernel': top_kernel['kernel_name'][:80] + '...' if len(top_kernel['kernel_name']) > 80 else top_kernel['kernel_name'],
            'count': top_kernel['count'],
            'total_time_ms': top_kernel['duration_ns'] / 1_000_000,
            'avg_time_us': top_kernel['avg_ns'] / 1000
        })
    
    results_df = pd.DataFrame(results)
    return results_df, df

def create_nsys_analysis_report():
    """Create a comprehensive report answering all nsys profiling questions."""
    
    output_file = Path("../outputs/nsys_analysis_report.md")
    
    with open(output_file, 'w') as f:
        f.write("# Nsight Systems Profiling Analysis Report\n\n")
        
        # Question (a)
        f.write("## (a) What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?\n\n")
        f.write("**Deliverable:** A 1-2 sentence response.\n\n")
        
        # Extract forward pass timings
        timings_df = extract_forward_pass_timings()
        if timings_df is not None:
            f.write("### Forward Pass Timings\n\n")
            f.write(timings_df[['file', 'duration_ms', 'percentage']].to_markdown(index=False))
            f.write("\n\n**Answer:** [TO BE FILLED: Compare these timings with Python standard library measurements]\n\n")
        
        # Question (b)
        f.write("## (b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model?\n\n")
        f.write("**Deliverable:** A 1-2 sentence response.\n\n")
        
        # Extract kernel data
        top_kernels_df, all_kernels_df = extract_forward_pass_kernels()
        if top_kernels_df is not None:
            f.write("### Top CUDA Kernels in Forward Pass by Model Configuration\n\n")
            f.write(top_kernels_df.to_markdown(index=False))
            f.write("\n\n")
            
            # Also show overall top kernel
            if all_kernels_df is not None:
                top_overall = all_kernels_df.nlargest(1, 'duration_ns').iloc[0]
                f.write(f"**Answer:** The CUDA kernel that takes the most cumulative GPU time is `{top_overall['kernel_name'][:80]}...`, ")
                f.write(f"invoked {top_overall['count']} times during a single forward pass.\n\n")
        
        # Placeholder for remaining questions
        f.write("## (c) What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?\n\n")
        f.write("**Deliverable:** A 1-2 sentence response.\n\n")
        f.write("**Answer:** [TO BE IMPLEMENTED]\n\n")
        
        f.write("## (d) Profile running one complete training step with your implementation of AdamW. How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?\n\n")
        f.write("**Deliverable:** A 1-2 sentence response.\n\n")
        f.write("**Answer:** [TO BE IMPLEMENTED]\n\n")
        
        f.write("## (e) Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?\n\n")
        f.write("**Deliverable:** A 1-2 sentence response.\n\n")
        f.write("**Answer:** [TO BE IMPLEMENTED]\n\n")
    
    print(f"\nComprehensive analysis saved to {output_file}")

if __name__ == "__main__":
    create_nsys_analysis_report()