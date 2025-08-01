#!/usr/bin/env python3
import subprocess
import os
import re
import json
import pandas as pd
from pathlib import Path
import argparse

def run_nsys_stats(nsys_file, report_type):
    """Run nsys stats on a file and return the output."""
    cmd = ["nsys", "stats", "--report", report_type, "--format", "csv", str(nsys_file)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running nsys stats on {nsys_file}: {e}")
        print(f"stderr: {e.stderr}")
        return None

def parse_csv_output(csv_text):
    """Parse CSV output from nsys stats."""
    if not csv_text:
        return None
    
    # Split into lines and find where the actual CSV starts
    lines = csv_text.strip().split('\n')
    
    # Find the header line (contains "Time" or similar)
    header_idx = None
    for i, line in enumerate(lines):
        if ',' in line and ('Time' in line or 'Duration' in line or 'Name' in line):
            header_idx = i
            break
    
    if header_idx is None:
        return None
    
    # Create a DataFrame from the CSV data
    csv_lines = '\n'.join(lines[header_idx:])
    from io import StringIO
    try:
        df = pd.read_csv(StringIO(csv_lines))
        return df
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return None

def analyze_nvtx_ranges(nsys_file):
    """Analyze NVTX ranges to get timing for different phases."""
    csv_output = run_nsys_stats(nsys_file, "nvtx_sum")
    df = parse_csv_output(csv_output)
    
    if df is None:
        return {}
    
    results = {}
    
    # Look for specific NVTX ranges
    ranges_of_interest = ['warmup', 'forward_pass', 'model_eval', 'loss', 'backward', 'train_step', 'optimizer_step']
    
    for range_name in ranges_of_interest:
        mask = df['Name'].str.contains(range_name, case=False, na=False)
        if mask.any():
            row = df[mask].iloc[0]
            # Extract timing - column names might vary
            time_col = None
            for col in ['Total Time (ns)', 'Duration (ns)', 'Time (ns)']:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col:
                results[range_name] = {
                    'time_ns': row[time_col],
                    'time_ms': row[time_col] / 1e6,
                    'count': row.get('Count', 1)
                }
    
    return results

def analyze_cuda_kernels(nsys_file):
    """Analyze CUDA kernel statistics."""
    csv_output = run_nsys_stats(nsys_file, "cuda_gpu_sum")
    df = parse_csv_output(csv_output)
    
    if df is None:
        return {}
    
    # Sort by total time to find most expensive kernels
    time_col = None
    for col in ['Total Time (ns)', 'Duration (ns)', 'Time (ns)']:
        if col in df.columns:
            time_col = col
            break
    
    if not time_col:
        return {}
    
    df_sorted = df.sort_values(by=time_col, ascending=False)
    
    # Categorize kernels
    results = {
        'top_kernels': [],
        'matrix_multiply_kernels': [],
        'softmax_kernels': [],
        'other_significant_kernels': []
    }
    
    # Get top 10 kernels
    for idx, row in df_sorted.head(10).iterrows():
        kernel_info = {
            'name': row.get('Name', row.get('Kernel', 'Unknown')),
            'time_ms': row[time_col] / 1e6,
            'count': row.get('Count', row.get('Instances', 1)),
            'avg_time_us': row.get('Avg (ns)', row[time_col] / row.get('Count', 1)) / 1e3
        }
        results['top_kernels'].append(kernel_info)
        
        # Categorize by type
        kernel_name = kernel_info['name'].lower()
        if any(pattern in kernel_name for pattern in ['gemm', 'gemv', 'trsm', 'sgemm', 'dgemm', 'hgemm', 'wmma', 'tensor_core']):
            results['matrix_multiply_kernels'].append(kernel_info)
        elif 'softmax' in kernel_name:
            results['softmax_kernels'].append(kernel_info)
        elif kernel_info['time_ms'] > 1.0:  # Significant if > 1ms total time
            results['other_significant_kernels'].append(kernel_info)
    
    # Calculate total time spent in matrix multiplies
    total_mm_time = sum(k['time_ms'] for k in results['matrix_multiply_kernels'])
    total_time = df[time_col].sum() / 1e6
    results['matrix_multiply_fraction'] = total_mm_time / total_time if total_time > 0 else 0
    
    return results

def analyze_attention_kernels(nsys_file):
    """Specifically analyze attention-related kernels."""
    csv_output = run_nsys_stats(nsys_file, "cuda_gpu_trace")
    df = parse_csv_output(csv_output)
    
    if df is None:
        return {}
    
    # Look for attention-specific patterns
    attention_patterns = ['attention', 'attn', 'qkv', 'query', 'key', 'value', 'scaled_dot']
    softmax_patterns = ['softmax']
    
    results = {
        'attention_kernels': [],
        'attention_softmax': [],
        'attention_matmul': []
    }
    
    for idx, row in df.iterrows():
        kernel_name = str(row.get('Kernel', '')).lower()
        
        # Check if this is an attention-related kernel
        if any(pattern in kernel_name for pattern in attention_patterns):
            kernel_info = {
                'name': row.get('Kernel', 'Unknown'),
                'duration_us': row.get('Duration (ns)', 0) / 1e3
            }
            
            if any(pattern in kernel_name for pattern in softmax_patterns):
                results['attention_softmax'].append(kernel_info)
            elif any(pattern in kernel_name for pattern in ['gemm', 'gemv', 'mm']):
                results['attention_matmul'].append(kernel_info)
            
            results['attention_kernels'].append(kernel_info)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze Nsight Systems profiles for assignment questions')
    parser.add_argument('--dir', default='../outputs/nsys', help='Directory containing .nsys files')
    parser.add_argument('--output', default='nsight_analysis_results.json', help='Output JSON file')
    args = parser.parse_args()
    
    nsys_dir = Path(args.dir)
    if not nsys_dir.exists():
        print(f"Directory {nsys_dir} not found!")
        return
    
    # Find all .nsys files
    nsys_files = list(nsys_dir.glob('*.nsys'))
    
    if not nsys_files:
        print(f"No .nsys files found in {nsys_dir}")
        return
    
    all_results = {}
    
    for nsys_file in nsys_files:
        print(f"\nAnalyzing {nsys_file.name}...")
        
        # Extract model size and context length from filename
        # Expected format: {size}_ctx{context}.nsys
        match = re.match(r'(.+?)_ctx(\d+)\.nsys', nsys_file.name)
        if match:
            model_size = match.group(1)
            context_length = int(match.group(2))
        else:
            model_size = nsys_file.stem
            context_length = 'unknown'
        
        results = {
            'file': nsys_file.name,
            'model_size': model_size,
            'context_length': context_length,
            'nvtx_ranges': analyze_nvtx_ranges(nsys_file),
            'cuda_kernels': analyze_cuda_kernels(nsys_file),
            'attention_analysis': analyze_attention_kernels(nsys_file)
        }
        
        all_results[nsys_file.name] = results
        
        # Print summary for this file
        print(f"Model: {model_size}, Context: {context_length}")
        
        if results['nvtx_ranges']:
            print("\nNVTX Range Timings:")
            for range_name, timing in results['nvtx_ranges'].items():
                print(f"  {range_name}: {timing['time_ms']:.2f} ms (count: {timing['count']})")
        
        if results['cuda_kernels']:
            print("\nTop CUDA Kernels:")
            for i, kernel in enumerate(results['cuda_kernels']['top_kernels'][:5]):
                print(f"  {i+1}. {kernel['name'][:50]}...")
                print(f"     Total: {kernel['time_ms']:.2f} ms, Count: {kernel['count']}, Avg: {kernel['avg_time_us']:.2f} us")
            
            print(f"\nMatrix multiply fraction: {results['cuda_kernels']['matrix_multiply_fraction']:.2%}")
    
    # Save results to JSON
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Generate answer template
    print("\n" + "="*80)
    print("ANSWERS TO ASSIGNMENT QUESTIONS:")
    print("="*80)
    
    # Try to provide answers based on the analysis
    if all_results:
        # Pick a representative file for answers
        sample_file = list(all_results.values())[0]
        
        print("\n(a) Forward Pass Timing:")
        if 'forward_pass' in sample_file['nvtx_ranges']:
            print(f"    Total forward pass time: {sample_file['nvtx_ranges']['forward_pass']['time_ms']:.2f} ms")
            print("    TODO: Compare this with your Python timeit benchmarking results")
        else:
            print("    No forward_pass NVTX range found. Check if NVTX annotations are correct.")
        
        print("\n(b) Most Time-Consuming CUDA Kernel:")
        if sample_file['cuda_kernels']['top_kernels']:
            top_kernel = sample_file['cuda_kernels']['top_kernels'][0]
            print(f"    Kernel: {top_kernel['name']}")
            print(f"    Time: {top_kernel['time_ms']:.2f} ms")
            print(f"    Invocations: {top_kernel['count']}")
            print("    TODO: Check if this is the same for forward+backward pass")
        
        print("\n(c) Non-Matrix-Multiply Kernels:")
        if sample_file['cuda_kernels']['other_significant_kernels']:
            print("    Significant non-GEMM kernels:")
            for kernel in sample_file['cuda_kernels']['other_significant_kernels'][:3]:
                print(f"    - {kernel['name']}: {kernel['time_ms']:.2f} ms")
        
        print("\n(d) Matrix Multiplication Fraction:")
        if 'matrix_multiply_fraction' in sample_file['cuda_kernels']:
            print(f"    Forward pass only: {sample_file['cuda_kernels']['matrix_multiply_fraction']:.2%}")
            print("    TODO: Compare with full training step (forward+backward+optimizer)")
        
        print("\n(e) Softmax vs Matrix Multiplication in Attention:")
        print("    TODO: Filter kernels by NVTX ranges to isolate attention layer")
        print("    TODO: Compare softmax kernel time vs attention matmul time")

if __name__ == "__main__":
    main()