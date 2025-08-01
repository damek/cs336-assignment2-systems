#!/usr/bin/env python3
"""
Detailed Nsight analysis script for answering assignment questions.
This script provides specific analysis for each question in the assignment.
"""

import subprocess
import pandas as pd
import json
from pathlib import Path
import re
from collections import defaultdict
import argparse

class NsightAnalyzer:
    def __init__(self, nsys_file):
        self.nsys_file = Path(nsys_file)
        self.results = {}
        
    def run_nsys_report(self, report_type, output_format="csv"):
        """Run nsys stats with specific report type."""
        cmd = ["nsys", "stats", "--report", report_type, 
               "--format", output_format, str(self.nsys_file)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            return None
    
    def parse_csv(self, csv_text):
        """Parse CSV output from nsys stats."""
        if not csv_text:
            return None
            
        lines = csv_text.strip().split('\n')
        
        # Find header line
        header_idx = None
        for i, line in enumerate(lines):
            if ',' in line and any(col in line for col in ['Time', 'Duration', 'Name', 'Kernel']):
                header_idx = i
                break
        
        if header_idx is None:
            return None
            
        from io import StringIO
        csv_data = '\n'.join(lines[header_idx:])
        
        try:
            return pd.read_csv(StringIO(csv_data))
        except:
            return None
    
    def analyze_forward_pass_timing(self):
        """Answer (a): Get forward pass timing from NVTX ranges."""
        nvtx_output = self.run_nsys_report("nvtx_sum")
        df = self.parse_csv(nvtx_output)
        
        if df is None:
            return None
        
        # Look for forward_pass NVTX range
        forward_mask = df['Name'].str.contains('forward_pass', case=False, na=False)
        
        if forward_mask.any():
            row = df[forward_mask].iloc[0]
            time_col = next((col for col in ['Total Time (ns)', 'Duration (ns)', 'Time (ns)'] 
                            if col in df.columns), None)
            
            if time_col:
                time_ms = row[time_col] / 1e6
                return {
                    'forward_pass_time_ms': time_ms,
                    'count': row.get('Count', 1)
                }
        
        return None
    
    def analyze_top_cuda_kernels(self):
        """Answer (b): Find most time-consuming CUDA kernels."""
        # Get CUDA kernel summary
        cuda_output = self.run_nsys_report("cuda_gpu_sum")
        df = self.parse_csv(cuda_output)
        
        if df is None:
            return None
        
        # Find time column
        time_col = next((col for col in ['Total Time (ns)', 'Duration (ns)', 'Time (ns)'] 
                        if col in df.columns), None)
        
        if not time_col:
            return None
        
        # Sort by total time
        df_sorted = df.sort_values(by=time_col, ascending=False)
        
        # Get top kernel
        if not df_sorted.empty:
            top_kernel = df_sorted.iloc[0]
            kernel_name = top_kernel.get('Name', top_kernel.get('Kernel', 'Unknown'))
            
            return {
                'top_kernel_name': kernel_name,
                'total_time_ms': top_kernel[time_col] / 1e6,
                'count': top_kernel.get('Count', top_kernel.get('Instances', 1)),
                'avg_time_us': top_kernel.get('Avg (ns)', top_kernel[time_col] / top_kernel.get('Count', 1)) / 1e3
            }
        
        return None
    
    def analyze_kernel_types(self):
        """Answer (c): Categorize kernels by type."""
        cuda_output = self.run_nsys_report("cuda_gpu_sum")
        df = self.parse_csv(cuda_output)
        
        if df is None:
            return None
        
        time_col = next((col for col in ['Total Time (ns)', 'Duration (ns)', 'Time (ns)'] 
                        if col in df.columns), None)
        
        if not time_col:
            return None
        
        # Categorize kernels
        categories = {
            'matrix_multiply': [],
            'softmax': [],
            'elementwise': [],
            'reduction': [],
            'memory': [],
            'other': []
        }
        
        # Pattern matching for kernel types
        patterns = {
            'matrix_multiply': ['gemm', 'gemv', 'trsm', 'sgemm', 'dgemm', 'hgemm', 'wmma', 'tensor_core', 'cublas'],
            'softmax': ['softmax'],
            'elementwise': ['elementwise', 'binary', 'unary', 'activation', 'add', 'mul', 'div'],
            'reduction': ['reduce', 'reduction', 'sum', 'mean'],
            'memory': ['memcpy', 'memset', 'copy', 'fill']
        }
        
        for idx, row in df.iterrows():
            kernel_name = str(row.get('Name', row.get('Kernel', ''))).lower()
            time_ms = row[time_col] / 1e6
            
            # Skip very small kernels
            if time_ms < 0.01:
                continue
            
            kernel_info = {
                'name': row.get('Name', row.get('Kernel', 'Unknown')),
                'time_ms': time_ms,
                'count': row.get('Count', 1)
            }
            
            # Categorize
            categorized = False
            for category, pattern_list in patterns.items():
                if any(pattern in kernel_name for pattern in pattern_list):
                    categories[category].append(kernel_info)
                    categorized = True
                    break
            
            if not categorized:
                categories['other'].append(kernel_info)
        
        # Sort each category by time
        for category in categories:
            categories[category].sort(key=lambda x: x['time_ms'], reverse=True)
        
        return categories
    
    def analyze_training_step_breakdown(self):
        """Answer (d): Analyze full training step breakdown."""
        nvtx_output = self.run_nsys_report("nvtx_sum")
        df = self.parse_csv(nvtx_output)
        
        if df is None:
            return None
        
        # Get timings for different phases
        phases = ['forward_pass', 'backward', 'optimizer_step', 'train_step']
        timings = {}
        
        time_col = next((col for col in ['Total Time (ns)', 'Duration (ns)', 'Time (ns)'] 
                        if col in df.columns), None)
        
        if not time_col:
            return None
        
        for phase in phases:
            mask = df['Name'].str.contains(phase, case=False, na=False)
            if mask.any():
                row = df[mask].iloc[0]
                timings[phase] = row[time_col] / 1e6
        
        return timings
    
    def analyze_attention_operations(self):
        """Answer (e): Analyze attention-specific operations."""
        # This requires correlating NVTX ranges with kernel calls
        # First, get kernels within attention NVTX ranges
        
        # Get all kernel traces
        trace_output = self.run_nsys_report("cuda_gpu_trace")
        df_trace = self.parse_csv(trace_output)
        
        if df_trace is None:
            return None
        
        # Look for attention-related kernels
        attention_patterns = ['attention', 'attn', 'scaled_dot', 'qkv']
        softmax_time = 0
        matmul_time = 0
        
        for idx, row in df_trace.iterrows():
            kernel_name = str(row.get('Kernel', '')).lower()
            duration_ns = row.get('Duration (ns)', 0)
            
            # Check if this is attention-related
            if any(pattern in kernel_name for pattern in attention_patterns):
                if 'softmax' in kernel_name:
                    softmax_time += duration_ns
                elif any(pattern in kernel_name for pattern in ['gemm', 'gemv', 'mm']):
                    matmul_time += duration_ns
        
        return {
            'attention_softmax_ms': softmax_time / 1e6,
            'attention_matmul_ms': matmul_time / 1e6,
            'ratio': (softmax_time / matmul_time) if matmul_time > 0 else 0
        }
    
    def generate_report(self):
        """Generate a comprehensive report answering all questions."""
        print(f"\nAnalyzing {self.nsys_file.name}")
        print("=" * 80)
        
        # Extract model info from filename
        match = re.match(r'(.+?)_ctx(\d+)\.nsys', self.nsys_file.name)
        if match:
            model_size = match.group(1)
            context_length = int(match.group(2))
            print(f"Model: {model_size}, Context Length: {context_length}")
        
        # Question (a)
        print("\n(a) Forward Pass Timing:")
        forward_timing = self.analyze_forward_pass_timing()
        if forward_timing:
            print(f"    Total forward pass time: {forward_timing['forward_pass_time_ms']:.2f} ms")
            print(f"    Number of iterations: {forward_timing['count']}")
            print("    → Compare this with your Python timeit results")
        else:
            print("    ⚠️  No forward_pass NVTX range found")
        
        # Question (b)
        print("\n(b) Most Time-Consuming CUDA Kernel:")
        top_kernel = self.analyze_top_cuda_kernels()
        if top_kernel:
            print(f"    Kernel: {top_kernel['top_kernel_name']}")
            print(f"    Total time: {top_kernel['total_time_ms']:.2f} ms")
            print(f"    Invocations: {top_kernel['count']}")
            print(f"    Average time: {top_kernel['avg_time_us']:.2f} μs")
        else:
            print("    ⚠️  Could not analyze CUDA kernels")
        
        # Question (c)
        print("\n(c) Non-Matrix-Multiply Kernels:")
        kernel_types = self.analyze_kernel_types()
        if kernel_types:
            print("    Significant non-GEMM kernels:")
            for category in ['softmax', 'elementwise', 'reduction', 'other']:
                if kernel_types[category]:
                    print(f"\n    {category.upper()}:")
                    for kernel in kernel_types[category][:3]:
                        print(f"      - {kernel['name'][:50]}... : {kernel['time_ms']:.2f} ms")
        
        # Question (d)
        print("\n(d) Training Step Breakdown:")
        training_breakdown = self.analyze_training_step_breakdown()
        if training_breakdown:
            if 'forward_pass' in training_breakdown and 'train_step' in training_breakdown:
                forward_only = training_breakdown['forward_pass']
                full_step = training_breakdown['train_step']
                print(f"    Forward pass only: {forward_only:.2f} ms")
                print(f"    Full training step: {full_step:.2f} ms")
                print(f"    Forward fraction: {(forward_only/full_step)*100:.1f}%")
            
            # Analyze kernel type distribution
            if kernel_types:
                total_mm_time = sum(k['time_ms'] for k in kernel_types['matrix_multiply'])
                total_time = sum(sum(k['time_ms'] for k in kernels) for kernels in kernel_types.values())
                if total_time > 0:
                    print(f"    Matrix multiply fraction: {(total_mm_time/total_time)*100:.1f}%")
        
        # Question (e)
        print("\n(e) Attention Layer Analysis:")
        attention_analysis = self.analyze_attention_operations()
        if attention_analysis:
            print(f"    Softmax time in attention: {attention_analysis['attention_softmax_ms']:.2f} ms")
            print(f"    MatMul time in attention: {attention_analysis['attention_matmul_ms']:.2f} ms")
            print(f"    Softmax/MatMul ratio: {attention_analysis['ratio']:.3f}")
            print("    → Compare this ratio with the FLOP ratio")
        
        return {
            'forward_timing': forward_timing,
            'top_kernel': top_kernel,
            'kernel_types': kernel_types,
            'training_breakdown': training_breakdown,
            'attention_analysis': attention_analysis
        }

def main():
    parser = argparse.ArgumentParser(description='Detailed Nsight analysis for assignment')
    parser.add_argument('nsys_file', help='Path to .nsys file to analyze')
    parser.add_argument('--json', help='Output results to JSON file')
    args = parser.parse_args()
    
    analyzer = NsightAnalyzer(args.nsys_file)
    results = analyzer.generate_report()
    
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {args.json}")

if __name__ == "__main__":
    main()