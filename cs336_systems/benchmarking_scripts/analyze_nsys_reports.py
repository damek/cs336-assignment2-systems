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

if __name__ == "__main__":
    extract_forward_pass_timings()