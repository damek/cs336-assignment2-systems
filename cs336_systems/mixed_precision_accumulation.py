import torch
import os

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Capture outputs
outputs = []

def capture_print(msg):
    print(msg)
    outputs.append(msg)

capture_print("float32 accumulator + float32 values:")
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float32)
capture_print(str(s))

capture_print("\nfloat16 accumulator + float16 values:")
s = torch.tensor(0,dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
capture_print(str(s))

capture_print("\nfloat32 accumulator + float16 values (implicit upcast):")
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
capture_print(str(s))

capture_print("\nfloat32 accumulator + float16 values (explicit upcast):")
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01,dtype=torch.float16)
    s += x.type(torch.float32)
capture_print(str(s))

# Write to markdown file
with open("outputs/mixed_precision_accumulation.md", "w") as f:
    f.write("# Mixed Precision Accumulation Results\n\n")
    f.write("## Code\n\n")
    f.write("```python\n")
    f.write("""import torch

print("float32 accumulator + float32 values:")
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float32)
print(s)

print("float16 accumulator + float16 values:")
s = torch.tensor(0,dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)

print("float32 accumulator + float16 values (implicit upcast):")
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)

print("float32 accumulator + float16 values (explicit upcast):")
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01,dtype=torch.float16)
    s += x.type(torch.float32)
print(s)
""")
    f.write("```\n\n")
    f.write("## Output\n\n")
    f.write("```\n")
    f.write("\n".join(outputs))
    f.write("\n```\n")

print(f"\nResults written to outputs/mixed_precision_accumulation.md")