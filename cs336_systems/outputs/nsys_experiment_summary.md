# Nsight Experiment Summary

I tried to vibe code my way out of looking very carefully at the nvidia nsight systems app. I failed. Claude opus 4 and o3 both generated scripts that failed to understand the nuance of the nvtx annotations I've made. So let us begin.

## First, how to run the benchmarking script: 

From the benchmarking script folder, simply
```bash 
uv run profile_nsight.sh
```
This will generate 20 nsys.rep files in the ../outputs/nsys folder. 

## Sifting through the nsys files

### Question (a)

The question asks:
> What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?

First, the csv file is contained in [here](csv/2025-07-28_table1.1.2.csv)

ctx 128, 256, 512, 1024
2.7B
71.903 ms