import os
import json
import csv
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--results", type=str, default="results.json", help="Path to results.json file")
    parser.add_argument("--cpu-name", required=True, type=str, default="cpu", help="Name of CPU for titles")

    args = parser.parse_args()

    if not os.path.exists(args.results):
        logger.error(f"Could not find results file at {args.results}")
        exit(1)
    
    with open(args.results, "r") as f:
        results = json.load(f)
    
    n = 10
    speedups = []
    for case in range(n):
        filename = f"{case}.txt"
        baseline_time = sum(results[filename]["baseline"]) / len(results[filename]["baseline"])
        our_time = sum(results[filename]["ours"]) / len(results[filename]["ours"])
        speedup = (baseline_time - our_time) / baseline_time * 100
        speedups.append(speedup)

    sns.set_theme(style="whitegrid")
    sns.barplot(x=list(range(n)), y=speedups)
    plt.xlabel("Input file number")
    plt.ylabel("% Improvement")
    plt.title(f"Benchmark Results for {args.cpu_name}")
    plt.savefig("results.png")

    # Make CSV file for results = {n.txt: {baseline: [times], ours: [times]}}
    with open("results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["File", "Baseline", "Ours", "Speedup"])
        for case in range(n):
            filename = f"{case}.txt"
            baseline_time = sum(results[filename]["baseline"]) / len(results[filename]["baseline"])
            our_time = sum(results[filename]["ours"]) / len(results[filename]["ours"])
            speedup = (baseline_time - our_time) / baseline_time * 100
            writer.writerow([filename, baseline_time, our_time, speedup])
