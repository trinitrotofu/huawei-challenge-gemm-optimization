import os
import argparse
import subprocess
import json
from loguru import logger


def get_compile_command() -> str:
    return "g++ -Ofast -march=native main.cpp -o main -lopenblas -lpthread"


def get_run_command(input_file: str, ours: bool) -> str:
    return f"./main {1 if ours else 0} < {input_file}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--compile", default=False, action="store_true", help="Compile the code")
    parser.add_argument("--gemm_inputs", required=True, type=str, help="Path to the gemm inputs")
    parser.add_argument("--num-trials", default=1, type=int, help="Number of trials to run")

    args = parser.parse_args()

    if not os.path.exists("main") or args.compile:
        logger.debug("Compiling...")
        os.system(get_compile_command())
        logger.debug("Compilation done.")

    logger.debug(f"Running benchmark on inputs {args.gemm_inputs}")

    results = {}
    for input_file in os.listdir(args.gemm_inputs):
        results[input_file] = {
            "ours": [],
            "baseline": []
        }

    logger.debug("Running benchmark...")
    for input_file in os.listdir(args.gemm_inputs):
        for ours in [True, False]:
            for trial in range(args.num_trials):
                logger.debug(f"Running benchmark on input {input_file} with ours={ours}, trial {trial}")
                cmd = get_run_command(os.path.join(args.gemm_inputs, input_file), ours)
                output = float(subprocess.check_output(cmd, shell=True).decode("utf-8"))
                results[input_file]["ours" if ours else "baseline"].append(output)
    logger.debug("Benchmark done.")

    logger.debug("Saving results...")
    with open("results.json", "w") as f:
        json.dump(results, f)
    logger.debug("Results saved.")
