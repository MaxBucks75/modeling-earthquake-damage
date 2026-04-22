import argparse
from src.experiments.experiments import run_experiment_log_regression

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, default="logreg")

args = parser.parse_args()

if args.experiment == "logreg":
    run_experiment_log_regression()
else:
    print("Unknown experiment")