import json
from train import main
from argparse import ArgumentParser


def run_instance_us_nz():
    config = {"train_countries": ["NZ"], "test_countries": ["NZ"]}
    main(config["train_countries"], config["test_countries"], True)


def run_us_nz():
    config = {"train_countries": ["US"], "test_countries": ["NZ"]}
    main(config["train_countries"], config["test_countries"])


def run_nz_nz():
    config = {"train_countries": ["NZ"], "test_countries": ["NZ"]}
    main(config["train_countries"], config["test_countries"])


methods = {
    "run_instance_us_nz": run_instance_us_nz,
    "run_us_nz": run_us_nz,
    "run_nz_nz": run_nz_nz,
}

parser = ArgumentParser()
parser.add_argument(
    "-m",
    "--method",
    dest="method",
    help=
    "which method do you want to run? 'run_us_nz', 'run_nz_nz', 'run_instance_us_nz'"
)

args = parser.parse_args()
methods.get(args.method)()
