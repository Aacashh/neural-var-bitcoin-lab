import argparse
import json
from .data import fetch


def main():
    parser = argparse.ArgumentParser(description="Reproduce the Neural VaR Lab case study")
    parser.add_argument("command", choices=["fetch", "train", "audit"])
    args = parser.parse_args()
    if args.command == "fetch":
        print(json.dumps(fetch(), indent=2))
    else:
        from .experiment import train, audit
        print(train() if args.command == "train" else json.dumps(audit(), indent=2))


if __name__ == "__main__":
    main()
