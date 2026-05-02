import argparse
from solution import run_inference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    args = parser.parse_args()

    run_inference(args.test_dir)

if __name__ == "__main__":
    main()