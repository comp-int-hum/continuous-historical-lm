import argparse
import json
import os
from tqdm import tqdm
from random import seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Jsonl of works, etc.")
    parser.add_argument("--output", help="Jsonl containing retrieved texts")
    parser.add_argument("--start_year", type=int, default=1750, help="Start year for filtering works")
    parser.add_argument("--end_year", type=int, default=1940, help="End year for filtering works")
    args, _ = parser.parse_known_args()
    
    all_works = []
    with open(args.input, "r") as input_file:
        for line in tqdm(input_file):
            work = json.loads(line)
            year = work.get("year", None)
            if year is not None and args.start_year <= year < args.end_year:
                all_works.append(work)

    with open(args.output, "w") as output_file:
        for work in all_works:
            output_file.write(json.dumps(work) + "\n")

                        