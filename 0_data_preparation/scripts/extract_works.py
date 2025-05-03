import argparse
import json
import os
from tqdm import tqdm
from random import seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help = "Jsonl of works, authors, etc.")
    parser.add_argument("--output", help = "Jsonl containing retrieved texts")
    args, _ = parser.parse_known_args()
    
    seed(42)

    all_works = []
    with open(args.input, "r") as input_file:
        for line in tqdm(input_file):
            author_info = json.loads(line)
            for name, pg_id in author_info["gb_works"].items():
                year = author_info["work_dates"][pg_id]
                temp = {"name": name, "pg_id": pg_id, "year": int(year)}
                all_works.append(temp)

    with open(args.output, "w") as output_file:
        for work in tqdm(all_works):
            output_file.write(json.dumps(work) + "\n")

                        