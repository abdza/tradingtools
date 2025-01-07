import pandas as pd
import argparse


def split_csv(input_file, output_file1, output_file2, split_ratio=0.5):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Calculate the split point
    split_point = int(len(df) * split_ratio)

    # Split the dataframe
    df1 = df[:split_point]
    df2 = df[split_point:]

    # Save to separate files
    df1.to_csv(output_file1, index=False)
    df2.to_csv(output_file2, index=False)

    print(
        f"Split complete:\n{output_file1}: {len(df1)} rows\n{output_file2}: {len(df2)} rows"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a CSV file into two parts")
    parser.add_argument("input_file", help="Input CSV file path")
    parser.add_argument("output_file1", help="First output CSV file path")
    parser.add_argument("output_file2", help="Second output CSV file path")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Split ratio (between 0 and 1, default: 0.5)",
    )

    args = parser.parse_args()

    if not 0 < args.ratio < 1:
        parser.error("Ratio must be between 0 and 1")

    split_csv(args.input_file, args.output_file1, args.output_file2, args.ratio)
