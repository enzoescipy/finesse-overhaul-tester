import pandas as pd
import argparse

def filter_and_sample_mteb_models(input_csv, output_csv, num_samples, seed):
    """
    Filters the MTEB leaderboard data and samples a specified number of models.
    - Total Parameters < 8.0 B
    - Max Tokens >= 8190
    """
    print(f"Loading data from {input_csv}...")
    try:
        df = pd.read_csv(input_csv, index_col=0)
    except FileNotFoundError:
        print(f"Error: The file {input_csv} was not found.")
        return

    initial_count = len(df)
    print(f"Initial model count: {initial_count}")

    # --- Filtering Logic ---
    df['Total Parameters (B)'] = pd.to_numeric(df['Total Parameters (B)'], errors='coerce')
    df.dropna(subset=['Total Parameters (B)'], inplace=True)
    df = df[df['Total Parameters (B)'] < 8.0]
    print(f"Models remaining after parameter filter (< 8B): {len(df)}")

    df['Max Tokens'] = pd.to_numeric(df['Max Tokens'], errors='coerce')
    df.dropna(subset=['Max Tokens'], inplace=True)
    df = df[df['Max Tokens'] >= 8190]
    print(f"Models remaining after context window filter (>= 8190): {len(df)}")

    # --- Shuffling and Sampling Logic ---
    if len(df) < num_samples:
        print(f"Warning: Filtered models ({len(df)}) are fewer than requested samples ({num_samples}). Using all filtered models.")
        num_to_sample = len(df)
    else:
        num_to_sample = num_samples

    print(f"\nShuffling {len(df)} filtered models with random seed {seed}...")
    df_shuffled = df.sample(frac=1, random_state=seed)

    print(f"Taking the top {num_to_sample} models from the shuffled list...")
    df_sampled = df_shuffled.head(num_to_sample)

    # --- Save Output ---
    final_count = len(df_sampled)
    print(f"\nSaving {final_count} sampled models to {output_csv}...")
    df_sampled.to_csv(output_csv, index=True)
    print(f"Successfully created {output_csv}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter and sample MTEB models for the FINESSE benchmark.")
    parser.add_argument('--input-csv', type=str, default='tmpukyk5dfv.csv', help='Input MTEB CSV file.')
    parser.add_argument('--output-csv', type=str, default='filtered_models.csv', help='Output file for filtered and sampled models.')
    parser.add_argument('--num-samples', type=int, default=30, help='Number of models to sample after shuffling.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling to ensure reproducibility.')
    
    args = parser.parse_args()

    filter_and_sample_mteb_models(args.input_csv, args.output_csv, args.num_samples, args.seed)
