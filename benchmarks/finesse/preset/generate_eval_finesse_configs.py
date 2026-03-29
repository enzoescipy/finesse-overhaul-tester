import pandas as pd
import yaml
import os
import re
from pathlib import Path

## you MUST check for the 'NaN' value after auto-generation.

def parse_model_name(markdown_link):
    """Extracts HF model name and creates a safe folder name."""
    match = re.search(r'\((.*?)\)', markdown_link)
    if not match:
        return None, None
    
    hf_name = match.group(1).split('/')[-1]
    model_path = match.group(1).split('/')[-2] + '/' + hf_name
    folder_name = model_path.replace('/', '_')
    return model_path, folder_name

def main():
    # Define paths relative to the project root
    project_root = Path(__file__).resolve().parents[3]
    csv_path = project_root / 'investigation-copy.csv'
    srs_template_path = project_root / 'benchmarks' / 'finesse' / 'preset' / 'srs.yaml'
    rss_template_path = project_root / 'benchmarks' / 'finesse' / 'preset' / 'rss.yaml'
    output_base_dir = project_root / 'benchmarks' / 'finesse' / 'model_eval'

    print(f"Project Root: {project_root}")
    print(f"Reading model list from: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: '{csv_path}' not found. Please ensure the file exists.")
        return

    with open(srs_template_path, 'r') as f:
        srs_template = yaml.safe_load(f)
    with open(rss_template_path, 'r') as f:
        rss_template = yaml.safe_load(f)

    for index, row in df.iterrows():
        model_markdown = row['Model']
        hf_path, folder_name = parse_model_name(model_markdown)

        if not hf_path:
            print(f"[SKIP] Could not parse model name from: {model_markdown}")
            continue

        print(f"\nProcessing {index + 1}/{len(df)}: {hf_path}")

        # 1. Create directory structure
        model_dir = output_base_dir / folder_name
        srs_dir = model_dir / 'srs'
        rss_dir = model_dir / 'rss'
        srs_dir.mkdir(parents=True, exist_ok=True)
        rss_dir.mkdir(parents=True, exist_ok=True)
        print(f"  -> Created directory: {model_dir}")

        # 2. Prepare configurations
        max_tokens = int(row['Max Tokens'])
        prefix = row.get('Document Prefix', 'NaN')
        prefix = '' if pd.isna(prefix) else 'NaN' # Handle potential NaN
        pool_type = row.get('Pooling Method', 'NaN')

        # 3. Create and save srs.yaml
        srs_config = srs_template.copy()
        srs_config['models']['native_embedder']['name'] = hf_path
        srs_config['models']['native_embedder']['max_context_length'] = max_tokens
        srs_config['models']['native_embedder']['prefix'] = prefix
        srs_config['models']['native_embedder']['pool_type'] = pool_type
        
        srs_output_path = srs_dir / 'srs.yaml'
        with open(srs_output_path, 'w') as f:
            yaml.dump(srs_config, f, sort_keys=False, indent=2)
        print(f"  -> Generated: {srs_output_path}")

        # 4. Create and save rss.yaml
        rss_config = rss_template.copy()
        rss_config['models']['native_embedder']['name'] = hf_path
        rss_config['models']['native_embedder']['max_context_length'] = max_tokens
        rss_config['models']['native_embedder']['prefix'] = prefix
        rss_config['models']['native_embedder']['pool_type'] = pool_type

        rss_output_path = rss_dir / 'rss.yaml'
        with open(rss_output_path, 'w') as f:
            yaml.dump(rss_config, f, sort_keys=False, indent=2)
        print(f"  -> Generated: {rss_output_path}")

    print("\n\n✨ All evaluation configurations generated successfully! ✨")

if __name__ == '__main__':
    main()
