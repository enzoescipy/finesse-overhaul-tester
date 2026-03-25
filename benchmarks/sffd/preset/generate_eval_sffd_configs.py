import pandas as pd
import json
import re
from pathlib import Path

## you MUST check for the 'NaN' value after auto-generation.

def parse_model_name(markdown_link):
    """Extracts HF model name and creates a safe folder name."""
    match = re.search(r'\((.*?)\)', markdown_link)
    if not match:
        return None, None
    
    model_url = match.group(1)
    # Correctly parse model path like 'Alibaba-NLP/gte-Qwen2-7B-instruct'
    hf_path = '/'.join(model_url.split('/')[-2:])
    folder_name = hf_path.replace('/', '_')
    return hf_path, folder_name

def main():
    # Define paths relative to the project root
    project_root = Path(__file__).resolve().parents[3]
    csv_path = project_root / 'investigation-copy.csv'
    output_base_dir = project_root / 'benchmarks' /  'sffd' / 'model_eval'

    print(f"Project Root: {project_root}")
    print(f"Reading model list from: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: '{csv_path}' not found. Please ensure the file exists.")
        return

    for index, row in df.iterrows():
        model_markdown = row['Model']
        hf_path, folder_name = parse_model_name(model_markdown)

        if not hf_path:
            print(f"[SKIP] Could not parse model name from: {model_markdown}")
            continue

        print(f"\nProcessing {index + 1}/{len(df)}: {hf_path}")

        # 1. Create directory structure
        model_dir = output_base_dir / folder_name
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"  -> Created directory: {model_dir}")

        query_prefix = row.get('Query Prefix', 'NaN')
        passage_prefix = row.get('Document Prefix', 'NaN')
        pool_type = row.get('Pooling Method', 'NaN')
        is_instruct = row.get('Is Instructed', 'NaN')

        query_prefix = '' if pd.isna(query_prefix) else "NaN"
        passage_prefix = '' if pd.isna(passage_prefix) else "NaN"

        if is_instruct.lower() == 'yes':
            is_instruct = True
        elif is_instruct.lower() == 'no':
            is_instruct = False
        else:
            is_instruct = "NaN"

        query_prefix = query_prefix.strip("'")
        passage_prefix = passage_prefix.strip("'")
        pool_type = pool_type.strip("'")


        # 2. Prepare JSON configuration
        config_data = {
            "model_name": hf_path,
            "query_prefix": query_prefix,
            "passage_prefix": passage_prefix,
            "max_ctx": int(row['Max Tokens']),
            "is_instruct": is_instruct,
            "pool_type": pool_type,
            "batch_size": 1
        }

        # Set prefixes to empty string if they are NaN
        if pd.isna(config_data['query_prefix']): config_data['query_prefix'] = ''
        if pd.isna(config_data['passage_prefix']): config_data['passage_prefix'] = ''
        
        # 3. Create and save config.json
        config_output_path = model_dir / 'config.json'
        with open(config_output_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)
        print(f"  -> Generated: {config_output_path}")

    print("\n\n✨ All SFfD evaluation configurations generated successfully! ✨")

if __name__ == '__main__':
    main()
