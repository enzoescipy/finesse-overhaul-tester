import pandas as pd
import sys

def filter_model_guideline(input_file='manually_selected_model.csv', output_file='filtered_model_guideline.csv'):
    """
    Filters manually_selected_model.csv to retain only essential columns:
    - Model
    - Total Parameters (B)
    - Max Tokens
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Select only the required columns
        required_columns = ['Model', 'Total Parameters (B)', 'Max Tokens']
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Filter the dataframe
        df_filtered = df[required_columns].copy()
        
        # Remove rows where all values are NaN (empty rows)
        df_filtered = df_filtered.dropna(how='all')
        
        # Save to new CSV
        df_filtered.to_csv(output_file, index=False)
        
        print(f"✅ Successfully filtered {len(df_filtered)} models")
        print(f"   Input:  {input_file}")
        print(f"   Output: {output_file}")
        print(f"\nPreview of filtered data:")
        print(df_filtered.head(10).to_string(index=False))
        
        return True
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == '__main__':
    # Allow command line arguments
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    else:
        input_file = 'model_guideline.csv'
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = 'filtered_model_guideline.csv'
    
    success = filter_model_guideline(input_file, output_file)
    sys.exit(0 if success else 1)
