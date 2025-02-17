import pandas as pd
import os
import json

def flatten_data_to_series(data):
    flattened = {}
    
    # Process each level in the data
    for level_key in data:
        if level_key == 'params':
            continue  # Handle params separately
        level_entry = data[level_key]
        
        # Identify samples (keys whose values are dictionaries)
        samples = [k for k in level_entry if isinstance(level_entry[k], dict)]
        
        # Process each sample's metrics
        for sample_key in samples:
            sample_data = level_entry[sample_key]
            for metric in sample_data:
                key = f"{level_key}_{sample_key}_{metric}"
                flattened[key] = sample_data[metric]
        
        # Process total metrics for the level
        totals = [k for k in level_entry if k not in samples]
        for total_key in totals:
            key = f"{level_key}_{total_key}"
            flattened[key] = level_entry[total_key]
    
    # Process hyperparameters
    params = data.get('params', {})
    for param_key, param_value in params.items():
        key = f"params_{param_key}"
        flattened[key] = param_value
    
    return pd.Series(flattened)

def load_and_flatten_data(directory, filter_prefix='HPTuning'):
    all_data = []
    
    # Collect all JSON files
    json_files = [f for f in os.listdir(directory) if f.startswith(filter_prefix) and f.endswith('.json')]
    
    for file in json_files:
        file_path = os.path.join(directory, file)
        
        # Load the JSON data from the file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Flatten the data and append to the list
        flattened_data = flatten_data_to_series(data)
        # Add the file name as a column
        flattened_data['file_name'] = file
        all_data.append(flattened_data)
    
    # Convert the list of series to a DataFrame
    df = pd.DataFrame(all_data)
    return df

        

if __name__ == '__main__':
    directory = './hyperparameters2'  # Current directory
    df = load_and_flatten_data(directory)
    output_dir = directory
    os.makedirs(output_dir, exist_ok=True)

    # Print the DataFrame
    print(df)
    
    # Save the DataFrame to an Excel
    df.to_excel(os.path.join(output_dir, 'compiled_data.xlsx'))
    
    # Find the 3 best trials based on the sum of MCC scores on some level
    level = 8
    sum_mcc_col = f"{level}_total_mcc"
    best_trials = df.sort_values(sum_mcc_col, ascending=False).head(3)

    # Save the best trials to an excel file
    best_trials.to_excel(os.path.join(output_dir, 'best_trials.xlsx'), index=False)
    
    # Find the best trial where params_filter_raw_sinogram_with_a is 0
    df_no_filter = df[df['params_filter_raw_sinogram_with_a'] == 0]
    best_no_filter = df_no_filter.sort_values(sum_mcc_col, ascending=False).head(1)
    best_no_filter.to_excel(os.path.join(output_dir, 'best_no_filter.xlsx'), index=False)
    
    # Find the best trial where params_use_tv_reg is 0
    df_no_tv = df[df['params_use_tv_reg'] == 0]
    best_no_tv = df_no_tv.sort_values(sum_mcc_col, ascending=False).head(1)
    best_no_tv.to_excel(os.path.join(output_dir, 'best_no_tv.xlsx'), index=False)
    
    # Find the best trial where params_use_no_model is True
    df_no_model = df[df['params_use_no_model'] == True]
    best_no_model = df_no_model.sort_values(sum_mcc_col, ascending=False).head(1)
    best_no_model.to_excel(os.path.join(output_dir, 'best_no_DIP.xlsx'), index=False)
    
    # Find the best trial where params_use_autoencoder_reg is 0
    df_no_autoencoder = df[df['params_use_autoencoder_reg'] == 0]
    best_no_autoencoder = df_no_autoencoder.sort_values(sum_mcc_col, ascending=False).head(1)
    best_no_autoencoder.to_excel(os.path.join(output_dir, 'best_no_PSR.xlsx'), index=False)
    
    
    