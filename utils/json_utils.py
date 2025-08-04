import pandas as pd
import json
import os
from pathlib import Path

def read_json_file(file_path):
    """
    Read JSON file and convert to DataFrame
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        pd.DataFrame: DataFrame containing the JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        raise

def save_to_json(data, file_path):
    """
    Save DataFrame to JSON file
    
    Args:
        data (pd.DataFrame): DataFrame to save
        file_path (str): Path to the output JSON file
    """
    try:
        json_data = data.to_dict('records')
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {e}")
        raise

def read_file(file_path):
    """
    Read file (JSON or Excel) and convert to DataFrame
    
    Args:
        file_path (str): Path to the file (.json or .xlsx)
        
    Returns:
        pd.DataFrame: DataFrame containing the file data
    """
    if file_path.endswith('.json'):
        return read_json_file(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def save_file(data, file_path):
    """
    Save DataFrame to file (JSON or Excel)
    
    Args:
        data (pd.DataFrame): DataFrame to save
        file_path (str): Path to the output file (.json or .xlsx)
    """
    if file_path.endswith('.json'):
        save_to_json(data, file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        data.to_excel(file_path, index=False)
        print(f"Successfully saved to {file_path}")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def list_json_files(directory):
    """
    List all JSON files in a directory
    
    Args:
        directory (str): Directory path
        
    Returns:
        list: List of JSON file paths
    """
    json_files = []
    for file in Path(directory).glob('*.json'):
        json_files.append(str(file))
    return json_files

def validate_json_structure(file_path, required_columns=None):
    """
    Validate JSON file structure
    
    Args:
        file_path (str): Path to the JSON file
        required_columns (list): List of required column names
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        df = read_json_file(file_path)
        
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing required columns: {missing_columns}")
                return False
        
        print(f"File structure is valid. Shape: {df.shape}, Columns: {list(df.columns)}")
        return True
        
    except Exception as e:
        print(f"Error validating file structure: {e}")
        return False 