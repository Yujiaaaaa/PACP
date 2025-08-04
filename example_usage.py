#!/usr/bin/env python3
"""
Example usage script for the Prompt Dissection & Perturbation Pipeline
Demonstrates how to use JSON format data throughout the entire workflow
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from utils.json_utils import read_json_file, save_to_json, list_json_files, validate_json_structure

def main():
    """Main example workflow"""
    
    print("=== Prompt Dissection & Perturbation Pipeline - JSON Example ===\n")
    
    # 1. List available datasets
    print("1. Available datasets:")
    datasets_dir = "Datasets"
    if os.path.exists(datasets_dir):
        json_files = list_json_files(datasets_dir)
        for file in json_files:
            print(f"   - {file}")
    else:
        print(f"   Datasets directory '{datasets_dir}' not found")
        return
    
    # 2. Validate dataset structure
    print("\n2. Validating dataset structure:")
    sample_dataset = "Datasets/Leetcode-PA.json"
    if os.path.exists(sample_dataset):
        is_valid = validate_json_structure(sample_dataset, required_columns=['context'])
        if is_valid:
            print("   ✓ Dataset structure is valid")
        else:
            print("   ✗ Dataset structure is invalid")
            return
    else:
        print(f"   Sample dataset '{sample_dataset}' not found")
        return
    
    # 3. Read and preview dataset
    print("\n3. Reading and previewing dataset:")
    try:
        df = read_json_file(sample_dataset)
        print(f"   Dataset shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   First row preview:")
        if len(df) > 0:
            first_row = df.iloc[0]
            for col in df.columns[:3]:  # Show first 3 columns
                value = str(first_row[col])[:100] + "..." if len(str(first_row[col])) > 100 else str(first_row[col])
                print(f"     {col}: {value}")
    except Exception as e:
        print(f"   Error reading dataset: {e}")
        return
    
    # 4. Example workflow commands
    print("\n4. Example workflow commands:")
    print("   # Step 1: Run prompt dissection")
    print("   python PromptAnatomy/process_sentence.py")
    print("   python PromptAnatomy/new_auto_recognition.py")
    print()
    print("   # Step 2: Run component perturbation")
    print("   python ComPerturb/main.py --strategy COD --tag Role --input_file process_leetcode_step2_llama3_8b.json --output_file results/leetcode_COD_Role.json")
    print()
    print("   # Step 3: Run complexity analysis")
    print("   python PCM.py")
    
    # 5. Show available strategies
    print("\n5. Available perturbation strategies:")
    strategies = [
        ("COD", "Component Deletion"),
        ("SCI", "Special Character Insertion"),
        ("SER", "Sentence Rewriting"),
        ("SYR", "Synonym Replacement"),
        ("WOD", "Word Deletion")
    ]
    for code, name in strategies:
        print(f"   - {code}: {name}")
    
    # 6. Show available tags
    print("\n6. Available component tags:")
    tags = [
        "Role",
        "Directive", 
        "Additional Information",
        "Output Formatting",
        "Examples"
    ]
    for tag in tags:
        print(f"   - {tag}")
    
    print("\n=== Example workflow complete ===")
    print("You can now run the actual processing using the commands shown above.")

if __name__ == "__main__":
    main() 