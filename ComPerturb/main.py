import argparse
import pandas as pd
import json
import logging
import os
from tqdm import tqdm
from utils.api_utils import init_openai_client, safe_openai_request
from models.similarity import SimilarityCalculator
from strategies.COD import CODStrategy
from strategies.SCI import SCIStrategy
from strategies.SER import SERStrategy
from strategies.SYR import SYRStrategy
from strategies.WOD import WODStrategy
from utils.prompt_utils import split_prompt, combine_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = ""
base_url = ""

class AttackProcessor:
    def __init__(self):
        self.client = init_openai_client(api_key, base_url)
        self.similarity_calculator = SimilarityCalculator(api_key, base_url)
        self.strategies = {
            'COD': CODStrategy,
            'SCI': SCIStrategy,
            'SER': SERStrategy,
            'SYR': SYRStrategy,
            'WOD': WODStrategy
        }
    
    def process_file(self, input_file, output_file, strategy_name, tag_name):
        if input_file.endswith('.json'):
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            df = pd.read_excel(input_file)
            
        context_col = 'context_output'
        if context_col not in df.columns:
            raise ValueError(f"Column '{context_col}' does not exist.")
        if 'answer' not in df.columns:
            raise ValueError(f"Column 'answer' does not exist.")
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' does not exist. Options: {list(self.strategies.keys())}")
        
        strategy = self.strategies[strategy_name]()
        rewritten_col = f"context_{tag_name}_{strategy_name}"
        output_col = f"output_{tag_name}_{strategy_name}"
        compare_col = f"compare_{tag_name}_{strategy_name}"
        for col in [rewritten_col, output_col, compare_col]:
            df[col] = None
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            context = row[context_col]
            parts = split_prompt(context)
            if tag_name not in parts:
                logging.warning(f"Row {idx} did not find tag {tag_name}, skipping.")
                continue
            modified_parts = parts.copy()
            if modified_parts[tag_name].strip():
                result_list = strategy._process_parts({tag_name: modified_parts[tag_name]})
                modified_parts[tag_name] = result_list[0]['context'] if result_list else modified_parts[tag_name]
            modified_context = combine_prompt(modified_parts)
            df.at[idx, rewritten_col] = modified_context
            response = safe_openai_request(
                self.client,
                messages=[{"role": "user", "content": modified_context}]
            )
            df.at[idx, output_col] = response if response is not None else "API request failed"
            df.at[idx, compare_col] = self.similarity_calculator.compare(
                df.at[idx, output_col],
                row['answer']
            )
        
        if output_file.endswith('.json'):
            json_data = df.to_dict('records')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
        else:
            df.to_excel(output_file, index=False)
            
        print(f"Processing complete, results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Adversarial sample generator: specify strategy and tag")
    parser.add_argument('--strategy', type=str, required=True, help='Strategy name, e.g. COD/SCI/SER/SYR/WOD')
    parser.add_argument('--tag', type=str, required=True, help='Tag name, e.g. Role/Directive/Additional Information/Output Formatting/Examples')
    parser.add_argument('--input_file', type=str, required=True, help='Input file path (.json)')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path (.json)')
    args = parser.parse_args()

    processor = AttackProcessor()
    processor.process_file(args.input_file, args.output_file, args.strategy, args.tag)

if __name__ == "__main__":
    main() 