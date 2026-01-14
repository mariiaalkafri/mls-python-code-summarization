
import os
import ast
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset

def is_valid_python(code):
    try:
        tree = ast.parse(code)
        return True, tree
    except SyntaxError:
        return False, None
    except Exception:
        return False, None

def has_docstring(tree):
    # Check if the code has a docstring in the AST (input code itself)
    return ast.get_docstring(tree) is not None

def clean_summary(docstring):
    if not docstring:
        return None
    
    # Take first line
    lines = docstring.strip().split('\n')
    summary = lines[0].strip()
    
    # Filter rules
    words = summary.split()
    if len(words) < 3:
        return None
    
    if summary[-1] in [',', ':', ';']:
        return None
    
    # Check for truncated starts
    bad_starts = ['After', 'Before', 'When']
    if words[0] in bad_starts:
        return None
        
    return summary

def process_split(dataset_split, output_path):
    print(f"Processing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        dropped_count = 0
        kept_count = 0
        
        for sample in tqdm(dataset_split):
            code = sample['code']
            doc_raw = sample['docstring']
            
            # 1. Parse code
            valid, tree = is_valid_python(code)
            if not valid:
                dropped_count += 1
                continue
                
            # 2. Drop if input code has docstring
            if has_docstring(tree):
                dropped_count += 1
                continue
                
            # 3. Clean summary
            summary = clean_summary(doc_raw)
            if not summary:
                dropped_count += 1
                continue
                
            # Save
            json_record = {
                "code": code,
                "summary": summary
            }
            f.write(json.dumps(json_record) + '\n')
            kept_count += 1
            
    print(f"Finished {output_path}. Kept: {kept_count}, Dropped: {dropped_count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/processed/codesearchnet_clean")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("AhmedSSoliman/CodeSearchNet", trust_remote_code=True)
    
    # Process each split
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            output_file = os.path.join(args.output_dir, f"{split}.jsonl")
            process_split(dataset[split], output_file)

if __name__ == "__main__":
    main()
