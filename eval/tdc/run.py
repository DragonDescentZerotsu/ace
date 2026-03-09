#!/usr/bin/env python3
"""
TDC (Therapeutics Data Commons) task runner for the ACE system.

Supports running ACE on any of the 16 TDC binary classification tasks:
  AMES, BBB_Martins, Bioavailability_Ma, CYP2C9_Substrate_CarbonMangels,
  CYP2D6_Substrate_CarbonMangels, CYP3A4_Substrate_CarbonMangels,
  Carcinogens_Lagunin, ClinTox, DILI, HIA_Hou, PAMPA_NCATS,
  Pgp_Broccatelli, SARSCoV2_3CLPro_Diamond, SARSCoV2_Vitro_Touret,
  Skin_Reaction, hERG

Usage:
    python -m eval.tdc.run --task_name AMES --save_path ./results/tdc

    # Run all tasks:
    python -m eval.tdc.run --run_all --save_path ./results/tdc
"""

import os
import sys
import json
import argparse
from datetime import datetime

from .data_processor import DataProcessor, load_data
from ace import ACE
from utils import initialize_clients


# All available TDC tasks
ALL_TASKS = [
    "AMES", "BBB_Martins", "Bioavailability_Ma",
    "CYP2C9_Substrate_CarbonMangels", "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels", "Carcinogens_Lagunin",
    "ClinTox", "DILI", "HIA_Hou", "PAMPA_NCATS", "Pgp_Broccatelli",
    "SARSCoV2_3CLPro_Diamond", "SARSCoV2_Vitro_Touret",
    "Skin_Reaction", "hERG"
]

# Default data directory
DEFAULT_DATA_DIR = "/data1/tianang/Projects/Intern-S1/DataPrepare/TDC_prepended/playbook_removed"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ACE System - TDC Tasks')
    
    # Task configuration
    parser.add_argument("--task_name", type=str, default=None,
                        help="Name of the TDC task (e.g., 'AMES', 'BBB_Martins')")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all 16 TDC tasks sequentially")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Base directory containing train/valid/test splits")
    parser.add_argument("--initial_playbook_path", type=str, default=None,
                        help="Path to initial playbook. Defaults to playbooks/{task_name}.txt")
    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "online", "eval_only"],
                        help="Run mode")
    
    # Model configuration
    parser.add_argument("--api_provider", type=str, default="local",
                        help="Default API provider (default: local)")
    parser.add_argument("--generator_model", type=str, default="gpt-oss-20b",
                        help="Model name for generator (served on generator endpoint)")
    parser.add_argument("--reflector_model", type=str, default="openai/gpt-5.4",
                        help="Model name for reflector (on OpenRouter)")
    parser.add_argument("--curator_model", type=str, default="gpt-oss-120b",
                        help="Model name for curator (served on curator endpoint)")
    
    # Endpoint configuration
    parser.add_argument("--generator_base_url", type=str, nargs='+',
                        default=["http://localhost:8001/v1", "http://localhost:8002/v1", "http://localhost:8003/v1", "http://localhost:8004/v1", "http://localhost:8005/v1", "http://localhost:8006/v1"],
                        help="Base URL(s) for generator LLM. Pass one or more URLs to distribute "
                             "evaluation workload across multiple servers in round-robin fashion. "
                             "Example: --generator_base_url http://localhost:8001/v1 http://localhost:8002/v1")
    parser.add_argument("--reflector_base_url", type=str,
                        default="https://openrouter.ai/api/v1",
                        help="Base URL for reflector LLM")
    parser.add_argument("--curator_base_url", type=str,
                        default="http://localhost:8000/v1",
                        help="Base URL for curator LLM")
    parser.add_argument("--generator_api_provider", type=str, default="local",
                        help="API provider for generator")
    parser.add_argument("--reflector_api_provider", type=str, default="openrouter",
                        help="API provider for reflector")
    parser.add_argument("--curator_api_provider", type=str, default="local",
                        help="API provider for curator")
    
    # Training configuration
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_num_rounds", type=int, default=3,
                        help="Max reflection rounds for incorrect answers")
    parser.add_argument("--curator_frequency", type=int, default=1,
                        help="Run curator every N steps")
    parser.add_argument("--curator_on_correction_only", action="store_true",
                        help="Only run curator when reflection corrects a wrong answer")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate on validation set every N steps")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save intermediate playbooks every N steps")
    
    # System configuration
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max tokens for LLM responses")
    parser.add_argument("--playbook_token_budget", type=int, default=80000,
                        help="Token budget for playbook")
    parser.add_argument("--test_workers", type=int, default=20,
                        help="Number of parallel workers for evaluation")
    
    # Prompt configuration
    parser.add_argument("--json_mode", action="store_true",
                        help="Enable JSON mode for LLM calls")
    parser.add_argument("--no_ground_truth", action="store_true",
                        help="Don't use ground truth in reflection")
    
    # Test evaluation control
    parser.add_argument("--skip_initial_test", action="store_true",
                        help="Skip initial test evaluation before training")
    parser.add_argument("--skip_final_test", action="store_true",
                        help="Skip final test evaluation after training")
    parser.add_argument("--run_initial_val", action="store_true",
                        help="Run validation evaluation before training")
    parser.add_argument("--run_final_val", action="store_true",
                        help="Run validation evaluation after training")
    
    # Bulletpoint analyzer
    parser.add_argument("--use_bulletpoint_analyzer", action="store_true",
                        help="Enable bulletpoint analyzer for deduplication")
    parser.add_argument("--bulletpoint_analyzer_threshold", type=float, default=0.90,
                        help="Similarity threshold for bulletpoint analyzer")
    
    # Parse retry configuration
    parser.add_argument("--max_parse_retries", type=int, default=4,
                        help="Max extra generation attempts when answer parsing fails (default: 4)")
    
    # Output configuration
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory to save results")
    
    # Limit samples (useful for debugging)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit number of training samples (for debugging)")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Limit number of validation samples (for debugging)")
    
    return parser.parse_args()


def load_initial_playbook(path: str, task_name: str) -> str:
    """
    Load initial playbook. Falls back to playbooks/{task_name}.txt if no path given.
    
    Args:
        path: Explicit path to playbook, or None
        task_name: Task name for default path lookup
        
    Returns:
        Playbook content string, or None if not found
    """
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            content = f.read()
        print(f"Loaded initial playbook from {path}")
        return content
    
    # Try default path
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "playbooks", f"{task_name}.txt"
    )
    if os.path.exists(default_path):
        with open(default_path, 'r') as f:
            content = f.read()
        print(f"Loaded initial playbook from {default_path}")
        return content
    
    print(f"No initial playbook found for {task_name}, using empty playbook")
    return None


def preprocess_data(task_name: str, data_dir: str, mode: str,
                    max_train_samples: int = None, max_val_samples: int = None):
    """
    Load and preprocess TDC data for a given task.
    
    Args:
        task_name: TDC task name
        data_dir: Base directory with train/valid/test subdirectories
        mode: Run mode
        max_train_samples: Optional limit on training samples
        max_val_samples: Optional limit on validation samples
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples, data_processor)
    """
    processor = DataProcessor(task_name=task_name)
    
    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None
        
        test_path = os.path.join(data_dir, "test", f"{task_name}.jsonl")
        if not os.path.exists(test_path):
            # Fall back to valid set if test doesn't exist
            test_path = os.path.join(data_dir, "valid", f"{task_name}.jsonl")
        test_samples = load_data(test_path)
        test_samples = processor.process_task_data(test_samples)
        
        print(f"{mode} mode: Testing on {len(test_samples)} examples")
    else:
        # Offline mode: load train and valid
        train_path = os.path.join(data_dir, "train", f"{task_name}.jsonl")
        val_path = os.path.join(data_dir, "valid", f"{task_name}.jsonl")
        
        train_samples = load_data(train_path)
        val_samples = load_data(val_path)
        
        train_samples = processor.process_task_data(train_samples)
        val_samples = processor.process_task_data(val_samples)
        
        # Apply sample limits if set
        if max_train_samples and max_train_samples < len(train_samples):
            train_samples = train_samples[:max_train_samples]
            print(f"Limited training samples to {max_train_samples}")
        if max_val_samples and max_val_samples < len(val_samples):
            val_samples = val_samples[:max_val_samples]
            print(f"Limited validation samples to {max_val_samples}")
        
        # Check for test set
        test_path = os.path.join(data_dir, "test", f"{task_name}.jsonl")
        if os.path.exists(test_path):
            test_samples = load_data(test_path)
            test_samples = processor.process_task_data(test_samples)
        else:
            test_samples = []
        
        print(f"Offline mode: Training on {len(train_samples)} examples, "
              f"validating on {len(val_samples)}, testing on {len(test_samples)}")
    
    return train_samples, val_samples, test_samples, processor


def run_single_task(task_name: str, args):
    """
    Run ACE on a single TDC task.
    
    Args:
        task_name: TDC task name
        args: Parsed command line arguments
    """
    print(f"\n{'='*60}")
    print(f"ACE SYSTEM - TDC TASK: {task_name}")
    print(f"{'='*60}")
    print(f"Mode: {args.mode.upper()}")
    gen_urls_display = args.generator_base_url if isinstance(args.generator_base_url, list) else [args.generator_base_url]
    print(f"Generator ({len(gen_urls_display)} server(s)): {args.generator_model}")
    for url in gen_urls_display:
        print(f"  -> {url}")
    print(f"Reflector: {args.reflector_model} @ {args.reflector_base_url}")
    print(f"Curator: {args.curator_model} @ {args.curator_base_url}")
    print(f"{'='*60}\n")
    
    # Load and preprocess data
    train_samples, val_samples, test_samples, data_processor = preprocess_data(
        task_name, args.data_dir, args.mode,
        args.max_train_samples, args.max_val_samples
    )
    
    # Load initial playbook
    initial_playbook = load_initial_playbook(args.initial_playbook_path, task_name)
    
    # Create ACE system
    ace_system = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=initial_playbook,
        use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
        bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold,
        generator_base_url=args.generator_base_url,
        reflector_base_url=args.reflector_base_url,
        curator_base_url=args.curator_base_url,
        generator_api_provider=args.generator_api_provider,
        reflector_api_provider=args.reflector_api_provider,
        curator_api_provider=args.curator_api_provider,
    )
    
    # Prepare configuration
    config = {
        'num_epochs': args.num_epochs,
        'max_num_rounds': args.max_num_rounds,
        'curator_frequency': args.curator_frequency,
        'curator_on_correction_only': args.curator_on_correction_only,
        'eval_steps': args.eval_steps,
        'save_steps': args.save_steps,
        'playbook_token_budget': args.playbook_token_budget,
        'task_name': task_name,
        'mode': args.mode,
        'json_mode': args.json_mode,
        'no_ground_truth': args.no_ground_truth,
        'skip_initial_test': args.skip_initial_test,
        'skip_final_test': args.skip_final_test,
        'run_initial_val': args.run_initial_val,
        'run_final_val': args.run_final_val,
        'save_dir': args.save_path,
        'test_workers': args.test_workers,
        'initial_playbook_path': args.initial_playbook_path,
        'use_bulletpoint_analyzer': args.use_bulletpoint_analyzer,
        'bulletpoint_analyzer_threshold': args.bulletpoint_analyzer_threshold,
        'max_parse_retries': args.max_parse_retries,
        'api_provider': args.api_provider,
    }
    
    # Run
    results = ace_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples if test_samples else None,
        data_processor=data_processor,
        config=config
    )
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.run_all:
        # Run all tasks sequentially
        all_results = {}
        for task_name in ALL_TASKS:
            print(f"\n{'#'*60}")
            print(f"# STARTING TASK: {task_name}")
            print(f"{'#'*60}\n")
            try:
                results = run_single_task(task_name, args)
                all_results[task_name] = results
            except Exception as e:
                print(f"ERROR running task {task_name}: {e}")
                all_results[task_name] = {"error": str(e)}
        
        # Save consolidated results
        summary_path = os.path.join(args.save_path, "all_tasks_summary.json")
        os.makedirs(args.save_path, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll tasks complete. Summary saved to {summary_path}")
        
    elif args.task_name:
        if args.task_name not in ALL_TASKS:
            print(f"Warning: '{args.task_name}' is not in the standard task list. "
                  f"Available tasks: {ALL_TASKS}")
        run_single_task(args.task_name, args)
    else:
        print("Error: Must specify either --task_name or --run_all")
        sys.exit(1)


if __name__ == "__main__":
    main()
