#!/usr/bin/env python3
import os
import re
import json
import openai
import tiktoken
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

def get_client(provider, base_url=None):
    if provider == "sambanova":
        default_base_url = "https://api.sambanova.ai/v1"
        api_key = os.getenv('SAMBANOVA_API_KEY', '')
    elif provider == "together":
        default_base_url = "https://api.together.xyz/v1"
        api_key = os.getenv('TOGETHER_API_KEY', '')
    elif provider == "openai":
        default_base_url = "https://api.openai.com/v1"
        api_key = os.getenv('OPENAI_API_KEY', '')
    elif provider == "openrouter":
        default_base_url = "https://openrouter.ai/api/v1"
        api_key = os.getenv('OPENROUTER_API_KEY_Mark_3', '')
    elif provider in ["vllm", "local", "ollama", "lmstudio"]:
        default_base_url = "http://localhost:8000/v1"
        api_key = os.getenv('LOCAL_API_KEY', 'local')
    else:
        raise ValueError(f"Invalid api_provider name: {provider}. Must be 'sambanova', 'together', 'openai', 'openrouter', 'vllm', 'ollama', 'lmstudio', or 'local'")
    
    if not api_key:
        raise ValueError(f"{provider} api key not found in environment variables")
        
    final_base_url = base_url if base_url else default_base_url
    return openai.OpenAI(api_key=api_key, base_url=final_base_url)

def initialize_clients(api_provider,
                       generator_base_url=None, generator_api_provider=None,
                       reflector_base_url=None, reflector_api_provider=None,
                       curator_base_url=None, curator_api_provider=None):
    """Initialize separate clients for generator, reflector, and curator"""
    gen_provider = generator_api_provider or api_provider
    ref_provider = reflector_api_provider or api_provider
    cur_provider = curator_api_provider or api_provider

    generator_client = get_client(gen_provider, generator_base_url)
    reflector_client = get_client(ref_provider, reflector_base_url)
    curator_client = get_client(cur_provider, curator_base_url)
    
    print(f"Initialized clients - Generator: {gen_provider}, Reflector: {ref_provider}, Curator: {cur_provider}")
    return generator_client, reflector_client, curator_client

def get_section_slug(section_name):
    """Convert section name to slug format (3-5 chars)"""
    # Common section mappings - updated to match original sections
    slug_map = {
        "financial_strategies_and_insights": "fin",
        "formulas_and_calculations": "calc",
        "code_snippets_and_templates": "code",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics": "prob",
        "context_clues_and_indicators": "ctx",
        "others": "misc",
        "meta_strategies": "meta"
    }
    
    # Clean and convert to snake_case
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and")
    
    if clean_name in slug_map:
        return slug_map[clean_name]
    
    # Generate slug from first letters
    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    else:
        return "".join(w[0] for w in words[:5])

def extract_boxed_content(text):
    """Helper function to extract content from \\boxed{} format"""
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)
    if not match:
        return None
    
    start = match.end() - 1  # Position of opening brace
    brace_count = 0
    i = start
    
    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start + 1:i]  # Content between braces
        i += 1
    return None

def extract_answer(response):
    """Extract final answer from model response.
    
    Returns:
        The extracted answer string, or None if no parseable answer was found.
    """
    try:
        # First try JSON parsing
        parsed = json.loads(response)
        answer = parsed.get("final_answer", None)
        if answer is not None:
            return str(answer)
            
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass

    # JSON parsing failed, use fallback logic
    matches = re.findall(r"Finish\[(.*?)\]", response)
    if matches:
        return matches[-1]
    
    # Try to get final answer from JSON style response with regex matching 
    # Try double quotes first
    matches = re.findall(r'"final_answer"\s*:\s*"([^"]*)"', response)
    if matches:
        return matches[-1]
    
    # Try single quotes
    matches = re.findall(r"'final_answer'\s*:\s*'([^']*)'", response)
    if matches:
        return matches[-1]
    
    # Handle JSON format without quotes (for simple expressions)
    matches = re.findall(r"""['"']final_answer['"']\s*:\s*([^,}]+)""", response)
    if matches:
        answer = matches[-1].strip()
        # Clean up trailing characters
        answer = re.sub(r'[,}]*$', '', answer)
        return answer
    
    # Fallback for "The final answer is: X" pattern with boxed
    final_answer_pattern = r'[Tt]he final answer is:?\s*\$?\\boxed\{'
    match = re.search(final_answer_pattern, response)
    if match:
        # Extract boxed content starting from this match
        remaining_text = response[match.start():]
        boxed_content = extract_boxed_content(remaining_text)
        if boxed_content:
            return boxed_content
    
    # More general pattern for "final answer is X"
    matches = re.findall(r'[Tt]he final answer is:?\s*([^\n.]+)', response)
    if matches:
        answer = matches[-1].strip()
        # Clean up common formatting
        answer = re.sub(r'^\$?\\boxed\{([^}]+)\}\$?$', r'\1', answer)
        answer = answer.replace('$', '').strip()
        if answer:
            return answer
    
    # All patterns failed — could not parse an answer
    return None
    
enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(prompt: str) -> int:
    return len(enc.encode(prompt))


def evaluate_single_test_sample(args_tuple, data_processor) -> Tuple[Dict, str]:
    """
    Evaluate a single test sample with parse-retry support.
    
    Args:
        args_tuple: Tuple of (index, task_dict, generator, playbook, max_tokens, log_dir,
                              use_json_mode, max_parse_retries)
        data_processor: DataProcessor instance with answer_is_correct method
    """
    (i, task_dict, generator, playbook, max_tokens, log_dir, use_json_mode, max_parse_retries) = args_tuple
    try:
        context = task_dict["context"]
        question = task_dict["question"]
        target = task_dict["target"]

        final_answer = None
        for attempt in range(max_parse_retries + 1):
            if attempt > 0:
                print(f"[eval] Sample {i}: parse retry {attempt}/{max_parse_retries}")
            gen_response, bullet_ids, call_info = generator.generate(
                question=question,
                playbook=playbook,
                context=context,
                reflection="(empty)",
                use_json_mode=use_json_mode,
                call_id=f"test_eval_{i}_attempt_{attempt}",
                log_dir=log_dir
            )
            final_answer = extract_answer(gen_response)
            if final_answer is not None:
                break

        parse_failed = (final_answer is None)
        is_correct = (not parse_failed) and data_processor.answer_is_correct(final_answer, target)

        return {
            "index": i,
            "final_answer": final_answer,
            "target": target,
            "is_correct": is_correct,
            "parse_failed": parse_failed,
            "success": True
        }, None

    except Exception as e:
        return None, f"Error evaluating sample {i}: {type(e).__name__}: {str(e)}"


def evaluate_test_set(data_processor, generator, playbook, test_samples,
                      max_tokens=4096, log_dir=None, max_workers=20, 
                      use_json_mode=False, eval_label="TEST SET",
                      max_parse_retries=4) -> Tuple[Dict, Dict]:
    """
    Parallel evaluation of test set - task-agnostic implementation.
    
    Args:
        data_processor: DataProcessor instance with answer_is_correct and evaluate_accuracy methods
        generator: A single Generator instance, or a list of Generator instances.
                   When a list is provided, samples are distributed round-robin across
                   all generators so that multiple servers are used in parallel.
        playbook: Current playbook string
        test_samples: List of test samples
        max_tokens: Max tokens for generation
        log_dir: Directory for logs
        max_workers: Number of parallel workers
        use_json_mode: Whether to use JSON mode
        eval_label: Label for the evaluation header (e.g., "TEST SET", "VALIDATION SET")
        max_parse_retries: Max number of extra generation attempts when answer parsing fails.
        
    Returns:
        Tuple of (results_dict, error_logs_dict)
    """
    # Normalise: always work with a list of generators
    generators = generator if isinstance(generator, list) else [generator]
    num_servers = len(generators)

    print(f"\n{'='*40}")
    print(f"EVALUATING {eval_label} - {len(test_samples)} samples, {max_workers} workers, {num_servers} server(s)")
    print(f"{'='*40}")

    # Distribute samples round-robin across the generator pool
    args_list = [
        (i, sample, generators[i % num_servers], playbook, max_tokens, log_dir, use_json_mode, max_parse_retries)
        for i, sample in enumerate(test_samples)
    ]

    results = {
        "correct": 0, "total": 0, "no_answer": 0,
        "answers": [], "targets": [], "errors": []
    }

    # Use a wrapper to pass data_processor to the evaluation function
    def eval_wrapper(args_tuple):
        return evaluate_single_test_sample(args_tuple, data_processor)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {
            executor.submit(eval_wrapper, args): args 
            for args in args_list
        }

        for i, future in enumerate(as_completed(future_to_args), 1):
            result, error = future.result()
            
            if error:
                print(error)
                continue

            if result and result["success"]:
                parse_failed = result.get("parse_failed", result["final_answer"] is None)
                
                if parse_failed:
                    # Parse failure: count as wrong but do NOT append None to answers list
                    # (None would crash evaluate_accuracy / answer_is_correct)
                    results["total"] += 1
                    results["no_answer"] += 1
                    results["errors"].append({
                        "index": result["index"],
                        "prediction": None,
                        "ground_truth": result["target"]
                    })
                else:
                    results["correct"] += (1 if result["is_correct"] else 0)
                    results["total"] += 1
                    results["answers"].append(result["final_answer"])
                    results["targets"].append(result["target"])
                    
                    if not result["is_correct"]:
                        results["errors"].append({
                            "index": result["index"],
                            "prediction": result["final_answer"],
                            "ground_truth": result["target"]
                        })

            if i % 50 == 0:
                curr_acc = results["correct"] / results["total"] if results["total"] > 0 else 0
                print(f"Progress: {i}/{len(args_list)}, Accuracy: {curr_acc:.3f}")
    
    if results["answers"] and results["targets"]:
        accuracy = data_processor.evaluate_accuracy(results["answers"], results["targets"])
        
        final_results = {
            "accuracy": accuracy,
            "correct": results["correct"],
            "total": results["total"],
            "no_answer": results["no_answer"]
        }
        
        # Compute macro F1 if the data processor supports it
        if hasattr(data_processor, 'evaluate_f1'):
            macro_f1 = data_processor.evaluate_f1(results["answers"], results["targets"])
            final_results["macro_f1"] = macro_f1
        
        error_logs = {
            "accuracy": accuracy,
            "errors": results["errors"]
        }
        
        f1_str = f", Macro F1: {final_results['macro_f1']:.3f}" if "macro_f1" in final_results else ""
        print(f"\n📊 Final Accuracy: {accuracy:.3f} ({results['correct']}/{results['total']}){f1_str}")
    else:
        final_results = {"accuracy": 0.0, "correct": 0, "total": 0}
        error_logs = {}
        print(f"\n📊 No valid results!")
        
    return final_results, error_logs