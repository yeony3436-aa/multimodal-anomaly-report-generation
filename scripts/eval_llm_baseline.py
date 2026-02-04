"""
MMAD LLM Baseline Evaluation Script

Reproduces the paper's evaluation protocol:
- Few-shot normal templates + query image → LLM → MCQ answers
- Calculates accuracy per question type and dataset

Usage:
    # GPT-4o (paper's main model)
    python scripts/eval_llm_baseline.py --model gpt-4o --few-shot 1 --similar-template

    # Claude
    python scripts/eval_llm_baseline.py --model claude --few-shot 1 --similar-template

    # Quick test (5 images)
    python scripts/eval_llm_baseline.py --model gpt-4o --max-images 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

# Path setup
SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.mllm.base import BaseLLMClient
from src.eval.metrics import calculate_accuracy_mmad


def get_llm_client(model_name: str, **kwargs) -> BaseLLMClient:
    """Factory function to get LLM client by name."""
    model_lower = model_name.lower()

    if model_lower in ["gpt-4o", "gpt4o", "gpt-4o-mini", "gpt4o-mini", "gpt-4v", "gpt4v"]:
        from src.mllm.openai_client import GPT4Client
        # Map common names to actual model names
        model_map = {
            "gpt-4o": "gpt-4o",
            "gpt4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt4o-mini": "gpt-4o-mini",
            "gpt-4v": "gpt-4-vision-preview",
            "gpt4v": "gpt-4-vision-preview",
        }
        return GPT4Client(model=model_map.get(model_lower, model_lower), **kwargs)

    elif model_lower in ["claude", "claude-sonnet", "claude-haiku", "claude-opus"]:
        from src.mllm.claude_client import ClaudeClient
        model_map = {
            "claude": "claude-sonnet-4-20250514",
            "claude-sonnet": "claude-sonnet-4-20250514",
            "claude-haiku": "claude-3-5-haiku-20241022",
            "claude-opus": "claude-opus-4-20250514",
        }
        return ClaudeClient(model=model_map.get(model_lower, model_lower), **kwargs)

    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: gpt-4o, gpt-4o-mini, claude, claude-sonnet, claude-haiku")


def load_mmad_data(json_path: str) -> dict:
    """Load MMAD dataset JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="MMAD LLM Baseline Evaluation")

    # Model settings
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="LLM model: gpt-4o, gpt-4o-mini, claude, claude-sonnet, claude-haiku")

    # Data settings
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root path containing MMAD images (default: datasets/MMAD)")
    parser.add_argument("--mmad-json", type=str, default=None,
                        help="Path to mmad.json (default: datasets/MMAD/mmad.json)")

    # Evaluation settings
    parser.add_argument("--few-shot", type=int, default=1,
                        help="Number of few-shot examples (0-8)")
    parser.add_argument("--similar-template", action="store_true",
                        help="Use similar templates instead of random")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images to evaluate (for testing)")
    parser.add_argument("--batch-mode", action="store_true",
                        help="Ask all questions in one API call (faster but may be less accurate)")

    # Output settings
    parser.add_argument("--output-dir", type=str, default="outputs/eval",
                        help="Output directory for results")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results file")

    args = parser.parse_args()

    # Resolve paths
    data_root = args.data_root or os.environ.get("MMAD_DATA_ROOT") or str(PROJ_ROOT / "datasets" / "MMAD")
    mmad_json = args.mmad_json or os.environ.get("MMAD_JSON_PATH") or str(Path(data_root) / "mmad.json")

    # Validate paths
    if not Path(mmad_json).exists():
        print(f"Error: mmad.json not found at {mmad_json}")
        print("Please set --mmad-json or MMAD_JSON_PATH environment variable")
        sys.exit(1)

    if not Path(data_root).exists():
        print(f"Error: Data root not found at {data_root}")
        print("Please set --data-root or MMAD_DATA_ROOT environment variable")
        sys.exit(1)

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template_type = "Similar_template" if args.similar_template else "Random_template"
    output_name = f"answers_{args.few_shot}_shot_{args.model}_{template_type}"
    answers_json_path = output_dir / f"{output_name}.json"

    print(f"=== MMAD LLM Baseline Evaluation ===")
    print(f"Model: {args.model}")
    print(f"Few-shot: {args.few_shot}")
    print(f"Template: {template_type}")
    print(f"Data root: {data_root}")
    print(f"MMAD JSON: {mmad_json}")
    print(f"Output: {answers_json_path}")
    print()

    # Load existing results if resuming
    all_answers = []
    existing_images = set()

    if args.resume and answers_json_path.exists():
        with open(answers_json_path, "r", encoding="utf-8") as f:
            all_answers = json.load(f)
        existing_images = {a["image"] for a in all_answers}
        print(f"Resuming from {len(existing_images)} existing images")

    # Initialize LLM client
    try:
        llm_client = get_llm_client(args.model)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load dataset
    mmad_data = load_mmad_data(mmad_json)
    image_paths = list(mmad_data.keys())

    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"Total images: {len(image_paths)}")
    print()

    # Evaluate
    for image_rel in tqdm(image_paths, desc="Evaluating"):
        if image_rel in existing_images:
            continue

        meta = mmad_data[image_rel]

        # Get templates
        if args.similar_template:
            few_shot = meta.get("similar_templates", [])[:args.few_shot]
        else:
            few_shot = meta.get("random_templates", [])[:args.few_shot]

        # Build absolute paths
        query_image_path = str(Path(data_root) / image_rel)
        few_shot_paths = [str(Path(data_root) / p) for p in few_shot]

        # Check if image exists
        if not Path(query_image_path).exists():
            print(f"Warning: Image not found: {query_image_path}")
            continue

        # Generate answers
        if args.batch_mode:
            questions, answers, predicted, q_types = llm_client.generate_answers_batch(
                query_image_path, meta, few_shot_paths
            )
        else:
            questions, answers, predicted, q_types = llm_client.generate_answers(
                query_image_path, meta, few_shot_paths
            )

        if predicted is None:
            print(f"Error at {image_rel}")
            continue

        if len(predicted) != len(answers):
            print(f"Warning: Answer count mismatch at {image_rel}: {len(predicted)} vs {len(answers)}")
            continue

        # Calculate accuracy for this image
        correct = sum(1 for p, a in zip(predicted, answers) if p == a)
        accuracy = correct / len(answers) if answers else 0
        print(f"Image: {image_rel}, Accuracy: {accuracy:.2f}, API time: {llm_client.api_time_cost:.2f}s")

        # Store results
        for q, a, pred, qt in zip(questions, answers, predicted, q_types):
            all_answers.append({
                "image": image_rel,
                "question": q,
                "question_type": qt,
                "correct_answer": a,
                "gpt_answer": pred  # Using 'gpt_answer' for compatibility with paper's metrics code
            })

        # Save incrementally
        with open(answers_json_path, "w", encoding="utf-8") as f:
            json.dump(all_answers, f, indent=4, ensure_ascii=False)

    print()
    print("=== Calculating Metrics ===")
    calculate_accuracy_mmad(str(answers_json_path))
    print()
    print(f"Results saved to: {answers_json_path}")


if __name__ == "__main__":
    main()
