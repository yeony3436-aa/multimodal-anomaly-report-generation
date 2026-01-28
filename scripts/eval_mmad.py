from __future__ import annotations
import argparse
from pathlib import Path

from mmad_inspector.service.settings import load_yaml, load_runtime_config
from mmad_inspector.anomaly.dummy_edge import DummyEdgeAnomaly
from mmad_inspector.mllm.echo import EchoMLLM
from mmad_inspector.eval.mmad_eval import evaluate_mmad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/eval.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    runtime_cfg = load_runtime_config(cfg["runtime_config"])

    # Step 1: dummy + echo
    anomaly = DummyEdgeAnomaly(**runtime_cfg.anomaly.get("dummy", {}))
    mllm = EchoMLLM(**runtime_cfg.mllm.get("echo", {}))

    e = cfg["eval"]
    out_dir = Path(e["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "mmad_eval.json"

    summary = evaluate_mmad(
        runtime_cfg=runtime_cfg,
        anomaly_model=anomaly,
        mllm_client=mllm,
        few_shot_k=int(e.get("few_shot_k", 1)),
        use_similar=bool(e.get("use_similar_template", True)),
        max_images=e.get("max_images", None),
        out_json=out_json,
    )

    print("=== MMAD Evaluation Summary ===")
    print(summary)
    print(f"Saved: {out_json}")

if __name__ == "__main__":
    main()
