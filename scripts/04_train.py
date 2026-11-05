"""4단계: GPL 데이터로 bi-encoder domain adaptation (MarginMSE)."""
import argparse

from shiprag.train import train
from shiprag.utils import load_config, load_table_lookup

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
cfg = load_config(ap.parse_args().config)
p, t = cfg["paths"], cfg["train"]
table_lookup = load_table_lookup(p["tables"]) if cfg["retriever"].get("embed_tables") else None

best = train(cfg["retriever"]["base_model"], p["chunks"], p["queries"], p["triplets"], p["model_out"],
             t["batch_size"], t["lr"], t["max_steps"], t["eval_every"], t["patience"],
             t["val_ratio"], t["max_seq_length"], t["fp16"], table_lookup=table_lookup,
             micro_batch_size=t.get("micro_batch_size"))
print(f"best val_loss {best:.4f}, model saved to {p['model_out']}")
