from contextlib import contextmanager
import gc
import torch
import logging
from tabulate import tabulate
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from lrn_sparseatt.masks import BooleanMask, boolean_mask_to_jagged_indices
from lrn_sparseatt.attention import (
    masked_attention,
    sparse_attention_masked,
    sparse_attention_3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="profiles/5_profile.log",
    filemode="w",
)


def format_timing(t: float) -> str:
    if t < 1e-3:
        return f"{t * 1e6:.2f} µs"
    elif t < 1.0:
        return f"{t * 1e3:.2f} ms"
    else:
        return f"{t:.2f} s"


N_RUNS = 50


@contextmanager
def profile_and_report(name):
    gc.collect()
    sched = schedule(skip_first=10, wait=5, warmup=5, active=N_RUNS - 20)
    try:
        with profile(activities=[ProfilerActivity.CPU], schedule=sched) as prof:
            with record_function(name):
                yield prof
    finally:
        logging.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))


def profile_attention(
    seq_len: int, d_model: int, n_heads: int, sparsity: float = 0.0
) -> tuple[float, float]:

    q = torch.randn((n_heads, seq_len, d_model // n_heads))
    k = torch.randn((n_heads, seq_len, d_model // n_heads))
    v = torch.randn((n_heads, seq_len, d_model // n_heads))
    attn_mask = BooleanMask.random(seq_len, sparsity)

    bool_mask = attn_mask.as_tensor(seq_len)
    indices = attn_mask.to_indices()
    k_indices, q_offsets = boolean_mask_to_jagged_indices(bool_mask)

    with profile_and_report("masked") as prof:
        for _ in range(N_RUNS):
            masked_attention(q, k, v, bool_mask)
            prof.step()

    dense_time = prof.key_averages().self_cpu_time_total / (N_RUNS - 20)

    with profile_and_report("sparse_masked") as prof:
        for _ in range(N_RUNS):
            sparse_attention_masked(q, k, v, bool_mask)
            prof.step()

    sparse_masked_time = prof.key_averages().self_cpu_time_total / (N_RUNS - 20)

    with profile_and_report("sparse_3") as prof:
        for _ in range(N_RUNS):
            sparse_attention_3(q, k, v, k_indices, q_offsets)
            prof.step()

    sparse_3_time = prof.key_averages().self_cpu_time_total / (N_RUNS - 20)

    return dense_time / 1e6, sparse_masked_time / 1e6, sparse_3_time / 1e6


def run_profiles():
    rows = []
    CASES = [
        (seq_len, d_model, sparsity)
        for seq_len in [512, 1024, 2048]
        for d_model in [32, 64, 128]
        for sparsity in [0.5, 0.7, 0.9, 0.95, 0.99]
    ]

    for case_num, (seq_len, d_model, sparsity) in enumerate(
        tqdm(CASES, desc="Profiling cases")
    ):
        logging.info(
            f"[{case_num+1}/{len(CASES)}] Profiling attention with "
            f"seq_len={seq_len}, d_model={d_model}, n_heads=1, sparsity={sparsity:.0%}"
        )
        dense_time, sparse_masked_time, sparse_3_time = profile_attention(
            seq_len, d_model, 1, sparsity
        )
        rows.append(
            {
                "seq_len": seq_len,
                "d_model": d_model,
                "sparsity": f"{sparsity:.0%}",
                "dense_time": format_timing(dense_time),
                "sparse_3_time": format_timing(sparse_3_time),
                "sparse_masked_time": format_timing(sparse_masked_time),
            }
        )

    return rows


if __name__ == "__main__":
    rows = run_profiles()
    table = tabulate(rows, headers="keys", tablefmt="github")
    print(table)
    with open("profiles/5_profile_results.md", "w") as f:
        f.write(table)
