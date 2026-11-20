"""Cut the per-second lab journals into every measured language block of the stage 2 runs.

Journals (tools/lab_monitor.sh, tools/lab_memmon.py in the context repo) are matched to blocks by the
UTC bounds the runner records (measured_started_utc / measured_finished_utc). Per block it adds CPU memory
read/write GB/s, CPU package W, GPU W, SM clock, power/thermal violations, PCIe MB/s, and the KV-cache
bytes the CPU is expected to read (CPU-decoded tokens/s x context x KV bytes per token).

    python scripts/align_journals.py RESULTS_DIR OUTPUT.json   # RESULTS_DIR = clone of pdc-lab-runs
"""
import json, glob, sys, bisect, statistics as st, collections as co
from datetime import datetime, timezone
D = sys.argv[1]
def ts(s): return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
def load_mem(pc):
    rows = []
    for l in open(f"{D}/{pc}/logs/cpu-mem.log"):
        if l.startswith("#"): continue
        p = l.split(); rows.append((ts(p[0] + " " + p[1]), float(p[2]), float(p[3]), float(p[4])))
    return rows
def load_dmon(pc, gpu):
    rows = []
    for l in open(f"{D}/{pc}/logs/gpu-dmon.log"):
        if l.startswith("#"): continue
        p = l.split()
        if len(p) < 18 or p[2] != str(gpu): continue
        d = p[0]; t = ts(f"{d[:4]}-{d[4:6]}-{d[6:]} {p[1]}")
        f = lambda x: float(x) if x not in ("-",) else 0.0
        rows.append((t, f(p[3]), f(p[13]), f(p[14]), f(p[15]), f(p[16]), f(p[17]), f(p[6])))  # W pclk pviol tviol rx tx sm
    return rows
def window(rows, a, b):
    i = bisect.bisect_left(rows, (a,)); j = bisect.bisect_right(rows, (b, 1e18))
    return rows[i:j]
KV = {"llama2": 2 * 32 * 4096 * 2, "llama3": 2 * 32 * 1024 * 2}
runs = [("pc1", "serving-llama2-pc1-v3", 0, "llama2"), ("pc1", "serving-llama3-pc1-v1", 0, "llama3"), ("pc2", "serving-llama2-pc2-v3", 1, "llama2")]
out = []
for pc, run, gpu, model in runs:
    mem, dmon = load_mem(pc), load_dmon(pc, gpu)
    for f in glob.glob(f"{D}/{pc}/runs/{run}/cells/*.json"):
        c = json.load(open(f)); k = c["cell"]
        for lang, b in c["languages"].items():
            a = datetime.fromisoformat(b["measured_started_utc"]).timestamp(); e = datetime.fromisoformat(b["measured_finished_utc"]).timestamp()
            m, g = window(mem, a, e), window(dmon, a, e)
            if not m or not g: continue
            w = b["work"]; wall = b["wall_s"]
            ctx = b["mean_input_tokens"] + 64
            cpu_tok_s = w.get("cpu_decode_tokens", 0) / wall
            out.append(dict(pc=pc, model=model, system=k["system"], blocks=k["blocks"], load=k["load"], lang=lang, round=c["round"],
                tok_s=b["generated_tokens_per_s"], cpu_share=w.get("cpu_decode_share"), swaps=w.get("swap_out_requests") or 0,
                read_gbs=st.mean(x[1] for x in m) / 1000, write_gbs=st.mean(x[2] for x in m) / 1000, cpu_w=st.mean(x[3] for x in m),
                gpu_w=st.mean(x[1] for x in g), pclk=st.mean(x[2] for x in g), pviol=st.mean(x[3] for x in g), tviol=max(x[4] for x in g),
                rx=st.mean(x[5] for x in g), tx=st.mean(x[6] for x in g), sm=st.mean(x[7] for x in g),
                kv_read_gbs=cpu_tok_s * ctx * KV[model] / 1e9, wall=wall, tokens=b["requests"] * 128))
json.dump(out, open(sys.argv[2], "w"))
print(len(out), "blocks with journal windows")
