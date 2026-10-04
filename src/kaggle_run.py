"""Push a notebook to Kaggle, stream its logs live, then download the outputs.

Usage: python src/kaggle_run.py notebooks/01_eda_baseline.py kaggle_01 [--no-push]

`kaggle_01` is the folder under notebooks/ holding kernel-metadata.json.
Outputs land in outputs/<folder>/.
"""
import json
import os
import subprocess
import sys
import time

from make_notebook import convert

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONUTF8"] = "1"  # Kaggle logs contain characters the Windows console encoding rejects


def status(kernel):
    out = subprocess.run(["kaggle", "kernels", "status", kernel], capture_output=True, text=True)
    return (out.stdout + out.stderr).strip().split("\n")[-1]


def main():
    source, folder = sys.argv[1], sys.argv[2]
    kdir = os.path.join(ROOT, "notebooks", folder)
    meta = json.load(open(os.path.join(kdir, "kernel-metadata.json")))
    kernel = meta["id"]

    if "--no-push" not in sys.argv:
        convert(os.path.join(ROOT, source), os.path.join(kdir, meta["code_file"]))
        subprocess.run(["kaggle", "kernels", "push", "-p", kdir], check=True)

    start = time.time()
    while "RUNNING" in status(kernel) or "QUEUED" in status(kernel):
        # -f streams every print() from the notebook as it happens
        subprocess.run(["kaggle", "kernels", "logs", "-f", kernel])
        time.sleep(5)
    print(f"\n{status(kernel)}  (watched for {time.time() - start:.0f}s)")

    out_dir = os.path.join(ROOT, "outputs", folder)
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", kernel, "-p", out_dir], check=True)


if __name__ == "__main__":
    main()
