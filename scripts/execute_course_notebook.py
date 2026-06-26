from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "DL课程设计.ipynb")
RUNTIME_DIR = ROOT / ".jupyter_runtime"
RUNTIME_DIR.mkdir(exist_ok=True)

os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(RUNTIME_DIR))
os.environ.setdefault("JUPYTER_ALLOW_INSECURE_WRITES", "1")

import nbformat
from nbclient import NotebookClient


nb = nbformat.read(NOTEBOOK, as_version=4)
client = NotebookClient(
    nb,
    timeout=7200,
    iopub_timeout=180,
    kernel_name="data",
    resources={"metadata": {"path": str(ROOT)}},
)
client.execute()
nbformat.write(nb, NOTEBOOK)
print(f"Executed and saved {NOTEBOOK}")
