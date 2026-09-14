#!/usr/bin/env python3
import json
from pathlib import Path

from massive_history import Client


cfg = json.loads(Path("config.json").read_text())
client = Client(cfg["api_key"], rpm=60)

queries = {
    "all_matching": ("/futures/v1/products", {"limit": 1000}),
    "HG": ("/futures/v1/products", {"product_code": "HG", "limit": 1000}),
    "ALI": ("/futures/v1/products", {"product_code": "ALI", "limit": 1000}),
}
out = {}
for name, (path, params) in queries.items():
    rows = [row for page in client.pages(path, params) for row in page]
    if name == "all_matching":
        words = ("copper", "aluminum", "aluminium", "nickel")
        rows = [row for row in rows if any(w in json.dumps(row).lower() for w in words)]
    out[name] = rows
Path("massive_metals_catalog.json").write_text(json.dumps(out, indent=2) + "\n")
print({key: len(value) for key, value in out.items()})
