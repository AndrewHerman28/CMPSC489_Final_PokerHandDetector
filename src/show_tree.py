"""
show_tree.py - prints the project folder structure.
Folders with many files (e.g. dataset images) are summarized as a count instead of listed.

Usage: python3 show_tree.py            (current project folder)
       python3 show_tree.py some/path
"""

import sys
from pathlib import Path

SKIP = {".git", "__pycache__", ".idea", ".venv", "venv", ".DS_Store"}
MAX_FILES = 15      # folders with more files than this show a count
MAX_DEPTH = 6


def size_str(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def walk(folder, prefix="", depth=0):
    entries = sorted((p for p in folder.iterdir() if p.name not in SKIP),
                     key=lambda p: (p.is_file(), p.name.lower()))
    dirs = [p for p in entries if p.is_dir()]
    files = [p for p in entries if p.is_file()]

    if len(files) > MAX_FILES:
        exts = sorted({p.suffix or "(none)" for p in files})
        shown = dirs + [f"[{len(files)} files: {', '.join(exts)}]"]
    else:
        shown = dirs + files

    for i, item in enumerate(shown):
        last = i == len(shown) - 1
        branch = "└── " if last else "├── "
        if isinstance(item, str):
            print(prefix + branch + item)
        elif item.is_dir():
            print(prefix + branch + item.name + "/")
            if depth + 1 < MAX_DEPTH:
                walk(item, prefix + ("    " if last else "│   "), depth + 1)
            else:
                print(prefix + ("    " if last else "│   ") + "└── ...")
        else:
            print(prefix + branch + f"{item.name}  ({size_str(item.stat().st_size)})")


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
    print(root.name + "/")
    walk(root)