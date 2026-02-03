from pathlib import Path

def print_tree(root, prefix=""):
    entries = sorted(
        [e for e in root.iterdir() if e.name not in {".git", "__pycache__", ".venv", "venv"}],
        key=lambda x: (x.is_file(), x.name.lower())
    )

    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry.name)

        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(entry, prefix + extension)


if __name__ == "__main__":
    ROOT = Path(".").resolve()
    print(ROOT.name)
    print_tree(ROOT)
