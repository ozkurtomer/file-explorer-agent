from pathlib import Path

FILES_DIR = (Path(__file__).resolve().parent / "sample_files").resolve()

filenames = [
    "notes.txt",
    "../main.py",
]

for filename in filenames:
    file_path = (FILES_DIR / filename).resolve()
    allowed = file_path.is_relative_to(FILES_DIR)

    print("Requested:", filename)
    print("Resolved:", file_path)
    print("Allowed:", allowed)
    print()