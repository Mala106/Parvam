from pathlib import Path


def read_rows(file_path: Path):
    rows = []
    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            text, emotion = line.rstrip("\n").split(";", 1)
            rows.append({"text": text, "emotion": emotion})
    return rows


rows = read_rows(Path("train.txt"))
for row in rows[:5]:
    print(row)
