from pathlib import Path
import re
import unicodedata
from bs4 import BeautifulSoup

INPUT_DIR = Path("data/raw_scrips")
OUTPUT_DIR = Path("data/cleaned_scripts")

MIN_LENGTH = 100

def clean_text(text: str) -> str:
    """Clean one IMSDb script."""

    text = unicodedata.normalize("NFKC", text)

    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text("\n")

    text = text.replace("\x00", "")

    marker = "ALL SCRIPTS"

    if marker in text:
        text = text.split(marker, 1)[1]

    text = re.sub(r"[ \t]+", " ", text)

    lines = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            lines.append(line)

    text = "\n".join(lines)

    text = text.strip()

    return text

def process_corpus():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    cleaned = 0
    removed = 0

    for input_file in INPUT_DIR.glob("*.txt"):
        total += 1

        text = input_file.read_text(
            encoding="utf-8",
            errors="replace"
        )

        cleaned_text = clean_text(text)

        if len(cleaned_text) < MIN_LENGTH:
            removed += 1
            print(f"[REMOVE] {input_file.name}")
            continue

        output_file = OUTPUT_DIR / input_file.name
        output_file.write_text(
            cleaned_text,
            encoding="utf-8"
        )

        cleaned += 1
        print(f"[OK] {input_file.name}")

    print("\nCleaning finished.")
    print(f"Documents before: {total}")
    print(f"Documents after:  {cleaned}")
    print(f"Removed:          {removed}")


if __name__ == "__main__":
    process_corpus()