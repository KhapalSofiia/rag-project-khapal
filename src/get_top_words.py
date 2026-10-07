from pathlib import Path
from collections import Counter
import re


INPUT_DIR = Path("data/cleaned_scripts")
TOP_N = 20

STOP_WORDS = {
    "the", "a", "an", "i", "you", "he", "she", "it", "we", "they",
    "to", "and", "of", "in", "on", "at", "for", "from", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those",
    "as", "by", "or", "but", "if", "then", "than",
    "so", "do", "does", "did", "not", "no",
    "my", "your", "his", "her", "our", "their",
    "me", "him", "us", "them",
    "what", "who", "when", "where", "why", "how",
    "all", "any", "some", "more", "most", "other",
    "can", "could", "will", "would", "should",
    "have", "has", "had",
    "just", "like", "really", "very",
    "up", "out", "about", "into", "over", "back",
    "there", "here", "now", "only", "too"
}


def get_top_words():
    counter = Counter()

    for file_path in INPUT_DIR.glob("*.txt"):
        text = file_path.read_text(
            encoding="utf-8",
            errors="replace"
        ).lower()

        words = re.findall(r"\b[a-z]+\b", text)

        words = [
            word for word in words
            if len(word) > 1 and word not in STOP_WORDS
        ]

        counter.update(words)

    return counter.most_common(TOP_N)


def main():
    top_words = get_top_words()

    print(f"Top {TOP_N} most frequent words:\n")

    for number, (word, count) in enumerate(top_words, start=1):
        print(f"{number:2}. {word:<20} {count}")


if __name__ == "__main__":
    main()