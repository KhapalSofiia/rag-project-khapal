import re
import time
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://imsdb.com"
ALL_SCRIPTS_URL = "https://imsdb.com/all-scripts.html"

OUTPUT_DIR = Path("data/raw_scrips")
MAX_DOCUMENTS = 1000

HEADERS = {
    "User-Agent": "Mozilla/5.0 (educational research project)"
}

def safe_filename(name: str) -> str:
    """Convert movie title to a safe filename."""
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:150]

def get_script_links():
    """Get movie script links from IMSDb."""
    response = requests.get(
        ALL_SCRIPTS_URL,
        headers=HEADERS,
        timeout=30
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    scripts = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        title = link.get_text(strip=True)

        if "/Movie Scripts/" in href and title:
            scripts.append((title, urljoin(BASE_URL, href)))

    unique = []
    seen = set()

    for title, url in scripts:
        if url not in seen:
            seen.add(url)
            unique.append((title, url))

    return unique

def download_script(title: str, url: str, number: int):
    """Download and extract one script."""
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    script_link = None

    for link in soup.find_all("a", href=True):
        if "/scripts/" in link["href"]:
            script_link = urljoin(BASE_URL, link["href"])
            break

    if not script_link:
        print(f"[SKIP] No script found: {title}")
        return False

    script_response = requests.get(
        script_link,
        headers=HEADERS,
        timeout=30
    )
    script_response.raise_for_status()

    script_soup = BeautifulSoup(script_response.text, "html.parser")

    text = script_soup.get_text("\n", strip=True)

    if len(text) < 100:
        print(f"[SKIP] Script too short: {title}")
        return False

    filename = f"{number:03d}_{safe_filename(title)}.txt"
    output_path = OUTPUT_DIR / filename

    output_path.write_text(text, encoding="utf-8")

    print(f"[OK] {number:03d} {title}")

    return True

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Getting list of scripts...")
    scripts = get_script_links()

    print(f"Found {len(scripts)} script pages.")
    print(f"Downloading up to {MAX_DOCUMENTS} documents...\n")

    downloaded = 0

    for title, url in scripts:
        if downloaded >= MAX_DOCUMENTS:
            break

        try:
            if download_script(
                title,
                url,
                downloaded + 1
            ):
                downloaded += 1

        except requests.RequestException as e:
            print(f"[ERROR] {title}: {e}")

        except Exception as e:
            print(f"[ERROR] {title}: {e}")

        time.sleep(1)

    print("\nDone.")
    print(f"Downloaded: {downloaded}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()