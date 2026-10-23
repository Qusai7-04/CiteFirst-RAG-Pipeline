import os
import requests

from pathlib import Path
ROOT = Path(__file__).resolve().parent
papers_dir = str(ROOT / "papers")
api_url = "http://127.0.0.1:8000/api/vault/upload"

def upload_papers():
    if not os.path.exists(papers_dir):
        print("No papers directory found.")
        return

    files = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]
    print(f"Found {len(files)} PDFs. Starting uploads to API...")

    for filename in files:
        filepath = os.path.join(papers_dir, filename)
        try:
            with open(filepath, "rb") as f:
                print(f"Uploading {filename}...")
                response = requests.post(api_url, files={"file": (filename, f, "application/pdf")})
                if response.status_code == 200:
                    print(f" -> Success: {response.json()}")
                else:
                    print(f" -> Failed: {response.text}")
        except Exception as e:
            print(f" -> Error reading {filename}: {e}")

if __name__ == "__main__":
    upload_papers()
