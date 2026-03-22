import os
import sys
import tarfile
import requests
from pathlib import Path

def download_and_extract(url, destination_root):
    filename = url.split('/')[-1]
    # Remove both .tgz and .tar.gz extensions for the folder name
    folder_name = filename.replace('.tgz', '').replace('.tar.gz', '')
    
    target_dir = Path(destination_root) / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_archive = target_dir / filename

    print(f"\n--- Processing: {folder_name} ---")

    try:
        print(f"Downloading...")
        response = requests.get(url, stream=True, timeout=None)
        response.raise_for_status() # Check for 404s or server errors
        
        with open(temp_archive, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024): # 1MB chunks
                if chunk:
                    f.write(chunk)

        print(f"Extracting...")
        with tarfile.open(temp_archive, "r:gz") as tar:
            tar.extractall(path=target_dir)
        
        print(f"Done: {target_dir}")

    except Exception as e:
        print(f"Error processing {url}: {e}")
    finally:
        if temp_archive.exists():
            os.remove(temp_archive)

def main():
    if len(sys.argv) < 3:
        print("Usage: python pull_eds.py <links_file.txt> <destination_path>")
        sys.exit(1)

    links_file = sys.argv[1]
    dest_path = sys.argv[2]

    if not os.path.exists(links_file):
        print(f"Error: File '{links_file}' not found.")
        return

    print("Dowloads can take up to ~15 minutes for a single sequence...give it some time!")
    with open(links_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or lines starting with #
            if not line or line.startswith('#'):
                continue
            
            download_and_extract(line, dest_path)

if __name__ == "__main__":
    main()