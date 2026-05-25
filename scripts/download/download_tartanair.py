#!/usr/bin/env python3
"""
Example:
python scripts/download_tartanair.py \
  --root data/tartanair \
  --env-event office \
  --env-air Office \
  --difficulty easy hard

Multiple environments can be passed as paired lists:
python scripts/download_tartanair.py \
  --root data/tartanair \
  --env-event office carwelding \
  --env-air Office CarWelding \
  --difficulty easy hard

Run this when these TartanEvent folders are already present under --root.
The office_events folder is the staging folder for office, so office is the
environment name to pass here.
python scripts/download/download_tartanair.py \
  --root data/tartanair \
  --env-event abandonedfactory abandonedfactory_night amusement carwelding endofworld gascola hospital japanesealley neighborhood ocean office oldtown seasidetown seasonsforest seasonsforest_winter soulcity westerndesert \
  --env-air abandonedfactory abandonedfactory_night amusement carwelding endofworld gascola hospital japanesealley neighborhood ocean office oldtown seasidetown seasonsforest seasonsforest_winter soulcity westerndesert \
  --difficulty easy hard \
  --skip-event
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
import urllib.request
import urllib.parse
import time
from collections.abc import Iterable
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

TARTANEVENT_ROOT_URL = (
    "https://download.ifi.uzh.ch/rpg/web/data/iros24_rampvo/datasets/TartanEvent"
)
TARTANAIR_FILE_LIST = REPO_ROOT / "download_training_zipfiles.txt"
TARTANAIR_FILE_LIST_URL = (
    "https://raw.githubusercontent.com/castacks/tartanair_tools/master/"
    "download_training_zipfiles.txt"
)
TARTANAIR_AIRLAB_ENDPOINT = (
    "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/"
    "AUTH_ac8533a83cff4d48bc8c608ad222d330"
)
TARTANAIR_HF_REPO_ID = "theairlabcmu/tartanair"
TARTANEVENT_TEST_URL = (
    "https://download.ifi.uzh.ch/rpg/web/data/iros24_rampvo/datasets/"
    "TartanEvent_competition.zip"
)
TARTANAIR_TEST_AIR_URL = (
    "https://drive.google.com/file/d/1N9BkpQuibIyIBkLxVPUuoB-eDOMFqY8D/view?usp=sharing"
)
TARTANAIR_TEST_AIR_ARCHIVE = "tartanair-test-mono-release.tar.gz"
IMAGE_EXTENSIONS = {".png", ".jpg"}

def download_with_retry(url: str, target: str) -> None:
    import os
    # Ora usiamo SOLO final_path, niente più part_path
    final_path = Path(target)
    
    # Controllo integrità iniziale
    if final_path.exists():
        if zipfile.is_zipfile(final_path):
            print(f"\nFile {final_path.name} already fully downloaded and valid.")
            return
        else:
            print(f"\nFound incomplete {final_path.name}. Resuming directly on the file...")

    while True:
        try:
            req = urllib.request.Request(url)
            file_size = final_path.stat().st_size if final_path.exists() else 0
            
            if file_size > 0:
                req.add_header("Range", f"bytes={file_size}-")
                print(f"Attempt: Resuming {final_path.name} from {file_size / (1024**3):.2f} GB...")
            else:
                print(f"Attempt: Starting new download for {final_path.name}...")

            with urllib.request.urlopen(req, timeout=30) as response:
                is_partial = response.status == 206
                mode = "ab" if is_partial else "wb"
                
                if not is_partial and file_size > 0:
                    print("Server ignored Range header. Restarting...")
                    file_size = 0
                    mode = "wb"
                
                total_size = int(response.headers.get('content-length', 0))
                if is_partial:
                    total_size += file_size

                with open(final_path, mode) as f:
                    with tqdm(
                        total=total_size, 
                        initial=file_size, 
                        unit='B', 
                        unit_scale=True, 
                        unit_divisor=1024,
                        desc=final_path.name
                    ) as pbar:
                        while True:
                            chunk = response.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno()) 
                            pbar.update(len(chunk))
            
            actual_size = final_path.stat().st_size
            if total_size > 0 and actual_size < total_size:
                raise Exception(f"Network error. Downloaded {actual_size} / {total_size} bytes.")

            print(f"\nDownload completed successfully: {final_path.name}")
            return

        except urllib.error.HTTPError as e:
            if e.code == 416:
                print("\nFile is already fully downloaded.")
                return
            print(f"\nAttempt failed (HTTP {e.code}). Retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"\nAttempt failed: {e}. Retrying in 5s...")
            time.sleep(5)
            

def google_drive_file_id(url_or_id: str) -> str:
    match = re.search(r"/d/([^/]+)", url_or_id)
    if match:
        return match.group(1)

    match = re.search(r"[?&]id=([^&]+)", url_or_id)
    if match:
        return match.group(1)

    return url_or_id


def is_supported_archive(path: Path) -> bool:
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def require_supported_archive(path: Path) -> None:
    if is_supported_archive(path):
        return
    path.unlink(missing_ok=True)
    raise RuntimeError(f"Downloaded file is not a supported archive: {path}")


def stream_download_response(response, target: Path) -> None:
    total_size = int(response.headers.get("content-length", 0))
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as fh:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=target.name,
        ) as pbar:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                pbar.update(len(chunk))


def normalize_google_drive_url(url: str) -> str:
    return html.unescape(url).replace("\\u003d", "=").replace("\\u0026", "&").replace("\\/", "/")


def parse_google_drive_form_url(body: str) -> str | None:
    for form_match in re.finditer(r"<form\b[^>]*>.*?</form>", body, re.IGNORECASE | re.DOTALL):
        form = form_match.group(0)
        if "download-form" not in form and "drive.usercontent.google.com/download" not in form:
            continue

        action_match = re.search(
            r"""\baction\s*=\s*["']([^"']+)["']""",
            form,
            re.IGNORECASE,
        )
        if not action_match:
            continue

        params = []
        for input_match in re.finditer(r"<input\b[^>]*>", form, re.IGNORECASE | re.DOTALL):
            input_tag = input_match.group(0)
            name_match = re.search(
                r"""\bname\s*=\s*["']([^"']+)["']""",
                input_tag,
                re.IGNORECASE,
            )
            value_match = re.search(
                r"""\bvalue\s*=\s*["']([^"']*)["']""",
                input_tag,
                re.IGNORECASE,
            )
            if name_match:
                params.append(
                    (
                        html.unescape(name_match.group(1)),
                        html.unescape(value_match.group(1)) if value_match else "",
                    )
                )

        confirm_url = urllib.parse.urljoin(
            "https://drive.google.com",
            html.unescape(action_match.group(1)),
        )
        if params:
            separator = "&" if urllib.parse.urlparse(confirm_url).query else "?"
            confirm_url = f"{confirm_url}{separator}{urllib.parse.urlencode(params)}"
        return confirm_url

    return None


def google_drive_confirm_url(body: str, cookie_jar, base_url: str) -> str | None:
    download_url_match = re.search(r'"downloadUrl"\s*:\s*"([^"]+)"', body)
    if download_url_match:
        return normalize_google_drive_url(download_url_match.group(1))

    link_match = re.search(
        r"""href=["']([^"']*(?:uc\?export=download|drive\.usercontent\.google\.com/download)[^"']*)["']""",
        body,
    )
    if link_match:
        confirm_url = normalize_google_drive_url(link_match.group(1))
        return urllib.parse.urljoin("https://drive.google.com", confirm_url)

    form_url = parse_google_drive_form_url(body)
    if form_url:
        return form_url

    for cookie in cookie_jar:
        if cookie.name.startswith("download_warning"):
            return f"{base_url}&confirm={cookie.value}"

    return None


def download_google_drive_file(url_or_id: str, target: Path) -> None:
    import http.cookiejar

    if target.exists():
        if is_supported_archive(target):
            print(f"\nFile {target.name} already exists.")
            return
        print(f"\nFound incomplete {target.name}. Restarting download.")
        target.unlink()

    file_id = google_drive_file_id(url_or_id)
    gdown_cmd = [
        sys.executable,
        "-m",
        "gdown",
        file_id,
        "-O",
        str(target),
    ]
    try:
        print(" ".join(gdown_cmd))
        subprocess.run(gdown_cmd, check=True)
        require_supported_archive(target)
        return
    except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as exc:
        print(f"\ngdown download failed: {exc}. Falling back to urllib downloader.")

    base_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            cookie_jar = http.cookiejar.CookieJar()
            opener = urllib.request.build_opener(
                urllib.request.HTTPCookieProcessor(cookie_jar)
            )

            with opener.open(base_url, timeout=30) as response:
                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type.lower():
                    stream_download_response(response, target)
                    require_supported_archive(target)
                    return

                body = response.read().decode("utf-8", errors="ignore")

            confirm_url = google_drive_confirm_url(body, cookie_jar, base_url)
            if confirm_url is None:
                raise RuntimeError("Could not find Google Drive download confirmation link.")

            with opener.open(confirm_url, timeout=30) as response:
                stream_download_response(response, target)
            require_supported_archive(target)
            return
        except Exception as exc:
            if attempt == max_attempts:
                raise RuntimeError(
                    "Google Drive download failed after "
                    f"{max_attempts} attempts. Install gdown in this Python "
                    "environment or pass --test-air-url with a direct archive URL."
                ) from exc
            print(
                f"\nGoogle Drive download failed: {exc}. "
                f"Retrying in 5s ({attempt}/{max_attempts})..."
            )
            time.sleep(5)


def should_skip_archive_member(name: str, skip_extensions: set[str] | None) -> bool:
    if not skip_extensions:
        return False
    return Path(name).suffix.lower() in skip_extensions


def extract_archive(
    archive_path: Path,
    extract_dir: Path,
    delete_archive: bool,
    skip_extensions: set[str] | None = None,
) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        print(f"Verifying archive: {archive_path}")
        with zipfile.ZipFile(archive_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"Corrupted zip archive {archive_path}, first bad file: {bad}")
            print(f"Extracting {archive_path} into {extract_dir}")
            for member in zf.infolist():
                if should_skip_archive_member(member.filename, skip_extensions):
                    continue
                zf.extract(member, path=extract_dir)
    elif tarfile.is_tarfile(archive_path):
        print(f"Extracting {archive_path} into {extract_dir}")
        with tarfile.open(archive_path, "r:*") as tf:
            members = (
                member
                for member in tf.getmembers()
                if not should_skip_archive_member(member.name, skip_extensions)
            )
            tf.extractall(path=extract_dir, members=members)
    else:
        raise RuntimeError(f"Unsupported or corrupted archive: {archive_path}")

    if delete_archive:
        print(f"Deleting {archive_path}")
        archive_path.unlink(missing_ok=True)


def download_archive(url: str, target: Path) -> None:
    if "drive.google.com" in url:
        download_google_drive_file(url, target)
    else:
        download_with_retry(url, str(target))



def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def download_tartanevent(root: Path, env_zip: str, unzip: bool, delete_zip: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)

    env_folder = root / f"{env_zip}_events"
    env_folder.mkdir(parents=True, exist_ok=True)

    zip_name = env_zip if env_zip.endswith(".zip") else f"{env_zip}.zip"
    target_zip = root / zip_name
    url = f"{TARTANEVENT_ROOT_URL}/{zip_name}"

    print(f"Downloading TartanEvent: {url}")
    # run(["curl", "-L", "--fail", "-C", "-", "-o", str(target_zip), url])
    download_with_retry(url, str(target_zip))

    print(f"Verifying archive: {target_zip}")
    with zipfile.ZipFile(target_zip, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupted zip archive, first bad file: {bad}")

    if unzip:
        print(f"Extracting {target_zip} into {env_folder}")
        with zipfile.ZipFile(target_zip, "r") as zf:
            zf.extractall(path=env_folder)

        if delete_zip:
            print(f"Deleting {target_zip}")
            target_zip.unlink(missing_ok=True)


def download_tartanair_imu(root: Path, env: str, difficulties: list[str]) -> None:
    import tartanair as ta
    import time
    import zipfile

    root.mkdir(parents=True, exist_ok=True)

    print(f"Initializing TartanAir at {root}")
    ta.init(str(root))

    for diff in difficulties:
        print(f"\nDownloading TartanAir IMU for env={env}, difficulty={diff}")
        # Loop to ensure download goes to the end
        while True:
            try:
                ta.download(
                    env=[env],
                    difficulty=[diff],
                    modality=['imu'],
                    camera_name=["lcam_front"],
                    unzip=True,
                )
                break 
                
            except Exception as e:
                print(f"\nIMU download failed for {diff} (Error: {e}). Retrying in 5s...")
                time.sleep(5)

        diff_dir = root / env / f"Data_{diff}"
        if diff_dir.exists():
            for item in diff_dir.iterdir():
                if item.suffix == '.zip':
                    print(f"Extracting {item.name}...")
                    try:
                        with zipfile.ZipFile(item, 'r') as zf:
                            zf.extractall(diff_dir)
                        item.unlink() 
                    except Exception as e:
                        print(f"Error while extracting {item.name}: {e}")


class DirectAirLabDownloader:
    def __init__(self, bucket_name: str, workers: int) -> None:
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.client import Config
        except ImportError as exc:
            raise ImportError("Direct TartanAir download requires boto3: pip install boto3") from exc

        self.client = boto3.client(
            "s3",
            endpoint_url=TARTANAIR_AIRLAB_ENDPOINT,
            config=Config(
                signature_version=UNSIGNED,
                connect_timeout=30,
                read_timeout=60,
                retries={"max_attempts": 5, "mode": "standard"},
            ),
        )
        self.bucket_name = bucket_name
        self.workers = workers

    def _download_one(self, source_file: str, root: Path) -> tuple[str, Path, str]:
        target_file = root / PurePosixPath(source_file)
        target_file.parent.mkdir(parents=True, exist_ok=True)

        if target_file.exists() and zipfile.is_zipfile(target_file):
            return source_file, target_file, "exists"

        part_file = target_file.with_suffix(target_file.suffix + ".part")
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=source_file)
            total_size = int(response.get("ContentLength", 0))
            with open(part_file, "wb") as fh:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=Path(source_file).name,
                ) as pbar:
                    for chunk in response["Body"].iter_chunks(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                            pbar.update(len(chunk))
            part_file.replace(target_file)
            return source_file, target_file, "ok"
        except Exception as exc:
            return source_file, target_file, f"error: {exc}"

    def download(self, filelist: list[str], root: Path) -> list[Path]:
        downloaded = []
        had_error = False

        print(f"Downloading {len(filelist)} TartanAir file(s) from {self.bucket_name}")
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._download_one, source_file, root) for source_file in filelist]
            for future in as_completed(futures):
                source_file, target_file, status = future.result()
                if status in {"ok", "exists"}:
                    print(f"  {status:6s} {source_file} -> {target_file}")
                    downloaded.append(target_file)
                else:
                    had_error = True
                    print(f"  FAIL   {source_file}: {status}")

        if had_error:
            raise RuntimeError("Some TartanAir files failed to download.")
        return downloaded


class HuggingFaceTartanAirDownloader:
    def __init__(self, repo_id: str = TARTANAIR_HF_REPO_ID, chunk_size: int = 100) -> None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Hugging Face TartanAir download requires huggingface-hub: "
                "pip install huggingface-hub"
            ) from exc

        self.repo_id = repo_id
        self.chunk_size = chunk_size
        self.snapshot_download = snapshot_download

    def download(self, filelist: list[str], root: Path) -> list[Path]:
        downloaded = []
        print(f"Downloading {len(filelist)} TartanAir file(s) from Hugging Face: {self.repo_id}")
        for start in range(0, len(filelist), self.chunk_size):
            chunk = filelist[start : start + self.chunk_size]
            self.snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                local_dir=str(root),
                allow_patterns=chunk,
            )
            downloaded.extend(root / PurePosixPath(source_file) for source_file in chunk)

        return downloaded


def ensure_tartanair_file_list(file_list: Path) -> None:
    if file_list.exists():
        return
    if file_list.name != "download_training_zipfiles.txt":
        raise FileNotFoundError(f"TartanAir file list not found: {file_list}")

    print(f"Downloading TartanAir v1 file list -> {file_list}")
    file_list.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TARTANAIR_FILE_LIST_URL, file_list)


def load_tartanair_file_sizes(file_list: Path) -> dict[str, float]:
    ensure_tartanair_file_list(file_list)
    if not file_list.exists():
        raise FileNotFoundError(f"TartanAir file list not found: {file_list}")

    file_sizes = {}
    with open(file_list, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2 or not parts[0].endswith(".zip"):
                continue
            file_sizes[parts[0]] = float(parts[1])
    return file_sizes


def select_tartanair_archives(
    file_sizes: dict[str, float],
    env: str,
    difficulties: list[str],
    archive_name: str,
) -> list[str]:
    wanted_difficulties = {diff.lower() for diff in difficulties}
    selected = []

    for source_file in file_sizes:
        parts = PurePosixPath(source_file).parts
        if len(parts) < 3:
            continue
        if parts[0].lower() != env.lower():
            continue
        if parts[-1].lower() != archive_name.lower():
            continue

        diff_part = parts[1].lower()
        diff = diff_part.replace("data_", "")
        if diff in wanted_difficulties:
            selected.append(source_file)

    if not selected:
        expected = []
        for diff in difficulties:
            expected.append(f"{env}/{diff.capitalize()}/{archive_name}")
            expected.append(f"{env}/Data_{diff.lower()}/{archive_name}")
        raise ValueError(
            f"No TartanAir archive found for env={env}, difficulties={difficulties}, "
            f"archive={archive_name}. "
            f"Expected entries like: {expected}"
        )

    return sorted(selected)


def extract_zip_archives(zip_paths: list[Path], delete_zip: bool) -> None:
    for zip_path in zip_paths:
        print(f"Verifying archive: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"Corrupted zip archive {zip_path}, first bad file: {bad}")

            print(f"Extracting {zip_path} into {zip_path.parent}")
            zf.extractall(path=zip_path.parent)

        if delete_zip:
            print(f"Deleting {zip_path}")
            zip_path.unlink(missing_ok=True)


def download_tartanair_direct(
    root: Path,
    env: str,
    difficulties: list[str],
    file_list: Path,
    bucket_name: str,
    archive_name: str,
    workers: int,
    delete_zip: bool,
) -> None:
    root.mkdir(parents=True, exist_ok=True)

    file_sizes = load_tartanair_file_sizes(file_list)
    archives = select_tartanair_archives(
        file_sizes,
        env,
        difficulties,
        archive_name,
    )
    total_size = sum(file_sizes[source_file] for source_file in archives)

    print("Direct TartanAir download")
    print(f"  env:          {env}")
    print(f"  difficulties: {', '.join(difficulties)}")
    print(f"  archive:      {archive_name}")
    print(f"  file list:    {file_list}")
    print(f"  bucket:       {bucket_name}")
    print(f"  total size:   {total_size:.3f} GB")

    downloader = DirectAirLabDownloader(bucket_name=bucket_name, workers=workers)
    zip_paths = downloader.download(archives, root)
    extract_zip_archives(zip_paths, delete_zip=delete_zip)


def download_tartanair_huggingface(
    root: Path,
    env: str,
    difficulties: list[str],
    file_list: Path,
    archive_name: str,
    delete_zip: bool,
) -> None:
    root.mkdir(parents=True, exist_ok=True)

    file_sizes = load_tartanair_file_sizes(file_list)
    archives = select_tartanair_archives(
        file_sizes,
        env,
        difficulties,
        archive_name,
    )
    total_size = sum(file_sizes[source_file] for source_file in archives)

    print("Hugging Face TartanAir download")
    print(f"  env:          {env}")
    print(f"  difficulties: {', '.join(difficulties)}")
    print(f"  archive:      {archive_name}")
    print(f"  file list:    {file_list}")
    print(f"  repo:         {TARTANAIR_HF_REPO_ID}")
    print(f"  total size:   {total_size:.3f} GB")

    downloader = HuggingFaceTartanAirDownloader()
    zip_paths = downloader.download(archives, root)
    extract_zip_archives(zip_paths, delete_zip=delete_zip)


def count_nonempty_lines(path: Path) -> int:
    with open(path, "r") as fh:
        return sum(1 for line in fh if line.strip())


def check_timestamps_pose_line_count(timestamps_file: Path, pose_file: Path) -> None:
    if not timestamps_file.exists() or not pose_file.exists():
        return

    timestamps_count = count_nonempty_lines(timestamps_file)
    pose_count = count_nonempty_lines(pose_file)
    if timestamps_count != pose_count:
        raise RuntimeError(
            f"TartanEvent/TartanAir frame count mismatch in {pose_file.parent}: "
            f"{timestamps_file.name} has {timestamps_count} lines, "
            f"{pose_file.name} has {pose_count} lines."
        )
    print(f"[OK] {pose_file.parent.name}: {timestamps_file.name} and {pose_file.name} have {timestamps_count} lines.")


def write_cam_time_from_event_timestamps(timestamps_file: Path, cam_time_file: Path) -> None:
    values = []
    with open(timestamps_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                values.append(float(line))

    if not values:
        return

    scale = 1e-9 if max(abs(v) for v in values) > 1e6 else 1.0
    cam_time_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cam_time_file, "w") as fh:
        for value in values:
            fh.write(f"{value * scale:.12f}\n")


def archive_payload_stem(archive_name: str) -> str:
    name = Path(archive_name).name
    if name.endswith(".tar.gz"):
        return name[:-7]
    if name.endswith(".tgz"):
        return name[:-4]
    return Path(name).stem


def prepare_training_layout(
    root: Path,
    env_event: str,
    difficulties: list[str],
    air_archive_name: str,
    keep_air_payload: bool,
) -> None:
    env_dir = root / env_event.lower()
    air_payload_dir_name = archive_payload_stem(air_archive_name)
    for diff in difficulties:
        diff_dir = env_dir / diff.capitalize()
        if not diff_dir.exists():
            continue

        for traj_dir in sorted(p for p in diff_dir.iterdir() if p.is_dir()):
            pose_left = traj_dir / "pose_left.txt"
            pose_lcam_front = traj_dir / "pose_lcam_front.txt"
            timestamps_file = traj_dir / "timestamps.txt"
            pose_source = pose_left if pose_left.exists() else pose_lcam_front
            check_timestamps_pose_line_count(timestamps_file, pose_source)
            if pose_left.exists():
                shutil.copy2(pose_left, pose_lcam_front)

            cam_time_file = traj_dir / "imu" / "cam_time.txt"
            if timestamps_file.exists() and not cam_time_file.exists():
                write_cam_time_from_event_timestamps(timestamps_file, cam_time_file)

            timestamps_file.unlink(missing_ok=True)
            pose_left.unlink(missing_ok=True)
            shutil.rmtree(traj_dir / "flow", ignore_errors=True)

            air_payload_dir = traj_dir / air_payload_dir_name
            if air_payload_dir.exists() and air_payload_dir.is_dir() and not keep_air_payload:
                shutil.rmtree(air_payload_dir)


def canonical_difficulty_name(name: str) -> str | None:
    diff = name.lower().replace("data_", "")
    if diff in {"easy", "hard", "test"}:
        return diff.capitalize()
    return None


def infer_env_and_difficulty(traj_dir: Path) -> tuple[str, str]:
    parts = traj_dir.parts
    for idx in range(len(parts) - 2, 0, -1):
        diff = canonical_difficulty_name(parts[idx])
        if diff is not None:
            return parts[idx - 1].lower(), diff

    return "competition", "Test"


def merge_tartan_test_sources(root: Path, source_roots: list[Path]) -> dict[str, set[str]]:
    traj_pattern = re.compile(r"^(?:P\d{3,4}|M[EH]\d{3,4})$", re.IGNORECASE)
    env_diffs: dict[str, set[str]] = {}

    for source_root in source_roots:
        if not source_root.exists():
            continue

        for traj_dir in sorted(p for p in source_root.rglob("*") if p.is_dir()):
            if not traj_pattern.match(traj_dir.name):
                continue

            env, diff = infer_env_and_difficulty(traj_dir)
            target_traj_dir = root / env / diff / traj_dir.name.upper()
            target_traj_dir.mkdir(parents=True, exist_ok=True)

            if traj_dir.resolve() == target_traj_dir.resolve():
                continue

            move_contents(traj_dir, target_traj_dir)
            env_diffs.setdefault(env, set()).add(diff.lower())

        cleanup_empty_dirs(source_root, source_root)

    return env_diffs


def check_test_version_match(root: Path, env_diffs: dict[str, set[str]]) -> None:
    for env, difficulties in sorted(env_diffs.items()):
        for diff in sorted(difficulties):
            diff_dir = root / env / diff.capitalize()
            if not diff_dir.exists():
                continue

            for traj_dir in sorted(p for p in diff_dir.iterdir() if p.is_dir()):
                timestamps_file = traj_dir / "timestamps.txt"
                pose_file = traj_dir / "pose_left.txt"
                if not pose_file.exists():
                    pose_file = traj_dir / "pose_lcam_front.txt"

                missing = []
                if not (traj_dir / "events.h5").exists():
                    missing.append("events.h5")
                if not timestamps_file.exists():
                    missing.append("timestamps.txt")
                if not pose_file.exists():
                    missing.append("pose_left.txt or pose_lcam_front.txt")

                if missing:
                    raise RuntimeError(
                        f"Test event/Air data mismatch in {traj_dir}: "
                        f"missing {', '.join(missing)}"
                    )

                check_timestamps_pose_line_count(timestamps_file, pose_file)


def remove_test_images(root: Path, env_diffs: dict[str, set[str]]) -> None:
    removed = 0
    for env, difficulties in sorted(env_diffs.items()):
        for diff in sorted(difficulties):
            diff_dir = root / env / diff.capitalize()
            if not diff_dir.exists():
                continue

            for traj_dir in sorted(p for p in diff_dir.iterdir() if p.is_dir()):
                for image_file in traj_dir.rglob("*"):
                    if image_file.is_file() and image_file.suffix.lower() in IMAGE_EXTENSIONS:
                        image_file.unlink()
                        removed += 1

    if removed:
        print(f"Removed {removed} image file(s) from merged test dataset.")


def download_tartan_test(
    root: Path,
    event_url: str,
    air_url: str,
    air_archive_name: str,
    keep_zip: bool,
    keep_air_payload: bool,
) -> None:
    stage_root = root / "_test_download"
    event_stage = stage_root / "event"
    air_stage = stage_root / "air"
    event_archive = root / PurePosixPath(event_url.split("?", 1)[0]).name
    air_archive = root / air_archive_name
    shutil.rmtree(event_stage, ignore_errors=True)
    shutil.rmtree(air_stage, ignore_errors=True)

    print("\n" + "=" * 72)
    print("Preparing TartanEvent competition/test data")
    print("=" * 72)

    print(f"Downloading test events: {event_url}")
    download_archive(event_url, event_archive)
    extract_archive(
        event_archive,
        event_stage,
        delete_archive=not keep_zip,
        skip_extensions=IMAGE_EXTENSIONS,
    )

    print(f"Downloading test Air data: {air_url}")
    download_archive(air_url, air_archive)
    extract_archive(
        air_archive,
        air_stage,
        delete_archive=not keep_zip,
        skip_extensions=None if keep_air_payload else IMAGE_EXTENSIONS,
    )

    env_diffs = merge_tartan_test_sources(root, [event_stage, air_stage])
    if not env_diffs:
        raise RuntimeError("No Pxxx/ME/MH trajectory folders found in downloaded test archives.")

    remove_test_images(root, env_diffs)
    check_test_version_match(root, env_diffs)
    for env, difficulties in sorted(env_diffs.items()):
        prepare_training_layout(
            root=root,
            env_event=env,
            difficulties=sorted(difficulties),
            air_archive_name=air_archive_name,
            keep_air_payload=keep_air_payload,
        )


def move_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            if item.is_dir() and target.is_dir():
                move_contents(item, target)
                try:
                    item.rmdir()
                except OSError:
                    pass
            else:
                print(f"Skipping existing file: {target}")
        else:
            shutil.move(str(item), str(target))


def remove_empty_tree(path: Path, stop_at: Path) -> None:
    while path.exists() and path != stop_at:
        try:
            path.rmdir()
        except OSError:
            break
        path = path.parent


def cleanup_empty_dirs(root_dir: Path, stop_at: Path) -> None:
    if not root_dir.exists():
        return

    subdirs = sorted(
        (path for path in root_dir.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in subdirs:
        remove_empty_tree(path, stop_at)
    remove_empty_tree(root_dir, stop_at)


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique = []
    seen = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path.absolute())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def parse_env_pairs(
    parser: argparse.ArgumentParser,
    env_event: list[str],
    env_air: list[str],
) -> list[tuple[str, str]]:
    if len(env_event) != len(env_air):
        parser.error(
            "--env-event and --env-air must have the same number of values "
            "so each TartanEvent environment can be paired with its TartanAir environment."
        )

    return list(zip(env_event, env_air))


def collect_difficulty_sources(
    source_root: Path,
    env_event: str,
    env_air: str,
    final_diff_name: str,
    air_diff_name: str,
) -> list[Path]:
    candidates = [
        source_root / env_event / final_diff_name,
        source_root / f"{env_event}_events" / final_diff_name,
        source_root / f"{env_event}_events" / env_event / final_diff_name,
        source_root / env_air / air_diff_name,
        source_root / env_air / final_diff_name,
    ]

    event_root = source_root / f"{env_event}_events"
    if event_root.exists():
        for child in event_root.iterdir():
            if child.is_dir():
                candidates.append(child / final_diff_name)

    air_root = source_root / env_air
    if air_root.exists():
        for child in air_root.iterdir():
            if child.is_dir():
                candidates.append(child / air_diff_name)
                candidates.append(child / final_diff_name)

    return [path for path in unique_paths(candidates) if path.exists() and path.is_dir()]


def move_env_level_items(source_root: Path, env_event: str, env_air: str, env_final: Path) -> None:
    env_air_dir = source_root / env_air
    if env_air_dir.exists() and env_air_dir.resolve() != env_final.resolve():
        for item in list(env_air_dir.iterdir()):
            target = env_final / item.name
            if target.exists():
                if item.is_dir() and target.is_dir():
                    move_contents(item, target)
                    remove_empty_tree(item, env_air_dir)
                else:
                    print(f"Skipping existing path: {target}")
            else:
                shutil.move(str(item), str(target))

        remove_empty_tree(env_air_dir, source_root)

    event_root = source_root / f"{env_event}_events"
    if not event_root.exists():
        return

    cleanup_empty_dirs(event_root, source_root)


def normalize_tartanair_layout(
    root: Path,
    env_event: str,
    env_air: str,
    difficulties: list[str],
    merge_roots: list[Path] | None = None,
) -> None:
    import os
    import shutil
    import re

    source_roots = [root]
    if merge_roots:
        source_roots.extend(merge_roots)

    final_env_dir = root / env_event.lower()
    final_env_dir.mkdir(parents=True, exist_ok=True)

    traj_pattern = re.compile(r"^P\d{3,4}$", re.IGNORECASE)
    
    allowed_diffs = [d.lower() for d in difficulties]

    env_keywords = [env_event.lower(), env_air.lower(), f"{env_event.lower()}_events"]

    for source_root in source_roots:
        if not source_root.exists():
            continue

        for root_path, dirs, files in os.walk(source_root):
            current_dir = Path(root_path)

            if traj_pattern.match(current_dir.name):
                path_str = str(current_dir).lower()

                if not any(k in path_str for k in env_keywords):
                    continue

                diff = None
                for d in allowed_diffs:
                    if d in path_str:
                        diff = d.capitalize() 
                        break

                if not diff:
                    continue 

                target_traj_dir = final_env_dir / diff / current_dir.name.upper()
                target_traj_dir.mkdir(parents=True, exist_ok=True)

                if current_dir.resolve() == target_traj_dir.resolve():
                    continue

                for item in current_dir.iterdir():
                    target_item = target_traj_dir / item.name

                    if item.is_file():
                        if not target_item.exists():
                            shutil.move(str(item), str(target_item))
                        elif item.name in {"pose_left.txt", "pose_lcam_front.txt"}:
                            target_item.unlink()
                            shutil.move(str(item), str(target_item))
                    elif item.is_dir():
                        target_item.mkdir(parents=True, exist_ok=True)
                        for sub_item in item.iterdir():
                            target_sub_item = target_item / sub_item.name
                            if not target_sub_item.exists():
                                shutil.move(str(sub_item), str(target_sub_item))
                        try:
                            item.rmdir()
                        except OSError:
                            pass

                try:
                    current_dir.rmdir()
                except OSError:
                    pass

        
        protected_dirs = [ (final_env_dir / d.capitalize()).resolve() for d in allowed_diffs ]
        
        for root_path, dirs, files in os.walk(source_root, topdown=False):
            current_dir = Path(root_path)

            if current_dir == source_root or current_dir == final_env_dir:
                continue
            if current_dir.resolve() in protected_dirs:
                continue

            path_str = str(current_dir).lower()
            if not any(k in path_str for k in env_keywords):
                continue

            try:
                current_dir.rmdir()
            except OSError:
                pass




def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Common root folder")
    parser.add_argument(
        "--env-event",
        type=str,
        nargs="+",
        default=None,
        help='TartanEvent env(s), e.g. "office" or "office carwelding"',
    )
    parser.add_argument(
        "--env-air",
        type=str,
        nargs="+",
        default=None,
        help='TartanAir env(s), e.g. "Office" or "Office CarWelding"',
    )
    parser.add_argument(
        "--difficulty",
        nargs="+",
        default=["easy", "hard"],
        choices=["easy", "hard"],
        help="Difficulties to download",
    )
    parser.add_argument("--skip-event", action="store_true")
    parser.add_argument("--skip-air", action="store_true")
    parser.add_argument("--keep-zip", action="store_true")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Download and merge the TartanEvent competition/test split.",
    )
    parser.add_argument(
        "--test-event-url",
        type=str,
        default=TARTANEVENT_TEST_URL,
        help="TartanEvent competition/test archive URL.",
    )
    parser.add_argument(
        "--test-air-url",
        type=str,
        default=TARTANAIR_TEST_AIR_URL,
        help="Matching TartanAir competition/test archive URL.",
    )
    parser.add_argument(
        "--test-air-archive-name",
        type=str,
        default=TARTANAIR_TEST_AIR_ARCHIVE,
        help="Local archive name for the downloaded TartanAir competition/test data.",
    )
    parser.add_argument(
        "--air-source",
        choices=["huggingface", "direct", "package"],
        default="huggingface",
        help="Download TartanAir from Hugging Face, AirLab, or through the tartanair package.",
    )
    parser.add_argument(
        "--air-file-list",
        type=Path,
        default=TARTANAIR_FILE_LIST,
        help="TartanAir zip list used by the direct downloader.",
    )
    parser.add_argument(
        "--air-bucket",
        type=str,
        default=None,
        help="AirLab S3 bucket used by the direct downloader.",
    )
    parser.add_argument(
        "--air-archive-name",
        type=str,
        default="flow_mask.zip",
        help="Archive name to download from the TartanAir file list. For v1 training, flow_mask.zip is the small archive used to get pose_left.txt.",
    )
    parser.add_argument(
        "--air-workers",
        type=int,
        default=8,
        help="Number of parallel TartanAir direct download workers.",
    )
    parser.add_argument(
        "--keep-air-images",
        action="store_true",
        help="Keep the extracted TartanAir payload folder after extracting pose_left.txt.",
    )
    parser.add_argument(
        "--merge-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional additional partial dataset root(s) to merge into --root during "
            "normalization. Useful when events and IMU were downloaded in separate runs."
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    merge_roots = [path.resolve() for path in args.merge_root]
    if args.test:
        download_tartan_test(
            root=root,
            event_url=args.test_event_url,
            air_url=args.test_air_url,
            air_archive_name=args.test_air_archive_name,
            keep_zip=args.keep_zip,
            keep_air_payload=args.keep_air_images,
        )

    if args.env_event is None and args.env_air is None:
        if args.test:
            print("\nDone.")
            print(f"Unified dataset root: {root}")
            return 0
        parser.error("--env-event and --env-air are required unless --test is used.")
    if args.env_event is None or args.env_air is None:
        parser.error("--env-event and --env-air must be provided together.")

    env_pairs = parse_env_pairs(parser, args.env_event, args.env_air)
    air_bucket = args.air_bucket
    if air_bucket is None:
        air_bucket = (
            "tartanair"
            if args.air_file_list.name == "download_training_zipfiles.txt"
            else "tartanair_v2"
        )

    for env_event, env_air in env_pairs:
        print("\n" + "=" * 72)
        print(f"Preparing Tartan environment pair: event={env_event}, air={env_air}")
        print("=" * 72)

        if not args.skip_air:
            if args.air_source == "huggingface":
                download_tartanair_huggingface(
                    root=root,
                    env=env_air,
                    difficulties=args.difficulty,
                    file_list=args.air_file_list,
                    archive_name=args.air_archive_name,
                    delete_zip=not args.keep_zip,
                )
            elif args.air_source == "direct":
                download_tartanair_direct(
                    root=root,
                    env=env_air,
                    difficulties=args.difficulty,
                    file_list=args.air_file_list,
                    bucket_name=air_bucket,
                    archive_name=args.air_archive_name,
                    workers=args.air_workers,
                    delete_zip=not args.keep_zip,
                )
            else:
                download_tartanair_imu(
                    root=root,
                    env=env_air,
                    difficulties=args.difficulty,
                )

        if not args.skip_event:
            download_tartanevent(
                root=root,
                env_zip=env_event,
                unzip=True,
                delete_zip=not args.keep_zip,
            )

        normalize_tartanair_layout(
            root=root,
            env_event=env_event,
            env_air=env_air,
            difficulties=args.difficulty,
            merge_roots=merge_roots,
        )
        prepare_training_layout(
            root=root,
            env_event=env_event,
            difficulties=args.difficulty,
            air_archive_name=args.air_archive_name,
            keep_air_payload=args.keep_air_images,
        )

    print("\nDone.")
    print("Unified dataset roots:")
    for env_event, _ in env_pairs:
        print(f"  - {root / env_event.lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
