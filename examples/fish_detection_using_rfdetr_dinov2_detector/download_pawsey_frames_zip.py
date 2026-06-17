#!/usr/bin/env python
"""Download the Pawsey FDFML frames.zip archive."""

import argparse
import os
from pathlib import Path
import ssl
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_URL = "https://storage.pawsey.org.au/public/m/FDFML/frames.zip"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "datasets" / "OzFish"
DEFAULT_FILENAME = "frames.zip"
USER_AGENT = "paz-fdfml-frames-zip-downloader/1.0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download frames.zip from the public Pawsey FDFML store."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where frames.zip will be saved.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Full output path. Overrides --output-dir.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download again even if the output file already exists.",
    )
    parser.add_argument(
        "--ca-bundle",
        type=Path,
        default=None,
        help="Path to a CA certificate bundle for HTTPS verification.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable HTTPS certificate verification.",
    )
    return parser.parse_args()


def make_ssl_context(args):
    if args.insecure:
        return ssl._create_unverified_context()
    if args.ca_bundle is not None:
        return ssl.create_default_context(cafile=str(args.ca_bundle))
    return None


def request_url(url, timeout, ssl_context=None, headers=None, method="GET"):
    all_headers = {"User-Agent": USER_AGENT}
    if headers:
        all_headers.update(headers)
    request = Request(url, headers=all_headers, method=method)
    return urlopen(request, timeout=timeout, context=ssl_context)


def remote_size(url, timeout, ssl_context=None):
    try:
        with request_url(url, timeout, ssl_context, method="HEAD") as response:
            size = response.headers.get("Content-Length")
            accept_ranges = response.headers.get("Accept-Ranges", "")
    except (HTTPError, URLError):
        return None, False
    return int(size) if size and size.isdigit() else None, "bytes" in accept_ranges.lower()


def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def download(url, output_path, timeout, retries, overwrite, ssl_context=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".part")
    expected_size, supports_ranges = remote_size(url, timeout, ssl_context)

    if output_path.exists() and not overwrite:
        if expected_size is None or output_path.stat().st_size == expected_size:
            print(f"SKIP existing file: {output_path}", flush=True)
            return "skipped"

    for attempt in range(1, retries + 1):
        try:
            resume_from = 0
            mode = "wb"
            headers = {}
            if partial.exists() and supports_ranges and not overwrite:
                resume_from = partial.stat().st_size
                if expected_size is None or resume_from < expected_size:
                    headers["Range"] = f"bytes={resume_from}-"
                    mode = "ab"

            with request_url(
                url, timeout, ssl_context, headers=headers
            ) as response, partial.open(mode) as f:
                downloaded = resume_from
                last_report = time.time()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_report >= 10:
                        if expected_size:
                            pct = downloaded * 100.0 / expected_size
                            print(
                                f"  {format_bytes(downloaded)} / "
                                f"{format_bytes(expected_size)} ({pct:.1f}%)",
                                flush=True,
                            )
                        else:
                            print(f"  {format_bytes(downloaded)}", flush=True)
                        last_report = now

            if expected_size is not None and partial.stat().st_size != expected_size:
                raise IOError(
                    f"incomplete download: got {partial.stat().st_size}, "
                    f"expected {expected_size}"
                )
            os.replace(partial, output_path)
            print(f"OK: {output_path}", flush=True)
            return "downloaded"
        except Exception as exc:
            if overwrite and partial.exists():
                partial.unlink()
            if attempt >= retries:
                print(f"FAIL: {exc}", file=sys.stderr, flush=True)
                return "failed"
            wait = min(2 ** attempt, 30)
            print(
                f"Retry {attempt}/{retries} after error: {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
    return "failed"


def main():
    args = parse_args()
    if args.retries <= 0:
        raise ValueError("--retries must be > 0")
    ssl_context = make_ssl_context(args)
    if args.insecure:
        print("WARNING: HTTPS certificate verification is disabled.", flush=True)

    output_path = (
        args.output_file.expanduser().resolve()
        if args.output_file is not None
        else args.output_dir.expanduser().resolve() / DEFAULT_FILENAME
    )
    print(f"URL        : {args.url}", flush=True)
    print(f"Output file: {output_path}", flush=True)
    status = download(
        args.url,
        output_path,
        timeout=args.timeout,
        retries=args.retries,
        overwrite=args.overwrite,
        ssl_context=ssl_context,
    )
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
