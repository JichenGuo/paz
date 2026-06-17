#!/usr/bin/env python
"""Download image frames from the public Pawsey FDFML frame store."""

import argparse
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import ssl
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urljoin, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


DEFAULT_URL = "https://storage.pawsey.org.au/public/m/FDFML/frames/"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "datasets" / "FDFML_frames"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
USER_AGENT = "paz-fdfml-frame-downloader/1.0"


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        attrs = dict(attrs)
        href = attrs.get("href")
        if href:
            self.links.append(href)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download images from the public Pawsey FDFML frames directory. "
            "Existing files with the expected size are skipped."
        )
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Source directory/listing URL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Local folder where images will be saved.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Follow subdirectories found in the listing.",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(IMAGE_EXTENSIONS)),
        help="Comma-separated image extensions to download.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images to download. 0 means no limit.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Download attempts per file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List images without downloading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download again even when the destination file already exists.",
    )
    parser.add_argument(
        "--ca-bundle",
        type=Path,
        default=None,
        help="Path to a CA certificate bundle to use for HTTPS verification.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help=(
            "Disable HTTPS certificate verification. Use only when the runtime "
            "container lacks CA certificates and you trust the network."
        ),
    )
    return parser.parse_args()


def make_ssl_context(args):
    if args.insecure:
        return ssl._create_unverified_context()
    if args.ca_bundle is not None:
        return ssl.create_default_context(cafile=str(args.ca_bundle))
    return None


def request_url(url, timeout, ssl_context=None):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    return urlopen(request, timeout=timeout, context=ssl_context)


def read_listing(url, timeout, ssl_context=None):
    with request_url(url, timeout, ssl_context) as response:
        content_type = response.headers.get("Content-Type", "")
        body = response.read()
    text = body.decode("utf-8", errors="replace")
    return text, content_type


def parse_html_links(base_url, text):
    parser = LinkParser()
    parser.feed(text)
    links = []
    for href in parser.links:
        if href.startswith(("#", "?", "mailto:")):
            continue
        links.append(urljoin(base_url, href))
    return links



def object_store_listing_url(url):
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    bucket = parts[0]
    prefix = "/".join(parts[1:]).rstrip("/") + "/"
    return f"{parsed.scheme}://{parsed.netloc}/{bucket}?prefix={quote(prefix)}"


def listing_candidates(root_url):
    candidates = []
    if "?" in root_url:
        candidates.append(root_url)
    else:
        candidates.append(root_url.rstrip("/") + "/")
        candidates.append(root_url.rstrip("/"))
        object_listing = object_store_listing_url(root_url)
        if object_listing is not None:
            candidates.append(object_listing)
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique

def parse_xml_links(base_url, text):
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []

    links = []
    parsed = urlparse(base_url)
    base_prefix = parsed.path.rstrip("/") + "/"
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0] + "}"

    bucket = parsed.path.strip("/").split("/", 1)[0]
    bucket_prefix = f"/{bucket}/" if bucket else "/"

    for key in root.findall(f".//{namespace}Key"):
        if key.text:
            path = key.text.lstrip("/")
            links.append(f"{parsed.scheme}://{parsed.netloc}{bucket_prefix}{path}")

    for prefix in root.findall(f".//{namespace}Prefix"):
        if prefix.text:
            path = prefix.text.lstrip("/")
            links.append(f"{parsed.scheme}://{parsed.netloc}{bucket_prefix}{path}")

    if not links and base_prefix:
        for element in root.iter():
            if element.text and element.text.startswith(base_prefix):
                links.append(f"{parsed.scheme}://{parsed.netloc}{element.text}")
    return links


def is_directory_url(url):
    path = urlparse(url).path
    return path.endswith("/")


def is_image_url(url, extensions):
    path = unquote(urlparse(url).path)
    return Path(path).suffix.lower() in extensions


def collect_image_urls(root_url, extensions, recursive, timeout, ssl_context=None):
    seen_dirs = set()
    seen_images = set()
    pending = listing_candidates(root_url)
    images = []

    while pending:
        listing_url = pending.pop(0)
        if listing_url in seen_dirs:
            continue
        seen_dirs.add(listing_url)

        print(f"Reading listing: {listing_url}", flush=True)
        try:
            text, content_type = read_listing(listing_url, timeout, ssl_context)
        except HTTPError as exc:
            if exc.code == 404:
                print(f"Listing not found, trying next form: {listing_url}", flush=True)
                continue
            raise
        links = parse_xml_links(listing_url, text)
        if not links:
            links = parse_html_links(listing_url, text)

        for link in links:
            link = link.split("#", 1)[0]
            if not link or link in (listing_url, "../"):
                continue
            if is_image_url(link, extensions):
                if link not in seen_images:
                    seen_images.add(link)
                    images.append(link)
            elif recursive and is_directory_url(link):
                if link not in seen_dirs:
                    pending.append(link)

    return images


def destination_for_url(url, root_url, output_dir):
    root_path = urlparse(root_url.rstrip("/") + "/").path.rstrip("/") + "/"
    path = unquote(urlparse(url).path)
    if path.startswith(root_path):
        relative = path[len(root_path):]
    else:
        relative = Path(path).name
    relative = relative.lstrip("/")
    return output_dir / relative


def remote_size(url, timeout, ssl_context=None):
    request = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout, context=ssl_context) as response:
            size = response.headers.get("Content-Length")
    except (HTTPError, URLError):
        return None
    return int(size) if size and size.isdigit() else None


def should_skip(destination, url, timeout, overwrite, ssl_context=None):
    if overwrite or not destination.exists():
        return False
    size = remote_size(url, timeout, ssl_context)
    if size is None:
        return destination.stat().st_size > 0
    return destination.stat().st_size == size


def download_file(url, destination, timeout, retries, overwrite, ssl_context=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if should_skip(destination, url, timeout, overwrite, ssl_context):
        print(f"SKIP {destination}", flush=True)
        return "skipped"

    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with request_url(url, timeout, ssl_context) as response, partial.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(partial, destination)
            print(f"OK   {destination}", flush=True)
            return "downloaded"
        except Exception as exc:
            if partial.exists():
                partial.unlink()
            if attempt >= retries:
                print(f"FAIL {url}: {exc}", file=sys.stderr, flush=True)
                return "failed"
            wait = min(2 ** attempt, 30)
            print(
                f"Retry {attempt}/{retries} after error for {url}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
    return "failed"


def write_manifest(output_dir, root_url, image_urls, status_counts):
    manifest = {
        "source_url": root_url,
        "num_images_found": len(image_urls),
        "status_counts": status_counts,
        "images": image_urls,
    }
    manifest_path = output_dir / "download_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}", flush=True)


def main():
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.retries <= 0:
        raise ValueError("--retries must be > 0")

    extensions = {
        ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
        for ext in args.extensions.split(",")
        if ext.strip()
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ssl_context = make_ssl_context(args)
    if args.insecure:
        print("WARNING: HTTPS certificate verification is disabled.", flush=True)

    image_urls = collect_image_urls(
        args.url,
        extensions=extensions,
        recursive=args.recursive,
        timeout=args.timeout,
        ssl_context=ssl_context,
    )
    if args.limit:
        image_urls = image_urls[:args.limit]

    print(f"Images found: {len(image_urls)}", flush=True)
    print(f"Output dir  : {output_dir}", flush=True)

    status_counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    if args.dry_run:
        for url in image_urls:
            print(url)
        write_manifest(output_dir, args.url, image_urls, status_counts)
        return

    for index, url in enumerate(image_urls, start=1):
        destination = destination_for_url(url, args.url, output_dir)
        print(f"[{index}/{len(image_urls)}] {url}", flush=True)
        status = download_file(
            url,
            destination,
            timeout=args.timeout,
            retries=args.retries,
            overwrite=args.overwrite,
            ssl_context=ssl_context,
        )
        status_counts[status] += 1

    write_manifest(output_dir, args.url, image_urls, status_counts)
    print(f"Done: {status_counts}", flush=True)


if __name__ == "__main__":
    main()
