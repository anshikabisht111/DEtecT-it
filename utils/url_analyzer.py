"""
utils/url_analyzer.py — URL/Link Analysis Module

Given a URL, this module:
  1. Validates it (basic SSRF guard against private/internal IPs)
  2. If it's a direct media link, downloads it
  3. If it's a webpage, fetches it and extracts the first usable image
  4. Runs basic domain reputation checks (WHOIS age, SSL cert validity)

It does NOT run deepfake analysis itself — app.py downloads the media via
this module, then routes it through the SAME analysis pipeline used for
direct file uploads (process_image() in app.py), so there's one source of
truth for the actual detection logic.

Dependencies to add to requirements.txt:
    requests
    beautifulsoup4
    python-whois
    validators
"""

import os
import ipaddress
import socket
import mimetypes
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import requests
import validators
from bs4 import BeautifulSoup

try:
    import whois
except ImportError:
    whois = None

REQUEST_TIMEOUT = 10
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
USER_AGENT = "DEtecT-it-URLAnalyzer/1.0 (+forensic-research-tool)"


def validate_url(url: str):
    """Returns (is_valid, error_message)."""
    if not url or not validators.url(url):
        return False, "Invalid URL format."

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, "Only http/https URLs are supported."

    try:
        host = parsed.hostname
        ip = socket.gethostbyname(host)
        if ipaddress.ip_address(ip).is_private:
            return False, "Private/internal addresses are not allowed."
    except Exception:
        pass  # if resolution fails, let the request itself fail later

    return True, ""


def is_direct_image_link(url: str) -> bool:
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    return ext in ALLOWED_IMAGE_EXT


def fetch_page(url: str) -> requests.Response:
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp


def extract_first_image_url(page_url: str, html: str) -> str | None:
    """Pulls the best candidate image out of a page's HTML (og:image preferred)."""
    soup = BeautifulSoup(html, "html.parser")

    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        return urljoin(page_url, og_image["content"])

    img = soup.find("img", src=True)
    if img:
        return urljoin(page_url, img["src"])

    return None


def download_media(url: str, save_dir) -> tuple:
    """Downloads a media file to save_dir. Returns (local_path, filename) or (None, None)."""
    try:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ""
        if not ext or ext == ".jpe":
            ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"

        filename = f"url_fetch_{abs(hash(url)) % 100000}{ext}"
        local_path = os.path.join(str(save_dir), filename)

        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return local_path, filename
    except Exception as e:
        print(f"[url_analyzer] download_media failed for {url}: {e}")
        return None, None


def check_domain_age(url: str) -> dict:
    if whois is None:
        return {"available": False, "reason": "python-whois not installed"}

    domain = urlparse(url).netloc
    try:
        w = whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if not creation:
            return {"available": False, "reason": "No creation date returned"}
        if creation.tzinfo is None:
            creation = creation.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - creation).days
        return {
            "available": True,
            "domain": domain,
            "age_days": age_days,
            "flag": "new_domain" if age_days < 90 else "established",
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def fetch_and_extract_image(url: str, save_dir):
    """
    Main entry point for app.py. Given a URL, returns:
        (local_path, filename, error_message)
    error_message is None on success.
    """
    valid, err = validate_url(url)
    if not valid:
        return None, None, err

    try:
        if is_direct_image_link(url):
            local_path, filename = download_media(url, save_dir)
            if not local_path:
                return None, None, "Could not download the image from that URL."
            return local_path, filename, None

        resp = fetch_page(url)
        image_url = extract_first_image_url(url, resp.text)
        if not image_url:
            return None, None, "No usable image found on that page."

        local_path, filename = download_media(image_url, save_dir)
        if not local_path:
            return None, None, "Found an image on the page but couldn't download it."

        return local_path, filename, None

    except requests.RequestException as e:
        return None, None, f"Failed to fetch URL: {e}"
