#!/usr/bin/env python3
"""
Merge discovered URLs with original seeds and re-verify everything.
Also crawl the newly discovered sites for more Kurdish links.
"""

import csv
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ku,ckb,en;q=0.5",
}

TIMEOUT = 12

# ─────────────────────────────────────────────────────────────────
# Language Detection (same as verify_and_export.py)
# ─────────────────────────────────────────────────────────────────

CKB_CHARS = set("ەێڕۆڵ")
CKB_KEYWORDS = [
    "هەواڵ", "کوردی", "کوردستان", "بەپەلە", "مووچە", "دەستوور",
    "پارلەمان", "حکوومەت", "ئەنجومەن", "زانکۆ", "فێرکردن",
    "بەشداری", "دامەزراوە", "وەزارەت", "بەڕێوەبردن", "تەندروستی",
    "پەروەردە", "ئابووری", "نەوت", "سیاسی", "کۆمەڵگا",
    "رۆژنامەوانی", "ئازادی", "مافی", "دادوەری", "هەڵبژاردن",
    "ئێمە", "ئەوان", "دەبێت", "چاوەڕوانی", "بەرپرسیار",
]

KMR_CHARS = set("êîûşçÊÎÛŞÇ")
KMR_KEYWORDS = [
    "Kurdî", "Kurdistanê", "Rojava", "Bakur", "Başûr", "azadî",
    "nûçe", "welatê", "gelê", "mafê", "demokrasî", "civakî",
    "perwerdehî", "tenduristî", "aborî", "siyaset", "hiqûq",
    "jin", "jiyan", "azadiya", "berxwedanê", "Hewlêr", "Diyarbekir",
    "Kobanê", "Efrîn", "Qamişlo", "Duhok", "Silêmanî",
    "pêşmerge", "newroz", "welat", "nûçeyên", "nûçe",
    "çand", "wêje", "muzîk", "hunermend", "helbestvan",
    "xwendevan", "mamosta", "dibistan",
]

ZZA_KEYWORDS = [
    "Zazaki", "Dimilî", "Kirmancki", "Dersim", "Tunceli",
    "Bingöl", "Pülümür", "Hozat", "Mazgirt", "Ovacık",
    "zazaca", "dimili", "kirmanckî",
]

HAW_KEYWORDS = [
    "هەورامی", "گۆرانی", "Hawrami", "Gorani", "Hewraman",
    "Halabja", "Nawsud", "Byara", "Tawela", "Xurmal",
    "hawramani", "goranî",
]

EN_KEYWORDS = [
    "the", "and", "for", "that", "with", "this", "from",
    "have", "been", "are", "was", "were", "about",
]

AR_KEYWORDS = [
    "في", "من", "على", "إلى", "عن", "هذا", "التي", "الذي",
    "أن", "كان", "عربي", "أخبار", "جديد",
]

TR_KEYWORDS = [
    "bir", "için", "olan", "ile", "gibi", "daha", "ancak",
    "ama", "çok", "haber", "Türkiye", "olan", "olarak",
]

FA_KEYWORDS = [
    "است", "این", "برای", "های", "ایران", "فارسی",
    "خبر", "تهران", "دولت", "جمهوری", "اسلامی",
]


def detect_language(html_content):
    """Detect language from HTML content."""
    if not html_content:
        return "UNK"

    text = html_content
    scores = {
        "CKB": 0, "KMR": 0, "ZZA": 0, "HAW": 0,
        "EN": 0, "AR": 0, "TR": 0, "FA": 0,
    }

    # Check HTML lang attribute
    lang_match = re.search(r'<html[^>]*\slang=["\']([^"\']+)', text, re.I)
    if lang_match:
        lang_val = lang_match.group(1).lower()
        if lang_val in ("ckb", "ku-arab", "ku-iq"):
            scores["CKB"] += 30
        elif lang_val in ("ku", "kmr", "ku-latn", "ku-tr"):
            scores["KMR"] += 30
        elif lang_val.startswith("ar"):
            scores["AR"] += 20
        elif lang_val.startswith("tr"):
            scores["TR"] += 20
        elif lang_val.startswith("fa"):
            scores["FA"] += 20
        elif lang_val.startswith("en"):
            scores["EN"] += 20

    # CKB: Check for unique Kurdish Arabic-script characters
    ckb_chars_found = sum(1 for c in text if c in CKB_CHARS)
    scores["CKB"] += min(ckb_chars_found * 0.05, 30)

    # KMR: Check for Kurdish Latin characters
    kmr_chars_found = sum(1 for c in text if c in KMR_CHARS)
    scores["KMR"] += min(kmr_chars_found * 0.03, 20)

    text_lower = text.lower()

    # Keyword matching
    for kw in CKB_KEYWORDS:
        if kw in text:
            scores["CKB"] += 3
    for kw in KMR_KEYWORDS:
        if kw.lower() in text_lower:
            scores["KMR"] += 3
    for kw in ZZA_KEYWORDS:
        if kw.lower() in text_lower:
            scores["ZZA"] += 4
    for kw in HAW_KEYWORDS:
        if kw.lower() in text_lower or kw in text:
            scores["HAW"] += 4
    for kw in EN_KEYWORDS:
        cnt = text_lower.count(" " + kw + " ")
        scores["EN"] += min(cnt * 0.3, 10)
    for kw in AR_KEYWORDS:
        if kw in text:
            scores["AR"] += 2
    for kw in TR_KEYWORDS:
        if kw.lower() in text_lower:
            scores["TR"] += 2
    for kw in FA_KEYWORDS:
        if kw in text:
            scores["FA"] += 2

    max_score = max(scores.values())
    if max_score < 3:
        return "UNK"

    top_langs = [lang for lang, sc in scores.items() if sc == max_score]

    # Also include secondary languages above 40% threshold
    result = []
    for lang, sc in sorted(scores.items(), key=lambda x: -x[1]):
        if sc >= max_score * 0.4 and sc >= 3:
            result.append(lang)
        if len(result) >= 3:
            break

    return "/".join(result) if result else "UNK"


def extract_title(html):
    """Extract title from HTML."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        tag = soup.find("title")
        if tag and tag.string:
            return tag.string.strip()[:200]
    except Exception:
        pass
    return ""


def verify_url(url, source="discovered", category="discovered", notes=""):
    """Verify a URL and return result dict."""
    domain = urlparse(url).hostname or ""
    domain = domain.lower().lstrip("www.")

    result = {
        "url": url,
        "domain": domain,
        "title": "",
        "language": "UNK",
        "status_code": "ERR",
        "source": source,
        "category": category,
        "notes": notes,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                            allow_redirects=True, verify=True)
        result["status_code"] = str(resp.status_code)
        if resp.status_code < 400:
            content = resp.text[:100000]
            result["title"] = extract_title(content)
            result["language"] = detect_language(content)
    except requests.exceptions.SSLError:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                                allow_redirects=True, verify=False)
            result["status_code"] = str(resp.status_code)
            result["notes"] = (notes + "; SSL warning").strip("; ")
            if resp.status_code < 400:
                content = resp.text[:100000]
                result["title"] = extract_title(content)
                result["language"] = detect_language(content)
        except Exception:
            result["status_code"] = "SSL_ERR"
    except requests.exceptions.ConnectionError:
        result["status_code"] = "CONN_ERR"
    except requests.exceptions.Timeout:
        result["status_code"] = "TIMEOUT"
    except Exception as e:
        result["status_code"] = f"ERR:{type(e).__name__}"

    return result


def main():
    base_dir = os.path.dirname(__file__)
    original_csv = os.path.join(base_dir, "kurdish_websites.csv")
    discovered_txt = os.path.join(base_dir, "discovered_urls.txt")
    output_csv = os.path.join(base_dir, "kurdish_websites_full.csv")

    # Load original verified results
    original_results = []
    known_domains = set()
    if os.path.exists(original_csv):
        with open(original_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                original_results.append(row)
                known_domains.add(row.get("domain", "").lower())
    print(f"[INFO] {len(original_results)} entries from original CSV ({len(known_domains)} domains).")

    # Load discovered URLs
    new_urls = []
    if os.path.exists(discovered_txt):
        with open(discovered_txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    url, domain = parts[0], parts[1]
                    if domain not in known_domains:
                        new_urls.append(url)
                        known_domains.add(domain)
    print(f"[INFO] {len(new_urls)} new URLs to verify from discovered list.")

    # Verify new URLs
    new_results = []
    if new_urls:
        print(f"\n[VERIFYING] {len(new_urls)} new URLs with language detection...")
        with ThreadPoolExecutor(max_workers=50) as pool:
            futures = {pool.submit(verify_url, url): url for url in new_urls}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Verifying new URLs"):
                result = future.result()
                new_results.append(result)

    # Combine all results
    all_results = original_results + new_results
    print(f"\n[TOTAL] {len(all_results)} total entries.")

    # Deduplicate by domain
    seen_domains = set()
    deduped = []
    for row in all_results:
        d = row.get("domain", "").lower()
        if d and d not in seen_domains:
            seen_domains.add(d)
            deduped.append(row)
    print(f"[DEDUP] {len(deduped)} unique domains.")

    # Write combined CSV
    fieldnames = ["url", "domain", "title", "language", "status_code",
                  "source", "category", "notes", "checked_at"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(deduped, key=lambda x: x.get("domain", "")):
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\n[CSV] Wrote {len(deduped)} rows to {output_csv}")

    # Summary statistics
    reachable = sum(1 for r in deduped if r.get("status_code", "").startswith("2"))
    lang_counts = defaultdict(int)
    cat_counts = defaultdict(int)
    for r in deduped:
        lang_counts[r.get("language", "UNK")] += 1
        cat_counts[r.get("category", "unknown")] += 1

    print(f"\n--- Summary ---")
    print(f"  Total unique domains: {len(deduped)}")
    print(f"  Reachable (2xx):      {reachable}")
    print(f"\n  Language distribution:")
    for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"    {lang:>8}: {cnt}")
    print(f"\n  Category distribution:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:>20}: {cnt}")


if __name__ == "__main__":
    main()
