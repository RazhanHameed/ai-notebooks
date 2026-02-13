#!/usr/bin/env python3
"""
Final merge: combine all sources, verify everything, produce final CSV.
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

# ─── Language Detection ───

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
    "pêşmerge", "newroz", "welat", "nûçeyên",
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
    "أن", "كان", "عربي", "أخبار",
]

TR_KEYWORDS = [
    "bir", "için", "olan", "ile", "gibi", "daha", "ancak",
    "ama", "çok", "haber", "Türkiye",
]

FA_KEYWORDS = [
    "است", "این", "برای", "های", "ایران", "فارسی",
    "خبر", "تهران", "دولت",
]


def detect_language(html_content):
    if not html_content:
        return "UNK"
    text = html_content
    scores = {"CKB": 0, "KMR": 0, "ZZA": 0, "HAW": 0,
              "EN": 0, "AR": 0, "TR": 0, "FA": 0}

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

    ckb_chars_found = sum(1 for c in text if c in CKB_CHARS)
    scores["CKB"] += min(ckb_chars_found * 0.05, 30)

    kmr_chars_found = sum(1 for c in text if c in KMR_CHARS)
    scores["KMR"] += min(kmr_chars_found * 0.03, 20)

    text_lower = text.lower()

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

    result = []
    for lang, sc in sorted(scores.items(), key=lambda x: -x[1]):
        if sc >= max_score * 0.4 and sc >= 3:
            result.append(lang)
        if len(result) >= 3:
            break

    return "/".join(result) if result else "UNK"


def extract_title(html):
    try:
        soup = BeautifulSoup(html, "html.parser")
        tag = soup.find("title")
        if tag and tag.string:
            return tag.string.strip()[:200]
    except Exception:
        pass
    return ""


def categorize_domain(domain):
    """Attempt to categorize a domain based on its name."""
    d = domain.lower()
    if any(x in d for x in [".edu.", "university", "zanko", "college"]):
        return "education"
    if any(x in d for x in [".gov.", "government", "parliament", "presidency", "ministry"]):
        return "government"
    if any(x in d for x in ["news", "press", "media", "rudaw", "nrt", "hawar", "nuce"]):
        return "news_media"
    if any(x in d for x in ["tv", "radio", "fm", "sat"]):
        return "tv_radio"
    if any(x in d for x in ["hospital", "health", "medical", "clinic"]):
        return "health"
    if any(x in d for x in ["sport", "football", "fc", "marathon"]):
        return "sports"
    if any(x in d for x in ["shop", "store", "mall", "bazar", "market"]):
        return "ecommerce"
    if any(x in d for x in ["ngo", "rights", "humanitarian"]):
        return "ngo"
    if any(x in d for x in ["yezidi", "ezidi", "lalish", "islam", "christian", "church"]):
        return "religious"
    if any(x in d for x in ["tech", "dev", "code", "software"]):
        return "technology"
    if any(x in d for x in ["travel", "tour", "visit", "hotel"]):
        return "tourism"
    if any(x in d for x in ["blog", "forum", "community"]):
        return "blogs"
    if any(x in d for x in ["kdp", "puk", "gorran", "party", "komala"]):
        return "political"
    return "general"


def verify_url(url, existing_data=None):
    """Full verification of a URL."""
    domain = urlparse(url).hostname or ""
    domain = domain.lower()
    if domain.startswith("www."):
        domain = domain[4:]

    result = {
        "url": url,
        "domain": domain,
        "title": "",
        "language": "UNK",
        "status_code": "ERR",
        "source": existing_data.get("source", "discovered") if existing_data else "discovered",
        "category": existing_data.get("category", categorize_domain(domain)) if existing_data else categorize_domain(domain),
        "notes": existing_data.get("notes", "") if existing_data else "",
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
            if resp.status_code < 400:
                content = resp.text[:100000]
                result["title"] = extract_title(content)
                result["language"] = detect_language(content)
                result["notes"] = (result["notes"] + "; SSL warning").strip("; ")
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
    full_csv = os.path.join(base_dir, "kurdish_websites_full.csv")
    deep_discovered = os.path.join(base_dir, "deep_discovered.txt")
    output_csv = os.path.join(base_dir, "kurdish_websites_final.csv")

    # Load existing data
    existing = {}
    if os.path.exists(full_csv):
        with open(full_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = row.get("domain", "").lower()
                existing[d] = row

    print(f"[INFO] {len(existing)} existing entries from full CSV.")

    # Load deep discovered
    deep_urls = []
    if os.path.exists(deep_discovered):
        with open(deep_discovered, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    url, domain = parts[0], parts[1]
                    if domain not in existing:
                        source = parts[3] if len(parts) > 3 else "discovered"
                        deep_urls.append((url, domain, source))

    print(f"[INFO] {len(deep_urls)} new URLs from deep discovery.")

    # Verify deep discovered URLs
    new_results = []
    if deep_urls:
        print(f"\n[VERIFYING] {len(deep_urls)} new URLs...")
        with ThreadPoolExecutor(max_workers=50) as pool:
            futures = {}
            for url, domain, source in deep_urls:
                f = pool.submit(verify_url, url, {"source": source, "category": categorize_domain(domain), "notes": ""})
                futures[f] = domain
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Verifying"):
                result = future.result()
                new_results.append(result)

    # Combine existing + new
    all_data = {}
    for d, row in existing.items():
        all_data[d] = row
    for result in new_results:
        d = result["domain"]
        if d not in all_data:
            all_data[d] = result

    print(f"\n[TOTAL] {len(all_data)} unique domains.")

    # Write final CSV
    fieldnames = ["url", "domain", "title", "language", "status_code",
                  "source", "category", "notes", "checked_at"]
    rows = sorted(all_data.values(), key=lambda x: x.get("domain", ""))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"[CSV] Wrote {len(rows)} rows to {output_csv}")

    # Summary
    reachable = sum(1 for r in rows if r.get("status_code", "").startswith("2"))
    lang_counts = defaultdict(int)
    cat_counts = defaultdict(int)
    status_counts = defaultdict(int)

    for r in rows:
        lang = r.get("language", "UNK")
        # Simplify multi-language labels
        if "/" in lang:
            primary = lang.split("/")[0]
            lang_counts[primary] += 1
        elif "," in lang:
            primary = lang.split(",")[0].strip()
            lang_counts[primary] += 1
        else:
            lang_counts[lang] += 1
        cat_counts[r.get("category", "unknown")] += 1
        sc = r.get("status_code", "ERR")
        if sc.startswith("2"):
            status_counts["2xx"] += 1
        elif sc.startswith("3"):
            status_counts["3xx"] += 1
        elif sc.startswith("4"):
            status_counts["4xx"] += 1
        elif sc.startswith("5"):
            status_counts["5xx"] += 1
        else:
            status_counts["error"] += 1

    print(f"\n{'='*60}")
    print(f"  KURDISH WEBSITES FINAL REPORT")
    print(f"{'='*60}")
    print(f"  Total unique domains:  {len(rows)}")
    print(f"  Reachable (2xx):       {reachable}")
    print(f"  Redirects (3xx):       {status_counts.get('3xx', 0)}")
    print(f"  Client errors (4xx):   {status_counts.get('4xx', 0)}")
    print(f"  Server errors (5xx):   {status_counts.get('5xx', 0)}")
    print(f"  Connection errors:     {status_counts.get('error', 0)}")

    print(f"\n  --- Language Distribution (primary) ---")
    for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1])[:15]:
        bar = "#" * min(cnt, 50)
        print(f"    {lang:>6}: {cnt:>4}  {bar}")

    print(f"\n  --- Category Distribution ---")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(cnt, 50)
        print(f"    {cat:>20}: {cnt:>4}  {bar}")

    # Kurdish-specific stats
    ckb_count = sum(1 for r in rows if "CKB" in r.get("language", ""))
    kmr_count = sum(1 for r in rows if "KMR" in r.get("language", ""))
    zza_count = sum(1 for r in rows if "ZZA" in r.get("language", ""))
    haw_count = sum(1 for r in rows if "HAW" in r.get("language", ""))
    print(f"\n  --- Kurdish Language Breakdown ---")
    print(f"    Central Kurdish (CKB):  {ckb_count}")
    print(f"    Northern Kurdish (KMR): {kmr_count}")
    print(f"    Zazaki (ZZA):           {zza_count}")
    print(f"    Hawrami/Gorani (HAW):   {haw_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
