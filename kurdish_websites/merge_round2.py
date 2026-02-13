#!/usr/bin/env python3
"""
Merge round2 discovered URLs into the complete dataset.
Verify new URLs with full language detection and produce updated CSV.
"""

import csv
import os
import re
import sys
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

ZZA_KEYWORDS = ["Zazaki", "Dimilî", "Kirmancki", "Dersim", "Tunceli",
                "Bingöl", "Pülümür", "Hozat", "Mazgirt"]
HAW_KEYWORDS = ["هەورامی", "گۆرانی", "Hawrami", "Gorani", "Hewraman",
                "Halabja", "Nawsud", "Byara"]
EN_KEYWORDS = ["the", "and", "for", "that", "with", "this", "from",
               "have", "been", "are", "was", "were", "about"]
AR_KEYWORDS = ["في", "من", "على", "إلى", "عن", "هذا", "التي",
               "أن", "كان", "عربي", "أخبار"]
TR_KEYWORDS = ["bir", "için", "olan", "ile", "gibi", "daha", "ancak",
               "ama", "çok", "haber", "Türkiye"]
FA_KEYWORDS = ["است", "این", "برای", "های", "ایران", "فارسی",
               "خبر", "تهران", "دولت"]


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
        elif lang_val.startswith("ar"): scores["AR"] += 20
        elif lang_val.startswith("tr"): scores["TR"] += 20
        elif lang_val.startswith("fa"): scores["FA"] += 20
        elif lang_val.startswith("en"): scores["EN"] += 20

    ckb_chars_found = sum(1 for c in text if c in CKB_CHARS)
    scores["CKB"] += min(ckb_chars_found * 0.05, 30)
    kmr_chars_found = sum(1 for c in text if c in KMR_CHARS)
    scores["KMR"] += min(kmr_chars_found * 0.03, 20)

    text_lower = text.lower()
    for kw in CKB_KEYWORDS:
        if kw in text: scores["CKB"] += 3
    for kw in KMR_KEYWORDS:
        if kw.lower() in text_lower: scores["KMR"] += 3
    for kw in ZZA_KEYWORDS:
        if kw.lower() in text_lower: scores["ZZA"] += 4
    for kw in HAW_KEYWORDS:
        if kw.lower() in text_lower or kw in text: scores["HAW"] += 4
    for kw in EN_KEYWORDS:
        cnt = text_lower.count(" " + kw + " ")
        scores["EN"] += min(cnt * 0.3, 10)
    for kw in AR_KEYWORDS:
        if kw in text: scores["AR"] += 2
    for kw in TR_KEYWORDS:
        if kw.lower() in text_lower: scores["TR"] += 2
    for kw in FA_KEYWORDS:
        if kw in text: scores["FA"] += 2

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
    except:
        pass
    return ""


def categorize_domain(domain):
    d = domain.lower()
    if any(x in d for x in [".edu.", "university", "zanko", "college"]): return "education"
    if any(x in d for x in [".gov.", "government", "parliament", "presidency"]): return "government"
    if any(x in d for x in ["news", "press", "media", "rudaw", "nrt", "hawar", "nuce"]): return "news_media"
    if any(x in d for x in ["tv", "radio", "fm", "sat"]): return "tv_radio"
    if any(x in d for x in ["hospital", "health", "medical", "clinic"]): return "health"
    if any(x in d for x in ["sport", "football", "fc", "varz"]): return "sports"
    if any(x in d for x in ["shop", "store", "mall", "bazar", "market"]): return "ecommerce"
    if any(x in d for x in ["ngo", "rights", "humanitarian"]): return "ngo"
    if any(x in d for x in ["tech", "dev", "code", "software"]): return "technology"
    if any(x in d for x in ["travel", "tour", "visit", "hotel"]): return "tourism"
    if any(x in d for x in ["law", "legal", "lawyer", "attorney"]): return "legal"
    if any(x in d for x in ["food", "recipe", "restaurant", "dolma"]): return "food"
    return "general"


def verify_url(url, source="discovered"):
    domain = urlparse(url).hostname or ""
    domain = domain.lower()
    if domain.startswith("www."): domain = domain[4:]
    category = categorize_domain(domain)

    result = {
        "url": url, "domain": domain, "title": "", "language": "UNK",
        "status_code": "ERR", "source": source, "category": category,
        "notes": "", "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
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
                result["notes"] = "SSL warning"
        except: result["status_code"] = "SSL_ERR"
    except requests.exceptions.ConnectionError: result["status_code"] = "CONN_ERR"
    except requests.exceptions.Timeout: result["status_code"] = "TIMEOUT"
    except Exception as e: result["status_code"] = f"ERR:{type(e).__name__}"

    return result


def normalize_language(lang):
    if not lang or lang == "UNK": return "Unknown"
    mapping = {
        "CKB": "CKB (Central Kurdish / Sorani)",
        "KMR": "KMR (Northern Kurdish / Kurmanji)",
        "ZZA": "ZZA (Zazaki)",
        "HAW": "HAW (Hawrami / Gorani)",
        "EN": "English", "AR": "Arabic", "TR": "Turkish", "FA": "Persian",
    }
    parts = []
    for sep in ["/", ","]:
        if sep in lang:
            parts = [p.strip() for p in lang.split(sep)]
            break
    if not parts: parts = [lang]
    return " / ".join(mapping.get(p, p) for p in parts)


def classify_dialect(lang):
    if not lang: return "Other"
    if "CKB" in lang: return "CKB (Sorani)"
    if "KMR" in lang: return "KMR (Kurmanji)"
    if "ZZA" in lang: return "ZZA (Zazaki)"
    if "HAW" in lang: return "HAW (Hawrami/Gorani)"
    return "Other"


def main():
    base_dir = os.path.dirname(__file__)
    complete_csv = os.path.join(base_dir, "kurdish_websites_complete.csv")
    round2_txt = os.path.join(base_dir, "round2_discovered.txt")
    output_csv = os.path.join(base_dir, "kurdish_websites_complete.csv")
    data_dir = os.path.join(os.path.dirname(base_dir), "data")
    data_all = os.path.join(data_dir, "kurdish_websites_all.csv")
    data_verified = os.path.join(data_dir, "kurdish_websites_verified.csv")

    # Load existing data
    existing = {}
    if os.path.exists(complete_csv):
        with open(complete_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row.get("domain", "").lower()] = row
    print(f"[INFO] {len(existing)} existing entries.")

    # Load round2 discovered
    new_urls = []
    if os.path.exists(round2_txt):
        with open(round2_txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    url, domain = parts[0], parts[1]
                    source = parts[3] if len(parts) > 3 else "discovered"
                    if domain not in existing:
                        new_urls.append((url, source))
    print(f"[INFO] {len(new_urls)} new URLs to verify.")

    # Verify new URLs
    new_results = []
    if new_urls:
        print(f"\n[VERIFYING] {len(new_urls)} new URLs...")
        with ThreadPoolExecutor(max_workers=50) as pool:
            futures = {pool.submit(verify_url, url, src): url for url, src in new_urls}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Verifying"):
                new_results.append(future.result())

    # Merge
    for result in new_results:
        d = result["domain"]
        if d not in existing:
            existing[d] = result

    rows = sorted(existing.values(), key=lambda x: x.get("domain", ""))
    print(f"\n[TOTAL] {len(rows)} unique domains.")

    # Write updated complete CSV
    fieldnames = ["url", "domain", "title", "language", "status_code",
                  "source", "category", "notes", "checked_at"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"[CSV] Updated {output_csv}")

    # Write deliverable CSVs
    os.makedirs(data_dir, exist_ok=True)
    out_fields = ["url", "domain", "title", "language_detected",
                  "kurdish_dialect_group", "status_code", "source",
                  "category", "notes", "checked_at"]

    all_out = []
    for row in rows:
        out = {
            "url": row.get("url", ""),
            "domain": row.get("domain", ""),
            "title": row.get("title", ""),
            "language_detected": normalize_language(row.get("language", "")),
            "kurdish_dialect_group": classify_dialect(row.get("language", "")),
            "status_code": row.get("status_code", ""),
            "source": row.get("source", ""),
            "category": row.get("category", "general"),
            "notes": row.get("notes", ""),
            "checked_at": row.get("checked_at", ""),
        }
        all_out.append(out)

    all_out.sort(key=lambda x: (x["kurdish_dialect_group"], x["domain"]))

    with open(data_all, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(all_out)

    verified = [r for r in all_out
                if r["status_code"].startswith("2") or r["status_code"].startswith("3")]

    with open(data_verified, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(verified)

    # Statistics
    ckb = sum(1 for r in rows if "CKB" in r.get("language", ""))
    kmr = sum(1 for r in rows if "KMR" in r.get("language", ""))
    zza = sum(1 for r in rows if "ZZA" in r.get("language", ""))
    haw = sum(1 for r in rows if "HAW" in r.get("language", ""))

    print(f"\n{'='*60}")
    print(f"  UPDATED KURDISH WEBSITES REPORT")
    print(f"{'='*60}")
    print(f"  Total unique domains:    {len(rows)}")
    print(f"  Verified (reachable):    {len(verified)}")
    print(f"  % Reachable:             {len(verified)/len(rows)*100:.1f}%")
    print(f"\n  Kurdish Language Sites:")
    print(f"    CKB (Sorani):   {ckb}")
    print(f"    KMR (Kurmanji): {kmr}")
    print(f"    ZZA (Zazaki):   {zza}")
    print(f"    HAW (Gorani):   {haw}")
    print(f"\n  Output files:")
    print(f"    {data_all}")
    print(f"    {data_verified}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
