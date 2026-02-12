#!/usr/bin/env python3
"""
Create the final deliverable CSV:
- Clean up data
- Group by Kurdish dialect
- Filter for reachable sites
- Produce both full and filtered CSVs
"""

import csv
import os
from collections import defaultdict
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
INPUT_CSV = os.path.join(BASE_DIR, "kurdish_websites_complete.csv")
OUTPUT_ALL = os.path.join(os.path.dirname(BASE_DIR), "data", "kurdish_websites_all.csv")
OUTPUT_VERIFIED = os.path.join(os.path.dirname(BASE_DIR), "data", "kurdish_websites_verified.csv")


def normalize_language(lang):
    """Normalize language detection labels."""
    if not lang or lang == "UNK":
        return "Unknown"
    # Handle multi-language
    parts = []
    for sep in ["/", ","]:
        if sep in lang:
            parts = [p.strip() for p in lang.split(sep)]
            break
    if not parts:
        parts = [lang]

    mapping = {
        "CKB": "CKB (Central Kurdish / Sorani)",
        "KMR": "KMR (Northern Kurdish / Kurmanji)",
        "ZZA": "ZZA (Zazaki)",
        "HAW": "HAW (Hawrami / Gorani)",
        "EN": "English",
        "AR": "Arabic",
        "TR": "Turkish",
        "FA": "Persian",
    }

    normalized = []
    for p in parts:
        p = p.strip()
        normalized.append(mapping.get(p, p))

    return " / ".join(normalized)


def classify_kurdish_dialect(lang):
    """Classify into Kurdish dialect groups."""
    if not lang:
        return "Other"
    if "CKB" in lang:
        return "CKB (Sorani)"
    if "KMR" in lang:
        return "KMR (Kurmanji)"
    if "ZZA" in lang:
        return "ZZA (Zazaki)"
    if "HAW" in lang:
        return "HAW (Hawrami/Gorani)"
    return "Other"


def normalize_category(cat):
    """Normalize category names."""
    mapping = {
        "news_media": "News & Media",
        "tv_radio": "TV & Radio",
        "tv_channels": "TV & Radio",
        "radio": "TV & Radio",
        "education": "Education",
        "government": "Government",
        "cultural": "Culture & Reference",
        "culture_arts": "Culture & Arts",
        "religious": "Religious",
        "ecommerce": "E-Commerce",
        "technology": "Technology",
        "health": "Health",
        "sports": "Sports",
        "tourism": "Tourism",
        "ngo": "NGO & Human Rights",
        "political_parties": "Political",
        "political": "Political",
        "diaspora": "Diaspora",
        "blogs": "Blogs & Forums",
        "food": "Food & Lifestyle",
        "general": "General",
        "discovered": "General",
    }
    return mapping.get(cat, cat.replace("_", " ").title() if cat else "General")


def main():
    # Read input
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"[INFO] Read {len(rows)} entries from complete CSV.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_ALL), exist_ok=True)

    # Process and write ALL entries
    fieldnames_out = [
        "url", "domain", "title", "language_detected",
        "kurdish_dialect_group", "status_code", "source",
        "category", "notes", "checked_at",
    ]

    all_out = []
    for row in rows:
        out = {
            "url": row.get("url", ""),
            "domain": row.get("domain", ""),
            "title": row.get("title", ""),
            "language_detected": normalize_language(row.get("language", "")),
            "kurdish_dialect_group": classify_kurdish_dialect(row.get("language", "")),
            "status_code": row.get("status_code", ""),
            "source": row.get("source", ""),
            "category": normalize_category(row.get("category", "")),
            "notes": row.get("notes", ""),
            "checked_at": row.get("checked_at", ""),
        }
        all_out.append(out)

    # Sort by dialect group then domain
    all_out.sort(key=lambda x: (x["kurdish_dialect_group"], x["domain"]))

    with open(OUTPUT_ALL, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_out)
        writer.writeheader()
        writer.writerows(all_out)

    print(f"[CSV] Wrote {len(all_out)} entries to {OUTPUT_ALL}")

    # Filter for verified (reachable) entries
    verified = [r for r in all_out
                if r["status_code"].startswith("2") or r["status_code"].startswith("3")]

    with open(OUTPUT_VERIFIED, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_out)
        writer.writeheader()
        writer.writerows(verified)

    print(f"[CSV] Wrote {len(verified)} verified entries to {OUTPUT_VERIFIED}")

    # Statistics
    dialect_counts = defaultdict(int)
    cat_counts = defaultdict(int)
    for r in verified:
        dialect_counts[r["kurdish_dialect_group"]] += 1
        cat_counts[r["category"]] += 1

    print(f"\n{'='*60}")
    print(f"  FINAL DELIVERABLE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total domains (all):       {len(all_out)}")
    print(f"  Verified (reachable):      {len(verified)}")
    print(f"  Date:                      {datetime.utcnow().strftime('%Y-%m-%d')}")
    print(f"\n  --- Kurdish Dialect Groups (verified) ---")
    for group, cnt in sorted(dialect_counts.items(), key=lambda x: -x[1]):
        pct = cnt/len(verified)*100
        bar = "#" * min(int(pct), 50)
        print(f"    {group:>25}: {cnt:>5} ({pct:5.1f}%)  {bar}")
    print(f"\n  --- Categories (verified) ---")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct = cnt/len(verified)*100
        print(f"    {cat:>25}: {cnt:>5} ({pct:5.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
