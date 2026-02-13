#!/usr/bin/env python3
"""
Kurdish Websites Verification and CSV Export Script
=====================================================
Verifies Kurdish website URLs by making HTTP requests, detecting
languages via custom Kurdish script analysis, and exporting results
to a CSV file with a summary report.

Imports URLs from expanded_seeds.py which provides:
    get_all_expanded_urls() -> list[dict] with keys:
        url, source, category, notes

Usage:
    python verify_and_export.py
"""

import csv
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from expanded_seeds import get_all_expanded_urls

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_WORKERS = 50
REQUEST_TIMEOUT = 10  # seconds
OUTPUT_CSV = "/home/user/ai-notebooks/kurdish_websites/kurdish_websites.csv"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"
)

# ---------------------------------------------------------------------------
# URL Normalization
# ---------------------------------------------------------------------------

def normalize_url(raw_url: str) -> str:
    """Normalize a URL for consistent deduplication and fetching.

    Steps:
        1. Strip whitespace.
        2. Ensure scheme is https (upgrade http or add if missing).
        3. Lower-case the hostname.
        4. Remove fragments (#...).
        5. Remove trailing slashes from the path (unless path is just '/').
        6. Reassemble.
    """
    raw_url = raw_url.strip()

    # Add scheme if missing
    if not raw_url.startswith(("http://", "https://")):
        raw_url = "https://" + raw_url

    # Upgrade http -> https
    if raw_url.startswith("http://"):
        raw_url = "https://" + raw_url[7:]

    parsed = urlparse(raw_url)

    # Lower-case the netloc (host + optional port)
    netloc = parsed.netloc.lower()

    # Strip trailing slashes from path (but keep bare '/')
    path = parsed.path.rstrip("/") or "/"

    # Remove fragment, keep query
    normalized = urlunparse((
        parsed.scheme,
        netloc,
        path,
        parsed.params,
        parsed.query,
        "",  # drop fragment
    ))

    return normalized


def extract_domain(url: str) -> str:
    """Return the bare domain (host) from a URL string."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    # Remove 'www.' prefix for deduplication purposes
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


# ---------------------------------------------------------------------------
# Custom Kurdish Language Detection
# ---------------------------------------------------------------------------

# --- Character sets unique to Kurdish dialects in Arabic script -----------
# Central Kurdish (Sorani / CKB) uses modified Arabic script with these chars:
CKB_CHARS = set("ەێڕۆڵۊژچگپکی")
# Note: many of these are shared with Persian/Arabic but the combination and
# frequency is distinctive.  The truly *Kurdish-only* additions to Arabic are:
CKB_UNIQUE_CHARS = set("ەێڕۆڵ")  # not used in standard Arabic

# --- Keyword indicators per language / dialect ---

CKB_KEYWORDS = [
    # Common Sorani words / phrases
    "هەواڵ",        # news
    "کوردی",        # Kurdish
    "کوردستان",     # Kurdistan
    "هەرێم",        # region
    "پەرلەمان",     # parliament
    "حکومەت",       # government
    "سیاسی",        # political (also Arabic but very common in Kurdish press)
    "ئابووری",      # economic
    "کۆمەڵایەتی",   # social
    "وەزارەت",      # ministry
    "هەولێر",       # Erbil
    "سلێمانی",      # Sulaymaniyah
    "دهۆک",         # Duhok
    "کەرکوک",       # Kirkuk
    "بەغدا",        # Baghdad (Kurdish spelling)
    "پێشمەرگە",     # Peshmerga
    "خوێندکار",     # student
    "زانکۆ",        # university
    "مامۆستا",      # teacher
    "رۆژنامە",      # newspaper
    "چاودێری",      # monitoring
    "تایبەت",       # special/private
    "دەستەبەر",     # available
    "بڵاوکراوەتەوە", # published
    "لەگەڵ",        # with
    "بەرنامە",      # programme
    "ئەنجومەن",     # council
    "دەزگا",        # institution
    "وتار",         # article/speech
    "بابەت",        # topic
    "لاپەڕە",       # page
]

KMR_KEYWORDS = [
    # Northern Kurdish (Kurmanji) - Latin script
    "Kurdî",
    "Kurdistan",
    "Kurdistanê",
    "Rojava",
    "Rojavayê",
    "Bakurê",
    "Başûrê",
    "Rojhilatê",
    "nûçe",         # news
    "nûçeyên",
    "siyaset",      # politics
    "civak",        # society
    "çand",         # culture
    "wêje",         # literature
    "Tirkiye",      # Turkey
    "dewlet",       # state
    "maf",          # right(s)
    "azadî",        # freedom
    "gelê",         # people's
    "Kurd",
    "Pêşmerge",
    "hevpeyvîn",    # interview
    "gotûbêj",      # discussion
    "raporta",      # report
    "weşanxane",    # publishing
    "hunermend",    # artist
    "nivîskar",     # writer
    "rêxistîn",     # organization
    "daxuyanî",     # statement
    "bernameya",    # programme
    "xwendevan",    # reader
    "malpera",      # website
    "serpêhatî",    # adventure/story
    "jiyan",        # life
    "Diyarbakir",
    "Amed",
    "Kobanê",
    "Efrîn",
    "Qamişlo",
    "Duhok",
    "Hewlêr",       # Erbil in Kurmanji
]

# Characters specific to Kurmanji Latin orthography (with diacritics)
KMR_SPECIAL_CHARS = set("êîûşçÊÎÛŞÇ")

ZAZAKI_KEYWORDS = [
    "Zazaki",
    "Zazakî",
    "Dimilî",
    "Dimilki",
    "Kirmancki",
    "Kirmanckî",
    "Zonê ma",
    "Zazaca",
    "Dersim",
    "Tunceli",
    "Bingöl",
    "Pülümür",
    "Çewlig",
]

HAWRAMI_KEYWORDS = [
    "هەورامی",      # Hawrami
    "گۆرانی",       # Gorani
    "پاوە",         # Paveh
    "هەورامان",     # Hawraman
    "نەوسوود",
    "ماچو",
    "Hawrami",
    "Gorani",
    "Hewramî",
    "Goranî",
    "Hawraman",
    "Paveh",
]

ARABIC_KEYWORDS = [
    "العربية",
    "الإخبارية",
    "الأخبار",
    "الرئيسية",
    "المزيد",
    "العراق",
    "مصر",
    "السعودية",
    "عربي",
    "موقع",
]

TURKISH_KEYWORDS = [
    "Türkçe",
    "Türkiye",
    "haberleri",
    "haberler",
    "Anasayfa",
    "İletişim",
    "Gündem",
    "Siyaset",
    "Ekonomi",
    "Dünya",
    "hakkında",
    "devamını",
    "gazetesi",
]

PERSIAN_KEYWORDS = [
    "فارسی",
    "ایران",
    "تهران",
    "خبرگزاری",
    "سایت",
    "صفحه",
    "مطالب",
    "درباره",
    "خبرنگار",
    "اجتماعی",
    "فرهنگی",
]

ENGLISH_KEYWORDS = [
    "English",
    "Home",
    "About",
    "Contact",
    "News",
    "Politics",
    "World",
    "Opinion",
    "Subscribe",
    "Copyright",
    "Privacy Policy",
    "Terms of Service",
    "All rights reserved",
]


def _count_char_matches(text: str, charset: set) -> int:
    """Count how many characters in *text* belong to *charset*."""
    return sum(1 for ch in text if ch in charset)


def _count_keyword_matches(text: str, keywords: list, case_sensitive: bool = True) -> int:
    """Count how many keywords appear in *text*.

    For case-insensitive matching (Latin-script languages), we lower-case
    both the text and the keywords.
    """
    if not case_sensitive:
        text_search = text.lower()
        return sum(1 for kw in keywords if kw.lower() in text_search)
    return sum(1 for kw in keywords if kw in text)


def detect_language(html_content: str) -> str:
    """Detect the primary language(s) of an HTML page.

    Returns a comma-separated string of detected language tags, ordered
    by confidence (most likely first).  Possible tags:

        CKB (Central Kurdish / Sorani)
        KMR (Northern Kurdish / Kurmanji)
        ZZA (Zazaki)
        HAW (Hawrami / Gorani)
        AR  (Arabic)
        TR  (Turkish)
        FA  (Persian / Farsi)
        EN  (English)
        UNK (Unknown)

    The detection is heuristic: it counts Kurdish-specific Unicode
    characters and keyword hits, then ranks each language by a
    weighted score.
    """
    if not html_content:
        return "UNK"

    # ----- Extract visible text from HTML ---------------------------------
    try:
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "noscript", "meta", "link"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
    except Exception:
        text = html_content

    # Also check the raw HTML (for lang= attributes, meta tags, etc.)
    raw_lower = html_content.lower()

    # ----- HTML-level hints -----------------------------------------------
    # Check <html lang="..."> attribute
    html_lang_hint = ""
    lang_match = re.search(r'<html[^>]*\slang=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
    if lang_match:
        html_lang_hint = lang_match.group(1).lower().strip()

    # Check <meta> content-language
    meta_lang_match = re.search(
        r'<meta[^>]*http-equiv=["\']content-language["\'][^>]*content=["\']([^"\']+)["\']',
        html_content, re.IGNORECASE,
    )
    if not meta_lang_match:
        meta_lang_match = re.search(
            r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*http-equiv=["\']content-language["\']',
            html_content, re.IGNORECASE,
        )
    if meta_lang_match:
        html_lang_hint = meta_lang_match.group(1).lower().strip()

    # ----- Score each language --------------------------------------------
    scores: dict[str, float] = {
        "CKB": 0.0,
        "KMR": 0.0,
        "ZZA": 0.0,
        "HAW": 0.0,
        "AR": 0.0,
        "TR": 0.0,
        "FA": 0.0,
        "EN": 0.0,
    }

    # --- Character-based signals ------------------------------------------
    # CKB unique characters (strong signal)
    ckb_unique_count = _count_char_matches(text, CKB_UNIQUE_CHARS)
    ckb_char_count = _count_char_matches(text, CKB_CHARS)
    scores["CKB"] += ckb_unique_count * 3.0  # very strong per-char signal
    scores["CKB"] += ckb_char_count * 0.5

    # KMR special Latin chars (ê, î, û, ş, ç)
    kmr_char_count = _count_char_matches(text, KMR_SPECIAL_CHARS)
    scores["KMR"] += kmr_char_count * 2.0

    # Turkish also uses ş and ç, but not ê, î, û with same frequency
    # Give Turkish a smaller boost for ş/ç
    turkish_shared = _count_char_matches(text, set("şçŞÇ"))
    scores["TR"] += turkish_shared * 0.3

    # General Arabic-script presence (could be Arabic, Persian, or CKB)
    arabic_script_count = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]', text))
    # Only boost AR/FA if we do NOT see strong CKB signals
    if ckb_unique_count < 10:
        scores["AR"] += arabic_script_count * 0.05
        scores["FA"] += arabic_script_count * 0.03

    # Latin-script presence (helps EN, KMR, TR, ZZA)
    latin_count = len(re.findall(r'[a-zA-Z]', text))

    # --- Keyword-based signals --------------------------------------------
    ckb_kw = _count_keyword_matches(text, CKB_KEYWORDS)
    scores["CKB"] += ckb_kw * 8.0

    kmr_kw = _count_keyword_matches(text, KMR_KEYWORDS, case_sensitive=False)
    scores["KMR"] += kmr_kw * 8.0

    zza_kw = _count_keyword_matches(text, ZAZAKI_KEYWORDS, case_sensitive=False)
    scores["ZZA"] += zza_kw * 10.0

    haw_kw = _count_keyword_matches(text, HAWRAMI_KEYWORDS)
    scores["HAW"] += haw_kw * 10.0

    ar_kw = _count_keyword_matches(text, ARABIC_KEYWORDS)
    scores["AR"] += ar_kw * 6.0

    tr_kw = _count_keyword_matches(text, TURKISH_KEYWORDS, case_sensitive=False)
    scores["TR"] += tr_kw * 6.0

    fa_kw = _count_keyword_matches(text, PERSIAN_KEYWORDS)
    scores["FA"] += fa_kw * 6.0

    en_kw = _count_keyword_matches(text, ENGLISH_KEYWORDS, case_sensitive=False)
    scores["EN"] += en_kw * 4.0  # lower weight — English words appear everywhere

    # --- HTML lang hint bonus ---------------------------------------------
    LANG_HINT_MAP = {
        "ckb": "CKB", "ku": "KMR", "kmr": "KMR",
        "ar": "AR", "tr": "TR", "fa": "FA", "en": "EN",
        "zza": "ZZA",
    }
    for prefix, lang_tag in LANG_HINT_MAP.items():
        if html_lang_hint.startswith(prefix):
            scores[lang_tag] += 30.0
            break

    # Also check for Kurdish sub-tags like ku-Arab (Sorani) vs ku-Latn (Kurmanji)
    if html_lang_hint.startswith("ku"):
        if "arab" in html_lang_hint:
            scores["CKB"] += 20.0
        elif "latn" in html_lang_hint:
            scores["KMR"] += 20.0

    # --- Build result -----------------------------------------------------
    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    # Only include languages that scored above a minimum threshold
    MIN_THRESHOLD = 5.0
    detected = [lang for lang, score in ranked if score >= MIN_THRESHOLD]

    if not detected:
        return "UNK"

    # Return top language(s).  Include a second language only if its score
    # is at least 40% of the top score (i.e. the page is multilingual).
    top_score = ranked[0][1]
    results = []
    for lang, score in ranked:
        if score < MIN_THRESHOLD:
            break
        if score >= top_score * 0.4:
            results.append(lang)
        else:
            break
    return ", ".join(results) if results else "UNK"


# ---------------------------------------------------------------------------
# Page title extraction
# ---------------------------------------------------------------------------

def extract_title(html_content: str) -> str:
    """Extract the <title> text from HTML, falling back gracefully."""
    if not html_content:
        return ""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        if soup.title and soup.title.string:
            return soup.title.string.strip()
    except Exception:
        pass
    # Regex fallback
    m = re.search(r"<title[^>]*>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Single-URL verification worker
# ---------------------------------------------------------------------------

def verify_url(entry: dict) -> dict:
    """Verify a single URL entry and return an enriched result dict.

    Parameters
    ----------
    entry : dict
        Must contain at least 'url'.  May contain 'source', 'category',
        'notes'.

    Returns
    -------
    dict with keys: url, domain, title, language, status_code, source,
                    category, notes, checked_at, error
    """
    raw_url = entry.get("url", "")
    source = entry.get("source", "")
    category = entry.get("category", "")
    notes = entry.get("notes", "")

    url = normalize_url(raw_url)
    domain = extract_domain(url)
    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    result = {
        "url": url,
        "domain": domain,
        "title": "",
        "language": "UNK",
        "status_code": None,
        "source": source,
        "category": category,
        "notes": notes,
        "checked_at": checked_at,
        "error": "",
    }

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5,ku;q=0.3,ar;q=0.2",
    }

    try:
        resp = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
            verify=True,
        )
        result["status_code"] = resp.status_code

        # Try to decode properly
        # requests usually does a good job, but force utf-8 for Kurdish pages
        if resp.encoding and resp.encoding.lower() == "iso-8859-1":
            resp.encoding = resp.apparent_encoding or "utf-8"

        html = resp.text

        result["title"] = extract_title(html)
        result["language"] = detect_language(html)

    except requests.exceptions.SSLError as exc:
        result["error"] = f"SSL Error: {exc}"
        result["status_code"] = None
        # Retry without SSL verification
        try:
            resp = requests.get(
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
                verify=False,
            )
            result["status_code"] = resp.status_code
            if resp.encoding and resp.encoding.lower() == "iso-8859-1":
                resp.encoding = resp.apparent_encoding or "utf-8"
            html = resp.text
            result["title"] = extract_title(html)
            result["language"] = detect_language(html)
            result["error"] = "SSL Error (fetched with verify=False)"
        except Exception as retry_exc:
            result["error"] = f"SSL Error + retry failed: {retry_exc}"

    except requests.exceptions.ConnectionError as exc:
        result["error"] = f"Connection Error: {exc}"

    except requests.exceptions.Timeout:
        result["error"] = "Timeout (>{} seconds)".format(REQUEST_TIMEOUT)

    except requests.exceptions.TooManyRedirects:
        result["error"] = "Too many redirects"

    except requests.exceptions.RequestException as exc:
        result["error"] = f"Request Error: {exc}"

    except Exception as exc:
        result["error"] = f"Unexpected Error: {exc}"

    return result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_by_domain(entries: list[dict]) -> list[dict]:
    """Remove duplicate entries that share the same domain.

    The first occurrence (in list order) is kept.  The domain is
    extracted after URL normalization and ignores 'www.' prefix.
    """
    seen_domains: set[str] = set()
    unique: list[dict] = []
    duplicates_removed = 0

    for entry in entries:
        url = normalize_url(entry.get("url", ""))
        domain = extract_domain(url)
        if domain and domain not in seen_domains:
            seen_domains.add(domain)
            unique.append(entry)
        else:
            duplicates_removed += 1

    if duplicates_removed:
        print(f"[INFO] Removed {duplicates_removed} duplicate domain(s).")
    return unique


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "url",
    "domain",
    "title",
    "language",
    "status_code",
    "source",
    "category",
    "notes",
    "checked_at",
]


def write_csv(results: list[dict], filepath: str) -> None:
    """Write verification results to a CSV file."""
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\n[CSV] Wrote {len(results)} rows to {filepath}")


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], elapsed: float) -> None:
    """Print a human-readable summary of the verification run."""
    total = len(results)
    status_counts: Counter = Counter()
    language_counts: Counter = Counter()
    category_counts: Counter = Counter()
    error_count = 0
    reachable = 0

    for r in results:
        sc = r.get("status_code")
        if sc is not None:
            status_counts[sc] += 1
            if 200 <= sc < 400:
                reachable += 1
        else:
            status_counts["No Response"] += 1

        if r.get("error"):
            error_count += 1

        # Count primary language only (first in comma-sep list)
        lang = r.get("language", "UNK")
        primary_lang = lang.split(",")[0].strip()
        language_counts[primary_lang] += 1

        cat = r.get("category", "uncategorized") or "uncategorized"
        category_counts[cat] += 1

    separator = "=" * 65

    print(f"\n{separator}")
    print("  KURDISH WEBSITES VERIFICATION REPORT")
    print(f"{separator}")
    print(f"  Date            : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Total URLs      : {total}")
    print(f"  Reachable (2xx) : {reachable}  ({reachable/total*100:.1f}%)" if total else "")
    print(f"  Errors          : {error_count}")
    print(f"  Elapsed time    : {elapsed:.1f} seconds")
    print(f"  Workers         : {MAX_WORKERS}")

    print(f"\n  --- HTTP Status Code Distribution ---")
    for code, count in sorted(status_counts.items(), key=lambda x: (-x[1], str(x[0]))):
        bar = "#" * min(count, 50)
        print(f"    {str(code):>15s} : {count:>4d}  {bar}")

    print(f"\n  --- Detected Languages ---")
    for lang, count in language_counts.most_common():
        bar = "#" * min(count, 50)
        print(f"    {lang:>15s} : {count:>4d}  {bar}")

    print(f"\n  --- Categories ---")
    for cat, count in category_counts.most_common():
        bar = "#" * min(count, 50)
        print(f"    {cat:>15s} : {count:>4d}  {bar}")

    # Show top errors
    error_types: Counter = Counter()
    for r in results:
        err = r.get("error", "")
        if err:
            # Simplify error for grouping
            if "SSL" in err:
                error_types["SSL Errors"] += 1
            elif "Timeout" in err:
                error_types["Timeouts"] += 1
            elif "Connection" in err:
                error_types["Connection Errors"] += 1
            elif "DNS" in err or "NameResolution" in err or "Name or service" in err:
                error_types["DNS Failures"] += 1
            elif "Too many redirects" in err:
                error_types["Redirect Loops"] += 1
            else:
                error_types["Other Errors"] += 1

    if error_types:
        print(f"\n  --- Error Breakdown ---")
        for etype, count in error_types.most_common():
            bar = "#" * min(count, 50)
            print(f"    {etype:>20s} : {count:>4d}  {bar}")

    print(f"\n{separator}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[INFO] Loading URLs from expanded_seeds.get_all_expanded_urls() ...")
    entries = get_all_expanded_urls()
    print(f"[INFO] Loaded {len(entries)} URL entries.")

    # Deduplicate by domain
    entries = deduplicate_by_domain(entries)
    print(f"[INFO] {len(entries)} unique domains to verify.")

    print(f"[INFO] Starting verification with {MAX_WORKERS} workers ...\n")

    results: list[dict] = []
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_entry = {
            executor.submit(verify_url, entry): entry
            for entry in entries
        }

        with tqdm(total=len(future_to_entry), desc="Verifying URLs", unit="url") as pbar:
            for future in as_completed(future_to_entry):
                try:
                    result = future.result()
                except Exception as exc:
                    # Should not happen since verify_url catches all errors,
                    # but handle just in case.
                    entry = future_to_entry[future]
                    result = {
                        "url": entry.get("url", ""),
                        "domain": extract_domain(normalize_url(entry.get("url", ""))),
                        "title": "",
                        "language": "UNK",
                        "status_code": None,
                        "source": entry.get("source", ""),
                        "category": entry.get("category", ""),
                        "notes": entry.get("notes", ""),
                        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "error": f"Future exception: {exc}",
                    }
                results.append(result)
                pbar.update(1)

    elapsed = time.monotonic() - start_time

    # Sort results by domain for readability
    results.sort(key=lambda r: r.get("domain", ""))

    # Write CSV
    write_csv(results, OUTPUT_CSV)

    # Print summary
    print_summary(results, elapsed)

    # Final counts for quick reference
    ok_count = sum(1 for r in results if r.get("status_code") and 200 <= r["status_code"] < 400)
    print(f"[DONE] {ok_count}/{len(results)} sites are reachable.  CSV saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
