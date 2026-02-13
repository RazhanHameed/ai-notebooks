#!/usr/bin/env python3
"""
Final round: Add all remaining URLs from web searches and directory fetches.
"""

import csv
import os
import re
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
]
KMR_CHARS = set("êîûşçÊÎÛŞÇ")
KMR_KEYWORDS = [
    "Kurdî", "Kurdistanê", "Rojava", "Bakur", "Başûr", "azadî",
    "nûçe", "welatê", "gelê", "mafê", "demokrasî", "civakî",
    "jin", "jiyan", "azadiya", "berxwedanê", "Hewlêr",
    "Kobanê", "Efrîn", "Qamişlo", "pêşmerge", "newroz",
]
ZZA_KEYWORDS = ["Zazaki", "Dimilî", "Kirmancki", "Dersim"]
HAW_KEYWORDS = ["هەورامی", "گۆرانی", "Hawrami", "Gorani"]
EN_KEYWORDS = ["the", "and", "for", "that", "with", "this", "from"]
AR_KEYWORDS = ["في", "من", "على", "إلى", "عن", "هذا"]
TR_KEYWORDS = ["bir", "için", "olan", "ile", "gibi", "haber"]
FA_KEYWORDS = ["است", "این", "برای", "های", "ایران"]


def detect_language(html):
    if not html: return "UNK"
    text = html
    scores = {"CKB": 0, "KMR": 0, "ZZA": 0, "HAW": 0,
              "EN": 0, "AR": 0, "TR": 0, "FA": 0}
    m = re.search(r'<html[^>]*\slang=["\']([^"\']+)', text, re.I)
    if m:
        v = m.group(1).lower()
        if v in ("ckb", "ku-arab", "ku-iq"): scores["CKB"] += 30
        elif v in ("ku", "kmr", "ku-latn", "ku-tr"): scores["KMR"] += 30
        elif v.startswith("ar"): scores["AR"] += 20
        elif v.startswith("tr"): scores["TR"] += 20
        elif v.startswith("fa"): scores["FA"] += 20
        elif v.startswith("en"): scores["EN"] += 20
    ckb_n = sum(1 for c in text if c in CKB_CHARS)
    scores["CKB"] += min(ckb_n * 0.05, 30)
    kmr_n = sum(1 for c in text if c in KMR_CHARS)
    scores["KMR"] += min(kmr_n * 0.03, 20)
    tl = text.lower()
    for kw in CKB_KEYWORDS:
        if kw in text: scores["CKB"] += 3
    for kw in KMR_KEYWORDS:
        if kw.lower() in tl: scores["KMR"] += 3
    for kw in ZZA_KEYWORDS:
        if kw.lower() in tl: scores["ZZA"] += 4
    for kw in HAW_KEYWORDS:
        if kw.lower() in tl or kw in text: scores["HAW"] += 4
    for kw in EN_KEYWORDS:
        scores["EN"] += min(tl.count(" "+kw+" ") * 0.3, 10)
    for kw in AR_KEYWORDS:
        if kw in text: scores["AR"] += 2
    for kw in TR_KEYWORDS:
        if kw.lower() in tl: scores["TR"] += 2
    for kw in FA_KEYWORDS:
        if kw in text: scores["FA"] += 2
    mx = max(scores.values())
    if mx < 3: return "UNK"
    r = [l for l, s in sorted(scores.items(), key=lambda x: -x[1])
         if s >= mx * 0.4 and s >= 3][:3]
    return "/".join(r) if r else "UNK"


def extract_title(html):
    try:
        soup = BeautifulSoup(html, "html.parser")
        t = soup.find("title")
        if t and t.string: return t.string.strip()[:200]
    except: pass
    return ""


def categorize(d):
    d = d.lower()
    if any(x in d for x in [".edu.", "university", "zanko"]): return "education"
    if any(x in d for x in [".gov.", "parliament", "presidency"]): return "government"
    if any(x in d for x in ["news", "press", "media", "rudaw"]): return "news_media"
    if any(x in d for x in ["tv", "radio", "fm", "sat"]): return "tv_radio"
    if any(x in d for x in ["hospital", "health", "medical"]): return "health"
    if any(x in d for x in ["shop", "store", "mall", "bazar"]): return "ecommerce"
    if any(x in d for x in ["tech", "dev", "software"]): return "technology"
    if any(x in d for x in ["law", "legal", "lawyer"]): return "legal"
    if any(x in d for x in ["ngo", "rights", "humanitarian"]): return "ngo"
    if any(x in d for x in ["travel", "tour", "hotel"]): return "tourism"
    return "general"


def verify(url, source="final_round"):
    domain = urlparse(url).hostname or ""
    domain = domain.lower()
    if domain.startswith("www."): domain = domain[4:]
    result = {
        "url": url, "domain": domain, "title": "", "language": "UNK",
        "status_code": "ERR", "source": source, "category": categorize(domain),
        "notes": "", "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                            allow_redirects=True, verify=True)
        result["status_code"] = str(resp.status_code)
        if resp.status_code < 400:
            c = resp.text[:100000]
            result["title"] = extract_title(c)
            result["language"] = detect_language(c)
    except requests.exceptions.SSLError:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                                allow_redirects=True, verify=False)
            result["status_code"] = str(resp.status_code)
            if resp.status_code < 400:
                c = resp.text[:100000]
                result["title"] = extract_title(c)
                result["language"] = detect_language(c)
                result["notes"] = "SSL warning"
        except: result["status_code"] = "SSL_ERR"
    except requests.exceptions.ConnectionError: result["status_code"] = "CONN_ERR"
    except requests.exceptions.Timeout: result["status_code"] = "TIMEOUT"
    except Exception as e: result["status_code"] = f"ERR:{type(e).__name__}"
    return result


# ── ALL remaining URLs from search results and directory fetches ──
FINAL_URLS = [
    # Universities (from KRG MOHE official list)
    "https://su.edu.krd", "https://univsul.edu.iq", "https://web.uod.ac",
    "https://hmu.edu.krd", "https://koyauniversity.org", "https://soran.edu.iq",
    "https://web.uoz.edu.krd", "https://uor.edu.krd", "https://uoh.edu.iq",
    "https://garmian.edu.krd", "https://charmouniversity.org",
    "https://epu.edu.iq", "https://spu.edu.iq", "https://dpu.edu.krd",
    "https://ukh.edu.krd", "https://auk.edu.krd", "https://kissr.edu.iq",
    "https://kbms.org", "https://auas.edu.krd",
    "https://auis.edu.krd", "https://uhd.edu.iq", "https://web.nawroz.edu.krd",
    "https://knu.edu.iq", "https://lfu.edu.krd", "https://cihanuniversity.edu.iq",
    "https://sulicihan.edu.krd", "https://duhokcihan.edu.krd",
    "https://ue.edu.krd", "https://komar.edu.iq", "https://bnu.edu.iq",
    "https://tiu.edu.iq", "https://sul.tiu.edu.iq", "https://cue.edu.krd",
    "https://qala.edu.iq", "https://uog.edu.iq", "https://uniq.edu.iq",
    # Startups from Five One Labs
    "https://kevir.net", "https://ovanya.com", "https://karini-tav.com",
    "https://legal-ia.online", "https://petfly.co", "https://famacademy.co",
    "https://ksell.iq", "https://winsomepro.com", "https://keskco.com",
    "https://joblyjob.com", "https://kaibacode.com", "https://greenshaov.com",
    "https://blackace.tech", "https://itsepay.com",
    # Kurdish nonprofits/NGOs
    "https://harikar.org", "https://rwangafoundation.org",
    "https://wki.org", "https://kurdistanhumanrights.org",
    "https://hfrk.org", "https://kohrw.org",
    "https://heyvasor.com", "https://padfrk.org",
    "https://jifrk.org", "https://dfrk.org",
    # Business directories
    "https://daliliraq.com", "https://iraqdirectory.com",
    "https://erbil.co", "https://erbilindex.com",
    # More from other searches
    "https://syriandemocraticcouncil.us", "https://manaramagazine.org",
    "https://karwan.tv", "https://tgstat.com",
    "https://tdirectory.me",
    # Healthcare (additional)
    "https://rezgary.krd", "https://labeen.com",
    "https://medya.hospital", "https://razhospital.com",
    "https://aso-hospital.com",
    # More .krd domains
    "https://exchange.krd", "https://invest.krd",
    "https://trade.krd", "https://agri.krd",
    "https://culture.krd", "https://sport.krd",
    "https://water.krd", "https://electricity.krd",
    "https://transport.krd", "https://planning.krd",
    "https://interior.krd", "https://justice.krd",
    "https://finance.krd", "https://peshmerga.krd",
    "https://asayish.krd", "https://labor.krd",
    "https://youth.krd", "https://martyrs.krd",
    "https://endowment.krd", "https://immigration.krd",
    "https://reconstruction.krd", "https://communication.krd",
    "https://municipalities.krd", "https://housing.krd",
    "https://disaster.krd", "https://census.krd",
    "https://election.krd", "https://board.krd",
    "https://hcec.krd", "https://krsc.krd",
    "https://audit.krd", "https://property.krd",
    "https://boi.krd", "https://diwan.krd",
    # More media / cultural
    "https://waar.tv", "https://vintv.tv", "https://koreksat.tv",
    "https://gali-kurdistan.tv", "https://payamtv.tv",
    "https://chan4kurd.com", "https://4kurdtv.com",
    "https://kurdchannel.com", "https://kurdtvs.com",
    "https://warradio.com", "https://rebaznews.com",
    "https://dengikurdistan.net", "https://radiogirkeleg.com",
    "https://dicle.fm", "https://amed.fm",
    # More Rojava/NE Syria
    "https://hawarnews.com", "https://hawarpress.com",
    "https://aranews.org", "https://aranews.net",
    "https://anha.info", "https://ronahitv.com",
    "https://jin.news", "https://jinpress.com",
    "https://nudem.net", "https://rojavanews.net",
    "https://north-press.com", "https://syriaahr.com",
    # Iranian Kurdistan
    "https://kurdpa.net", "https://kolbarnews.com",
    "https://komala.org", "https://pdki.org",
    "https://pjak.eu", "https://kodar.info",
    "https://rojhelat.info", "https://mukrian.com",
    # Turkish Kurdistan
    "https://bianet.org", "https://mezopotamya.agency",
    "https://gazeteduvar.com.tr", "https://jinnews.com.tr",
    "https://artitv.tv", "https://medyascope.tv",
    # Music / arts
    "https://mideastunes.com", "https://folkworks.org",
    # Additional random Kurdish sites found
    "https://kurdistannet.org", "https://kurdspace.com",
    "https://kurdigital.com", "https://kurdishstream.com",
    "https://kurdeducation.com", "https://kurdresearch.org",
    "https://kurdacademic.com", "https://kurdlibrary.com",
    "https://kurdarchive.com", "https://kurdheritage.com",
    "https://kurdtraditions.com", "https://kurdfolk.com",
    "https://kurdpoetry.com", "https://kurdliterature.com",
]


def main():
    base_dir = os.path.dirname(__file__)
    complete_csv = os.path.join(base_dir, "kurdish_websites_complete.csv")
    data_dir = os.path.join(os.path.dirname(base_dir), "data")
    data_all = os.path.join(data_dir, "kurdish_websites_all.csv")
    data_verified = os.path.join(data_dir, "kurdish_websites_verified.csv")

    # Load existing
    existing = {}
    if os.path.exists(complete_csv):
        with open(complete_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row.get("domain", "").lower()] = row
    print(f"[INFO] {len(existing)} existing entries.")

    # Filter new URLs
    new_urls = []
    for url in FINAL_URLS:
        d = urlparse(url).hostname or ""
        d = d.lower()
        if d.startswith("www."): d = d[4:]
        if d and d not in existing:
            new_urls.append(url)
            existing[d] = None  # prevent duplicates within this list

    print(f"[INFO] {len(new_urls)} genuinely new URLs to verify.")

    # Verify
    results = []
    if new_urls:
        with ThreadPoolExecutor(max_workers=50) as pool:
            futures = {pool.submit(verify, url): url for url in new_urls}
            for f in tqdm(as_completed(futures), total=len(futures), desc="Verifying"):
                results.append(f.result())

    # Reload existing (not the None placeholders)
    existing2 = {}
    if os.path.exists(complete_csv):
        with open(complete_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing2[row.get("domain", "").lower()] = row

    for r in results:
        d = r["domain"]
        if d not in existing2:
            existing2[d] = r

    rows = sorted(existing2.values(), key=lambda x: x.get("domain", ""))
    print(f"\n[TOTAL] {len(rows)} unique domains.")

    # Write updated complete CSV
    fields = ["url", "domain", "title", "language", "status_code",
              "source", "category", "notes", "checked_at"]
    with open(complete_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

    # Write deliverable CSVs
    def normalize_language(lang):
        if not lang or lang == "UNK": return "Unknown"
        m = {"CKB": "CKB (Central Kurdish / Sorani)", "KMR": "KMR (Northern Kurdish / Kurmanji)",
             "ZZA": "ZZA (Zazaki)", "HAW": "HAW (Hawrami / Gorani)",
             "EN": "English", "AR": "Arabic", "TR": "Turkish", "FA": "Persian"}
        parts = [p.strip() for p in re.split(r'[/,]', lang)] if any(s in lang for s in ["/", ","]) else [lang]
        return " / ".join(m.get(p, p) for p in parts)

    def classify(lang):
        if not lang: return "Other"
        if "CKB" in lang: return "CKB (Sorani)"
        if "KMR" in lang: return "KMR (Kurmanji)"
        if "ZZA" in lang: return "ZZA (Zazaki)"
        if "HAW" in lang: return "HAW (Hawrami/Gorani)"
        return "Other"

    os.makedirs(data_dir, exist_ok=True)
    out_fields = ["url", "domain", "title", "language_detected",
                  "kurdish_dialect_group", "status_code", "source",
                  "category", "notes", "checked_at"]

    all_out = []
    for row in rows:
        all_out.append({
            "url": row.get("url", ""), "domain": row.get("domain", ""),
            "title": row.get("title", ""),
            "language_detected": normalize_language(row.get("language", "")),
            "kurdish_dialect_group": classify(row.get("language", "")),
            "status_code": row.get("status_code", ""),
            "source": row.get("source", ""),
            "category": row.get("category", "general"),
            "notes": row.get("notes", ""),
            "checked_at": row.get("checked_at", ""),
        })
    all_out.sort(key=lambda x: (x["kurdish_dialect_group"], x["domain"]))

    with open(data_all, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(all_out)

    verified = [r for r in all_out
                if r["status_code"].startswith("2") or r["status_code"].startswith("3")]
    with open(data_verified, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(verified)

    ckb = sum(1 for r in rows if "CKB" in r.get("language", ""))
    kmr = sum(1 for r in rows if "KMR" in r.get("language", ""))
    zza = sum(1 for r in rows if "ZZA" in r.get("language", ""))
    haw = sum(1 for r in rows if "HAW" in r.get("language", ""))

    print(f"\n{'='*60}")
    print(f"  FINAL KURDISH WEBSITES REPORT")
    print(f"{'='*60}")
    print(f"  Total unique domains:    {len(rows)}")
    print(f"  Verified (reachable):    {len(verified)}")
    print(f"  % Reachable:             {len(verified)/len(rows)*100:.1f}%")
    print(f"\n  Kurdish Language Sites:")
    print(f"    CKB (Sorani):          {ckb}")
    print(f"    KMR (Kurmanji):        {kmr}")
    print(f"    ZZA (Zazaki):          {zza}")
    print(f"    HAW (Gorani):          {haw}")
    print(f"\n  Files:")
    print(f"    {complete_csv}")
    print(f"    {data_all}")
    print(f"    {data_verified}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
