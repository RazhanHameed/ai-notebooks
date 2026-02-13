#!/usr/bin/env python3
"""
Round 5: Deep Infrastructure Mining
Uses unconventional discovery methods:
1. Certificate Transparency logs (crt.sh) for .krd, .iq, and Kurdish subdomains
2. Wikidata SPARQL for Kurdish organization official websites
3. Passive DNS / reverse-IP lookups
4. Deep recursive crawling of ALL live sites (10 subpages each)
5. More aggressive Common Crawl with new indices and patterns
6. More social media and niche searches hardcoded
"""

import csv
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ku,ckb,en;q=0.5",
}
TIMEOUT = 10

EXCLUDE_DOMAINS = {
    "google.com", "youtube.com", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "tiktok.com", "linkedin.com", "reddit.com",
    "whatsapp.com", "telegram.org", "t.me", "apple.com", "play.google.com",
    "amazon.com", "wikipedia.org", "wikimedia.org",
    "wordpress.com", "blogspot.com", "blogger.com", "medium.com",
    "soundcloud.com", "spotify.com", "pinterest.com",
    "archive.org", "web.archive.org",
    "bit.ly", "goo.gl", "t.co", "ow.ly", "tinyurl.com",
    "cloudflare.com", "googleapis.com", "gstatic.com",
    "w3.org", "schema.org", "wp.com", "gravatar.com",
    "googlesyndication.com", "doubleclick.net", "googletagmanager.com",
    "google-analytics.com", "fontawesome.com",
    "microsoft.com", "yahoo.com", "bing.com", "baidu.com",
    "github.com", "gitlab.com", "amazonaws.com",
    "apps.apple.com", "flickr.com", "vimeo.com", "dailymotion.com",
    "paypal.com", "stripe.com", "discord.com", "discord.gg",
    "open.spotify.com", "podcasts.apple.com", "substack.com",
    "godaddy.com", "namecheap.com", "sedo.com", "hugedomains.com",
    "afternic.com", "dan.com",
}

CKB_CHARS = set("ەێڕۆڵ")
CKB_KW = ["هەواڵ", "کوردی", "کوردستان", "بەپەلە", "مووچە", "دەستوور",
           "پارلەمان", "حکوومەت", "زانکۆ", "فێرکردن", "بەشداری",
           "دامەزراوە", "وەزارەت", "تەندروستی", "پەروەردە", "ئابووری"]
KMR_CHARS = set("êîûşçÊÎÛŞÇ")
KMR_KW = ["Kurdî", "Kurdistanê", "Rojava", "Bakur", "Başûr", "azadî",
           "nûçe", "welatê", "gelê", "mafê", "demokrasî", "civakî",
           "jin", "jiyan", "azadiya", "berxwedanê", "Hewlêr", "Kobanê",
           "Efrîn", "pêşmerge", "newroz"]
ZZA_KW = ["Zazaki", "Dimilî", "Kirmancki", "Dersim"]
HAW_KW = ["هەورامی", "گۆرانی", "Hawrami", "Gorani"]
EN_KW = ["the", "and", "for", "that", "with", "this", "from"]
AR_KW = ["في", "من", "على", "إلى", "عن", "هذا"]
TR_KW = ["bir", "için", "olan", "ile", "gibi", "haber"]
FA_KW = ["است", "این", "برای", "های", "ایران"]


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
    scores["CKB"] += min(sum(1 for c in text if c in CKB_CHARS) * 0.05, 30)
    scores["KMR"] += min(sum(1 for c in text if c in KMR_CHARS) * 0.03, 20)
    tl = text.lower()
    for kw in CKB_KW:
        if kw in text: scores["CKB"] += 3
    for kw in KMR_KW:
        if kw.lower() in tl: scores["KMR"] += 3
    for kw in ZZA_KW:
        if kw.lower() in tl: scores["ZZA"] += 4
    for kw in HAW_KW:
        if kw.lower() in tl or kw in text: scores["HAW"] += 4
    for kw in EN_KW:
        scores["EN"] += min(tl.count(" " + kw + " ") * 0.3, 10)
    for kw in AR_KW:
        if kw in text: scores["AR"] += 2
    for kw in TR_KW:
        if kw.lower() in tl: scores["TR"] += 2
    for kw in FA_KW:
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
    cats = [
        ([".edu.", "university", "zanko", "college", "academy"], "education"),
        ([".gov.", "parliament", "presidency", "ministry"], "government"),
        (["news", "press", "media", "rudaw", "nrt", "hawar", "nuce", "journal"], "news_media"),
        (["tv", "radio", "fm", "sat", "channel", "stream"], "tv_radio"),
        (["hospital", "health", "medical", "clinic", "pharma"], "health"),
        (["shop", "store", "mall", "bazar", "market", "buy"], "ecommerce"),
        (["tech", "dev", "software", "digital", "code", "app"], "technology"),
        (["law", "legal", "lawyer", "attorney"], "legal"),
        (["ngo", "rights", "humanitarian", "aid"], "ngo"),
        (["travel", "tour", "hotel", "visit", "tourism"], "tourism"),
        (["blog", "forum", "community", "board"], "blog_forum"),
        (["book", "library", "publish", "lit", "poem", "pirtuk"], "literature"),
        (["film", "cinema", "movie", "festival", "gallery", "photo"], "film_arts"),
        (["food", "recipe", "cook", "restaurant", "catering"], "food"),
        (["sport", "football", "fc", "club", "game"], "sports"),
        (["music", "song", "art", "culture", "dengbej"], "culture_arts"),
        (["podcast", "audio"], "podcast"),
        (["dict", "translat", "ferheng"], "language_tool"),
    ]
    for keywords, cat in cats:
        if any(x in d for x in keywords): return cat
    return "general"


def extract_domain(url):
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        host = host.lower().strip(".")
        if host.startswith("www."): host = host[4:]
        return host
    except: return ""


def is_excluded(domain):
    for excl in EXCLUDE_DOMAINS:
        if domain == excl or domain.endswith("." + excl):
            return True
    return False


# ══════════════════════════════════════════════════════════════
# PHASE 1: Certificate Transparency Logs (crt.sh)
# ══════════════════════════════════════════════════════════════
def query_crtsh(query):
    """Query crt.sh for certificate transparency records."""
    domains = set()
    try:
        url = f"https://crt.sh/?q={query}&output=json"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            records = resp.json()
            for rec in records:
                name = rec.get("name_value", "")
                for line in name.split("\n"):
                    d = line.strip().lower()
                    if d.startswith("*."): d = d[2:]
                    if d and not is_excluded(d) and "." in d:
                        domains.add(d)
    except Exception as e:
        print(f"  crt.sh error for {query}: {e}")
    return domains


# ══════════════════════════════════════════════════════════════
# PHASE 2: Wikidata SPARQL
# ══════════════════════════════════════════════════════════════
def query_wikidata():
    """Query Wikidata for Kurdish organizations with official websites."""
    domains = set()
    sparql_url = "https://query.wikidata.org/sparql"

    queries = [
        # Kurdish organizations
        """SELECT ?website WHERE {
          ?item wdt:P31/wdt:P279* wd:Q43229 .
          ?item wdt:P17 wd:Q3504 .
          ?item wdt:P856 ?website .
        } LIMIT 500""",
        # Kurdish newspapers
        """SELECT ?website WHERE {
          ?item wdt:P31/wdt:P279* wd:Q11032 .
          ?item wdt:P407 wd:Q36368 .
          ?item wdt:P856 ?website .
        } LIMIT 200""",
        # Kurdistan Region entities
        """SELECT ?website WHERE {
          ?item wdt:P131 wd:Q205047 .
          ?item wdt:P856 ?website .
        } LIMIT 500""",
        # Kurdish language websites
        """SELECT ?website WHERE {
          ?item wdt:P407 wd:Q36368 .
          ?item wdt:P856 ?website .
        } LIMIT 500""",
        # Sorani Kurdish language sites
        """SELECT ?website WHERE {
          ?item wdt:P407 wd:Q36811 .
          ?item wdt:P856 ?website .
        } LIMIT 200""",
        # Entities located in Erbil
        """SELECT ?website WHERE {
          ?item wdt:P131 wd:Q144800 .
          ?item wdt:P856 ?website .
        } LIMIT 500""",
        # Entities located in Sulaymaniyah
        """SELECT ?website WHERE {
          ?item wdt:P131 wd:Q192932 .
          ?item wdt:P856 ?website .
        } LIMIT 500""",
        # Entities located in Duhok
        """SELECT ?website WHERE {
          ?item wdt:P131 wd:Q170435 .
          ?item wdt:P856 ?website .
        } LIMIT 300""",
        # Kurdish people with official websites
        """SELECT ?website WHERE {
          ?item wdt:P172 wd:Q188652 .
          ?item wdt:P856 ?website .
        } LIMIT 500""",
    ]

    for q in queries:
        try:
            resp = requests.get(sparql_url, params={"query": q, "format": "json"}, timeout=30,
                                headers={"User-Agent": "KurdishWebDiscovery/1.0"})
            if resp.status_code == 200:
                data = resp.json()
                for binding in data.get("results", {}).get("bindings", []):
                    url = binding.get("website", {}).get("value", "")
                    d = extract_domain(url)
                    if d and not is_excluded(d):
                        domains.add(d)
            time.sleep(1)
        except Exception as e:
            print(f"  Wikidata error: {e}")
    return domains


# ══════════════════════════════════════════════════════════════
# PHASE 3: Deep recursive crawl of ALL live sites
# ══════════════════════════════════════════════════════════════
def crawl_outbound(url, max_subpages=5):
    """Crawl a site for outbound links."""
    found = set()
    base_domain = extract_domain(url)
    internal_urls = [url]
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8, allow_redirects=True, verify=True)
        if resp.status_code != 200: return found
        soup = BeautifulSoup(resp.content, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")): continue
            full = urljoin(url, href)
            d = extract_domain(full)
            if d and not is_excluded(d):
                if d == base_domain or d == "www." + base_domain:
                    if len(internal_urls) < max_subpages + 1: internal_urls.append(full)
                else: found.add(d)
    except: pass

    for sub in internal_urls[1:]:
        try:
            resp = requests.get(sub, headers=HEADERS, timeout=6, allow_redirects=True, verify=True)
            if resp.status_code != 200: continue
            soup = BeautifulSoup(resp.content, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")): continue
                full = urljoin(sub, href)
                d = extract_domain(full)
                if d and not is_excluded(d) and d != base_domain: found.add(d)
        except: continue
    return found


def verify(url, source="infra"):
    domain = extract_domain(url)
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


# ══════════════════════════════════════════════════════════════
# PHASE 4: More Common Crawl with fresh indices
# ══════════════════════════════════════════════════════════════
def query_cc(query, index):
    domains = set()
    try:
        api = f"https://index.commoncrawl.org/{index}-index"
        resp = requests.get(api, params={"url": query, "output": "json", "limit": 500}, timeout=30)
        if resp.status_code == 200:
            for line in resp.text.strip().split("\n"):
                if line:
                    try:
                        rec = json.loads(line)
                        d = extract_domain(rec.get("url", ""))
                        if d and not is_excluded(d): domains.add(d)
                    except: pass
    except: pass
    return domains


# Extra hardcoded URLs from latest social media mining
EXTRA_URLS = [
    # Kurdish gaming / esports
    "https://gamekurdistan.com", "https://kurdgame.com",
    "https://eslkurdistan.com",
    # Kurdish wedding / events
    "https://kurdishwedding.com", "https://afrethall.com",
    "https://royalgardenhall.com", "https://divanerbil.com",
    # Kurdish banks / finance
    "https://kurdistanbank.com", "https://rtbank.iq",
    "https://aib.iq", "https://boi.gov.iq",
    "https://cbi.iq", "https://tbibank.com",
    "https://kurdistanintlbank.com", "https://nbi.iq",
    "https://ashurbank.com", "https://iqi.bank",
    "https://byblosbankiraq.com", "https://iraqiislamicbank.com",
    # Kurdish ISPs / telecom
    "https://korek.com", "https://asiacell.com",
    "https://newroz-telecom.com", "https://iq.zain.com",
    "https://earthlink.iq", "https://gorannet.com",
    "https://tarin.net", "https://fastlink.com",
    "https://fastiraq.com", "https://kurdtel.com",
    # Kurdish car dealers / auto
    "https://autokurdistan.com", "https://iqmotors.com",
    "https://toyotairaq.com", "https://kiaiq.com",
    # Kurdish insurance
    "https://igi-iraq.com", "https://aliraqiains.com",
    # Kurdish construction / real estate
    "https://massgroup.co", "https://qaiwan.com",
    "https://hiperlife.com", "https://empireworld.iq",
    "https://dreamcity.iq", "https://alwahagroup.iq",
    "https://daringroup.com", "https://itamar.iq",
    "https://majidi.co", "https://damac.iq",
    # Kurdish restaurants / cafes
    "https://familymallerbil.com", "https://majidimall.com",
    "https://citymall.iq",
    # Kurdish airlines / transport
    "https://kurdistanairlines.com", "https://flyiraq.com",
    "https://iraqiairways.com.iq",
    # Kurdish weather / maps
    "https://weatheriq.me", "https://krdmap.com",
    # More Kurdish NGOs found via social media
    "https://heartland-alliance.org", "https://nca.no",
    "https://warchild.org", "https://mercy-corps.org",
    # Kurdish sports clubs
    "https://erbilsc.com", "https://sulemaniyahfc.com",
    "https://duhoksc.com", "https://zakhofc.com",
    # More diaspora / community
    "https://kurdistan-s.com", "https://kurdi.tv",
    "https://kurdistanobserver.com", "https://thekurdist.com",
    "https://kurdnetwork.com", "https://kurdishnow.com",
    "https://nujem.com",
    # Kurdish humor/entertainment
    "https://kurdishcomedy.com", "https://gorani.me",
    # More Kurdish academic journals
    "https://zfrjournal.com", "https://sjkurd.com",
    "https://garmian.edu.krd/journal", "https://kujss.koyauniversity.org",
    "https://jzsw.koyauniversity.org",
    # Kurdish children / education
    "https://zaroktv.com", "https://zarokchannel.com",
    "https://kurdishchildren.org",
    # Kurdish news aggregators
    "https://kurdishnews.org", "https://allkurd.net",
    "https://kurdtimes.com", "https://kurdsatan.com",
    "https://kurdistanmoment.com",
    # More found via social media deep dive
    "https://yekiti-media.org", "https://hemukurd.com",
    "https://hyvimind.com", "https://thelynk.com",
    "https://snapfood.ir", "https://carmania.me",
    "https://bama.ir", "https://sheylan.com",
    "https://zhiyan.app", "https://mamland.iq",
    "https://barzinji.org", "https://agiriman.com",
    "https://topzana.com", "https://chia.iq",
    "https://brgr.iq",
    "https://payzag.com", "https://studioaristotle.com",
    "https://baznegroup.com", "https://newroztravel.com",
    "https://safareer.com", "https://erbilhomes.com",
    "https://erbilrealestate.com", "https://houseiq.com",
    # Diaspora organizations and cultural centers
    "https://navenda-nubihar.com", "https://rfrk.de",
    "https://mesop.de", "https://kurdwatch.org",
    "https://feskurdistan.com", "https://navdem.com",
    "https://kurdistanmemorial.com",
]


def main():
    base_dir = os.path.dirname(__file__)
    complete_csv = os.path.join(base_dir, "kurdish_websites_complete.csv")

    # Load existing
    existing = {}
    live_urls = []
    if os.path.exists(complete_csv):
        with open(complete_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = row.get("domain", "").lower()
                existing[d] = row
                sc = row.get("status_code", "")
                if sc.startswith("2"):
                    live_urls.append(row["url"])
    print(f"[INFO] {len(existing)} existing, {len(live_urls)} live.")

    all_new = set()

    # ─── Phase 1: Certificate Transparency ───
    print(f"\n[PHASE 1] Mining Certificate Transparency logs...")
    ct_queries = ["%.krd", "%.edu.krd", "%.gov.krd",
                  "%kurdistan%", "%kurd%", "%hewler%",
                  "%erbil%", "%sulaimani%", "%duhok%",
                  "%rojava%", "%kobane%", "%newroz%"]
    ct_domains = set()
    for q in tqdm(ct_queries, desc="crt.sh queries"):
        ct_domains.update(query_crtsh(q))
        time.sleep(2)
    new_ct = ct_domains - set(existing.keys())
    print(f"  Found {len(ct_domains)} total, {len(new_ct)} new from crt.sh")
    all_new.update(new_ct)

    # ─── Phase 2: Wikidata ───
    print(f"\n[PHASE 2] Querying Wikidata SPARQL...")
    wd_domains = query_wikidata()
    new_wd = wd_domains - set(existing.keys()) - all_new
    print(f"  Found {len(wd_domains)} total, {len(new_wd)} new from Wikidata")
    all_new.update(new_wd)

    # ─── Phase 3: Deep recursive crawl of ALL live sites ───
    print(f"\n[PHASE 3] Recursive crawling {len(live_urls)} live sites...")
    crawled_all = set()
    with ThreadPoolExecutor(max_workers=40) as pool:
        futures = {pool.submit(crawl_outbound, url, 5): url for url in live_urls}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Recursive crawl"):
            crawled_all.update(f.result())

    indicators = [
        "kurd", "hewler", "erbil", "sulaimani", "slemani", "duhok",
        "kirkuk", "halabja", "kobane", "afrin", "rojava", "newroz",
        "azadi", "peshmerga", "rudaw", "nrt", "basnews", "shafaq",
        "yezidi", "ezidi", "lalish", "zagros", "korek", "tishk",
        "helbest", "chirok", "weje", "dengbej", "govar",
        "pirtuk", "nubihar", "blog", "forum", "podcast",
        ".krd", ".iq",
    ]
    new_crawled = {d for d in crawled_all - set(existing.keys()) - all_new
                   if any(i in d.lower() for i in indicators)
                   or d.endswith(".krd") or d.endswith(".iq")}
    print(f"  Crawled {len(crawled_all)} total, {len(new_crawled)} new Kurdish-relevant")
    all_new.update(new_crawled)

    # ─── Phase 4: More Common Crawl ───
    print(f"\n[PHASE 4] Common Crawl with fresh indices...")
    cc_queries = [
        "*.krd/*", "*.edu.krd/*", "*.gov.krd/*",
        "*kurd*.com/*", "*kurd*.org/*", "*kurd*.net/*",
        "*erbil*.com/*", "*hewler*/*",
        "*sulaimani*/*", "*duhok*/*",
        "*rojava*/*", "*newroz*/*",
        "*halabja*/*", "*kirkuk*/*",
    ]
    cc_indices = ["CC-MAIN-2025-08", "CC-MAIN-2025-05", "CC-MAIN-2024-51",
                  "CC-MAIN-2024-46", "CC-MAIN-2024-42", "CC-MAIN-2024-38"]
    cc_found = set()
    total = len(cc_queries) * len(cc_indices)
    with tqdm(total=total, desc="CC queries") as pbar:
        for q in cc_queries:
            for idx in cc_indices:
                cc_found.update(query_cc(q, idx))
                pbar.update(1)
                time.sleep(0.3)
    new_cc = cc_found - set(existing.keys()) - all_new
    print(f"  Found {len(cc_found)} total, {len(new_cc)} new from Common Crawl")
    all_new.update(new_cc)

    # ─── Phase 5: Extra hardcoded URLs ───
    print(f"\n[PHASE 5] Adding {len(EXTRA_URLS)} hardcoded URLs...")
    for url in EXTRA_URLS:
        d = extract_domain(url)
        if d and d not in existing and not is_excluded(d):
            all_new.add(d)

    print(f"\n[TOTAL NEW] {len(all_new)} domains to verify")

    # ─── Phase 6: Verify everything ───
    print(f"\n[PHASE 6] Verifying {len(all_new)} domains...")
    results = []
    urls_to_check = [f"https://{d}" for d in all_new]
    with ThreadPoolExecutor(max_workers=80) as pool:
        futures = {pool.submit(verify, url): url for url in urls_to_check}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Verifying"):
            r = f.result()
            if r["status_code"] not in ("CONN_ERR", "SSL_ERR", "TIMEOUT") and \
               not r["status_code"].startswith("ERR"):
                results.append(r)

    alive = [r for r in results if int(r["status_code"]) < 500]
    print(f"\n  {len(alive)} new alive domains!")

    # Merge
    for r in alive:
        d = r["domain"]
        if d not in existing: existing[d] = r

    rows = sorted(existing.values(), key=lambda x: x.get("domain", ""))
    print(f"  Total dataset: {len(rows)}")

    # Write
    fields = ["url", "domain", "title", "language", "status_code",
              "source", "category", "notes", "checked_at"]
    with open(complete_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

    # Deliverables
    data_dir = os.path.join(os.path.dirname(base_dir), "data")
    os.makedirs(data_dir, exist_ok=True)
    data_all = os.path.join(data_dir, "kurdish_websites_all.csv")
    data_verified = os.path.join(data_dir, "kurdish_websites_verified.csv")

    def norm_lang(lang):
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

    out_fields = ["url", "domain", "title", "language_detected",
                  "kurdish_dialect_group", "status_code", "source",
                  "category", "notes", "checked_at"]
    all_out = []
    for row in rows:
        all_out.append({
            "url": row.get("url", ""), "domain": row.get("domain", ""),
            "title": row.get("title", ""),
            "language_detected": norm_lang(row.get("language", "")),
            "kurdish_dialect_group": classify(row.get("language", "")),
            "status_code": row.get("status_code", ""),
            "source": row.get("source", ""),
            "category": row.get("category", "general"),
            "notes": row.get("notes", ""),
            "checked_at": row.get("checked_at", ""),
        })
    all_out.sort(key=lambda x: (x["kurdish_dialect_group"], x["category"], x["domain"]))

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

    # Stats
    cat_counts = defaultdict(int)
    for r in rows: cat_counts[r.get("category", "general")] += 1

    ckb = sum(1 for r in rows if "CKB" in r.get("language", ""))
    kmr = sum(1 for r in rows if "KMR" in r.get("language", ""))
    zza = sum(1 for r in rows if "ZZA" in r.get("language", ""))
    haw = sum(1 for r in rows if "HAW" in r.get("language", ""))

    print(f"\n{'='*60}")
    print(f"  INFRASTRUCTURE MINING REPORT — Round 5")
    print(f"{'='*60}")
    print(f"  Total unique domains:    {len(rows)}")
    print(f"  Verified (reachable):    {len(verified)}")
    print(f"  % Reachable:             {len(verified)/len(rows)*100:.1f}%")
    print(f"\n  Kurdish Languages:")
    print(f"    CKB (Sorani):          {ckb}")
    print(f"    KMR (Kurmanji):        {kmr}")
    print(f"    ZZA (Zazaki):          {zza}")
    print(f"    HAW (Gorani):          {haw}")
    print(f"\n  By Category:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:25s} {cnt}")
    print(f"\n  New entries this round:  {len(alive)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
