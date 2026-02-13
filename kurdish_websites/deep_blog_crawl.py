#!/usr/bin/env python3
"""
Deep Blog/Author/Forum/Culture crawler — Round 3.
Targets Kurdish blogs, author sites, forums, literary magazines, film festivals,
food blogs, diaspora orgs, radio stations, sports, and more.
Also deep-crawls known blog/forum/literary sites for outbound links.
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
    "paypal.com", "stripe.com",
}

CKB_CHARS = set("ەێڕۆڵ")
CKB_KEYWORDS = [
    "هەواڵ", "کوردی", "کوردستان", "بەپەلە", "مووچە", "دەستوور",
    "پارلەمان", "حکوومەت", "ئەنجومەن", "زانکۆ", "فێرکردن",
    "بەشداری", "دامەزراوە", "وەزارەت", "بەڕێوەبردن", "تەندروستی",
    "پەروەردە", "ئابووری", "نەوت", "سیاسی", "کۆمەڵگا",
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
    if not html:
        return "UNK"
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
        scores["EN"] += min(tl.count(" " + kw + " ") * 0.3, 10)
    for kw in AR_KEYWORDS:
        if kw in text: scores["AR"] += 2
    for kw in TR_KEYWORDS:
        if kw.lower() in tl: scores["TR"] += 2
    for kw in FA_KEYWORDS:
        if kw in text: scores["FA"] += 2
    mx = max(scores.values())
    if mx < 3:
        return "UNK"
    r = [l for l, s in sorted(scores.items(), key=lambda x: -x[1])
         if s >= mx * 0.4 and s >= 3][:3]
    return "/".join(r) if r else "UNK"


def extract_title(html):
    try:
        soup = BeautifulSoup(html, "html.parser")
        t = soup.find("title")
        if t and t.string:
            return t.string.strip()[:200]
    except:
        pass
    return ""


def categorize(d):
    d = d.lower()
    if any(x in d for x in [".edu.", "university", "zanko", "college"]): return "education"
    if any(x in d for x in [".gov.", "parliament", "presidency"]): return "government"
    if any(x in d for x in ["news", "press", "media", "rudaw", "nrt"]): return "news_media"
    if any(x in d for x in ["tv", "radio", "fm", "sat", "channel"]): return "tv_radio"
    if any(x in d for x in ["hospital", "health", "medical"]): return "health"
    if any(x in d for x in ["shop", "store", "mall", "bazar"]): return "ecommerce"
    if any(x in d for x in ["tech", "dev", "software", "digital"]): return "technology"
    if any(x in d for x in ["law", "legal", "lawyer"]): return "legal"
    if any(x in d for x in ["ngo", "rights", "humanitarian"]): return "ngo"
    if any(x in d for x in ["travel", "tour", "hotel", "visit"]): return "tourism"
    if any(x in d for x in ["blog", "forum", "community", "board"]): return "blog_forum"
    if any(x in d for x in ["book", "library", "publish", "lit", "poem"]): return "literature"
    if any(x in d for x in ["film", "cinema", "movie", "festival"]): return "film_arts"
    if any(x in d for x in ["food", "recipe", "cook", "restaurant"]): return "food"
    if any(x in d for x in ["sport", "football", "fc", "club"]): return "sports"
    if any(x in d for x in ["music", "song", "art", "culture"]): return "culture_arts"
    return "general"


def extract_domain(url):
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        host = host.lower().strip(".")
        if host.startswith("www."):
            host = host[4:]
        return host
    except:
        return ""


def is_excluded(domain):
    for excl in EXCLUDE_DOMAINS:
        if domain == excl or domain.endswith("." + excl):
            return True
    return False


def crawl_deep(url, max_subpages=8):
    """Deep crawl a site for outbound links, going deeper for blog/forum sites."""
    all_found = set()
    base_domain = extract_domain(url)
    internal_urls = [url]

    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                            allow_redirects=True, verify=True)
        if resp.status_code != 200:
            return all_found
        soup = BeautifulSoup(resp.content, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            full = urljoin(url, href)
            d = extract_domain(full)
            if d and not is_excluded(d):
                if d == base_domain or d == "www." + base_domain:
                    if len(internal_urls) < max_subpages + 1:
                        internal_urls.append(full)
                else:
                    all_found.add(d)
    except:
        pass

    for sub_url in internal_urls[1:]:
        try:
            resp = requests.get(sub_url, headers=HEADERS, timeout=8,
                                allow_redirects=True, verify=True)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.content, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                    continue
                full = urljoin(sub_url, href)
                d = extract_domain(full)
                if d and not is_excluded(d) and d != base_domain:
                    all_found.add(d)
        except:
            continue

    return all_found


def verify(url, source="blog_crawl"):
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
        except:
            result["status_code"] = "SSL_ERR"
    except requests.exceptions.ConnectionError:
        result["status_code"] = "CONN_ERR"
    except requests.exceptions.Timeout:
        result["status_code"] = "TIMEOUT"
    except Exception as e:
        result["status_code"] = f"ERR:{type(e).__name__}"
    return result


# ─────────────────────────────────────────────────────────────
# ALL URLs from searches — Blogs, Authors, Forums, Culture...
# ─────────────────────────────────────────────────────────────
BLOG_AUTHOR_FORUM_URLS = [
    # Kurdish literary sites
    "https://kurdilit.net", "https://kurdishscholar.com",
    "https://kurdishstudies.net", "https://tishk.org",
    "https://jokl.uok.ac.ir", "https://echolibrary.org",
    "https://ideasbeyondborders.org", "https://kurdistanica.com",
    "https://lyricstranslate.com/en/collection/kurdish-poets",
    "https://justiceforkurds.org", "https://fromaltaytoyughur.blog",
    "https://wordswithoutborders.org",
    "https://iranicaonline.org",

    # Kurdish blogs (WordPress hosted - use their own domains)
    "https://kurdaily.com", "https://mandalawi.wordpress.com",
    "https://kurdistanblogcount.wordpress.com",
    "https://kurdii.wordpress.com", "https://kurdishwords.wordpress.com",
    "https://fidayeekurdistan.wordpress.com",
    "https://sana23kurd.wordpress.com", "https://handeran.wordpress.com",
    "https://paswecity.wordpress.com", "https://akurdishgirl.wordpress.com",
    "https://kurdishcooking.wordpress.com",
    "https://kurdishfood.home.blog",

    # Kurdish food blogs
    "https://sourandsweets.com", "https://kurdishbike.com",
    "https://globaldining.ch",

    # Kurdish film festivals
    "https://lkff.co.uk", "https://nykcc.org",
    "https://kurdsofillinois.org", "https://kurdisches-filmfestival.de",
    "https://zagrosfestival.com", "https://thefridacinema.org",

    # Kurdish radio streaming
    "https://oiradio.co", "https://radio.menu",
    "https://kuasark.com", "https://allradio.net",
    "https://liveonlineradio.net",
    "https://hit-tuner.net",

    # Kurdish sports
    "https://kurdistanboards.com", "https://babagol.net",
    "https://sofascore.com", "https://tablesleague.com",
    "https://kurdistan-fa.net",

    # Kurdish women's / feminist orgs
    "https://nlka.net", "https://womensvoicesnow.org",
    "https://intercoll.net", "https://defendrojava.org",
    "https://polarjournal.org",

    # Kurdish diaspora organizations
    "https://tnkcc.org", "https://kdctn.org",
    "https://unitedcitizensofeurope.com",

    # Kurdish language tech
    "https://kurdishtts.com", "https://kurd.ai",
    "https://kurditgroup.org", "https://translatiz.com",
    "https://zimannas.wordpress.com",

    # Kurdish publishers / bookstores
    "https://kitebhen.com", "https://shepherdsbooks.org",
    "https://alibris.com",

    # Kurdish academic / research
    "https://kii.krd", "https://sahipkiran.org",
    "https://merip.org", "https://jstribune.com",
    "https://kurdishacademy.org",

    # More Kurdish blogs and cultural sites
    "https://kurdistan-blog.com", "https://thekurdishblog.com",
    "https://kurdishaspect.com", "https://kurdmedia.com",
    "https://kurdistantribune.com", "https://ekurd.net",
    "https://peaceinkurdistancampaign.com",
    "https://kurdishquestion.com", "https://kurdistanmemorandum.com",
    "https://anfenglish.com", "https://anfarabic.com",
    "https://hawzhin.net", "https://zhiannews.com",
    "https://peyaminternational.com",

    # Kurdish Wikipedia projects
    "https://ku.wikipedia.org", "https://ckb.wikipedia.org",

    # Kurdish poetry / literature
    "https://helbest.com", "https://helbest.net",
    "https://chirok.net", "https://chirok.com",
    "https://weje.net", "https://weje.com",
    "https://nubiharkurdi.com", "https://nubihar.net",
    "https://nubihar.com",
    "https://avesta-kurd.net", "https://avestakurd.net",

    # Kurdish heritage / cultural preservation
    "https://kurdishheritage.org", "https://kurdishnews.com",
    "https://kurdishglobe.net", "https://kurds.eu",
    "https://thekurds.net", "https://kurdipress.com",

    # More from various searches
    "https://helbestvan.com", "https://romanekurdi.com",
    "https://pirtukxane.com", "https://pirtuk.org",
    "https://pirtukekurdi.com", "https://ktebxane.com",
    "https://ktebxana.com",

    # Kurdish forums and communities
    "https://forumkurd.com", "https://rojbashkurdistan.org",
    "https://kurdishandproud.com", "https://kurdistan-forum.com",
    "https://kurdishforum.com", "https://kurdforum.net",
    "https://kurd.cc", "https://kurd.io",
    "https://kurdnet.net",

    # More Kurdish media / news blogs
    "https://kurdpress.com", "https://kurdistanmedia.org",
    "https://basnews.com/ckb", "https://zhiannews.com",
    "https://kurdsat.tv", "https://gelawej.net",
    "https://gelawej.org", "https://berbang.net",
    "https://sterk.tv", "https://ronahitv.net",
    "https://jin.live", "https://jinha.com.tr",

    # Kurdish diaspora blogs in Europe
    "https://kurdishinstitute.be", "https://institut-kurde.org",
    "https://fikp.de", "https://navend.de",
    "https://cfrk.de", "https://kurdischer-verein.de",
    "https://yxk.de", "https://civaka-azad.org",
    "https://rfrk.de", "https://kurdistankomitee.de",

    # More specialized blogs
    "https://kurdishart.com", "https://kurdishmusic.com",
    "https://kurdishcinema.com", "https://kurdishtheatre.com",
    "https://dengbej.com", "https://dengbej.net",
    "https://dengbej.org", "https://dengbeji.com",

    # Kurdish Yezidi/Ezidi blogs and culture
    "https://ezidipress.com", "https://lalishcenter.com",
    "https://lalesh.com", "https://yezidipost.com",
    "https://yeziditruth.org", "https://ezidipost.com",
    "https://ezidikhan.com",

    # More author/literary blogs
    "https://bakhtiarali.com", "https://abdullapashew.com",
    "https://abdullapashew.net", "https://sherko-bekas.com",
    "https://sherkobekas.com", "https://ferhad-shakely.com",
    "https://marufbarznji.com", "https://hashem-ahmadzadeh.com",
    "https://mehmeduzun.com", "https://cigerxwin.com",
    "https://cigerxwin.net", "https://ehmedxani.com",
    "https://ehmedexani.com",

    # Kurdish calendar / holidays
    "https://newroz.com", "https://newroz.net", "https://newroz.org",
    "https://kurdishcalendar.com", "https://rojikurdi.com",

    # More KRG/institution blogs
    "https://krg.org", "https://dfr.gov.krd",
    "https://cabinet.gov.krd/blog", "https://mop.gov.krd",
    "https://moe.gov.krd",
]

# Sites to deep-crawl for outbound blog/author/forum links
DEEP_CRAWL_TARGETS = [
    "https://kurdilit.net",
    "https://kurdilit.net/?page_id=2466&lang=en",  # writers
    "https://kurdilit.net/?page_id=2497&lang=en",  # useful links
    "https://kurdilit.net/?page_id=1959&lang=en",  # publishers
    "https://kurdishscholar.com/literature/",
    "https://kurdishstudies.net",
    "https://ekurd.net",
    "https://kurdistantribune.com",
    "https://kurdistanboards.com",
    "https://nykcc.org",
    "https://thekurdishproject.org",
    "https://thekurdishproject.org/kurdish-nonprofits/",
    "https://koord.com",
    "https://c4kurd.com",
    "https://justiceforkurds.org",
    "https://kurdishpeace.org",
    "https://echolibrary.org",
    "https://ideasbeyondborders.org",
    "https://kurdaily.com",
    "https://100berhemenkurdi.com",
    "https://100berhemenkurdi.com/collection/music/",
    "https://100berhemenkurdi.com/collection/literature/",
    "https://lkff.co.uk",
    "https://zagrosfestival.com",
    "https://kurdisches-filmfestival.de",
    "https://sourandsweets.com/all-recipe/kurdish-food-recipes/",
    "https://kurdishfood.home.blog",
    "https://kurdishcentral.org",
    "https://nubiharkurdi.com",
    "https://ezidipress.com",
    "https://anfenglish.com",
]


def main():
    base_dir = os.path.dirname(__file__)
    complete_csv = os.path.join(base_dir, "kurdish_websites_complete.csv")

    # Load existing domains
    existing = {}
    if os.path.exists(complete_csv):
        with open(complete_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row.get("domain", "").lower()] = row
    print(f"[INFO] {len(existing)} existing entries.")

    # ─── Phase 1: Deep crawl blog/literary/forum targets ───
    print(f"\n[PHASE 1] Deep crawling {len(DEEP_CRAWL_TARGETS)} blog/literary/forum sites...")
    crawled_domains = set()
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(crawl_deep, url, 8): url for url in DEEP_CRAWL_TARGETS}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Deep crawl"):
            crawled_domains.update(future.result())

    # Filter for Kurdish relevance
    indicators = [
        "kurd", "hewler", "erbil", "sulaimani", "slemani", "duhok",
        "kirkuk", "halabja", "kobane", "afrin", "rojava", "newroz",
        "azadi", "peshmerga", "rudaw", "nrt", "basnews", "shafaq",
        "yezidi", "ezidi", "lalish", "zagros", "korek", "tishk",
        "helbest", "chirok", "weje", "dengbej", "govar",
        "pirtuk", "roman", "nubihar", "blog", "forum",
        ".krd", ".iq",
    ]
    relevant_crawled = {d for d in crawled_domains - set(existing.keys())
                        if any(i in d.lower() for i in indicators)
                        or d.endswith(".krd") or d.endswith(".iq")}
    print(f"  Crawled {len(crawled_domains)} total, {len(relevant_crawled)} new Kurdish-relevant.")

    # ─── Phase 2: Collect all new direct URLs ───
    print(f"\n[PHASE 2] Processing {len(BLOG_AUTHOR_FORUM_URLS)} direct URLs...")
    direct_domains = set()
    for url in BLOG_AUTHOR_FORUM_URLS:
        d = extract_domain(url)
        if d and d not in existing and not is_excluded(d):
            direct_domains.add(d)
    print(f"  {len(direct_domains)} new from direct URLs.")

    # ─── Phase 3: Combine and verify ───
    all_new = direct_domains | relevant_crawled
    print(f"\n[PHASE 3] Verifying {len(all_new)} new domains...")

    results = []
    urls_to_check = [f"https://{d}" for d in all_new]
    with ThreadPoolExecutor(max_workers=60) as pool:
        futures = {pool.submit(verify, url): url for url in urls_to_check}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Verifying"):
            r = f.result()
            if r["status_code"] not in ("CONN_ERR", "SSL_ERR", "TIMEOUT") and \
               not r["status_code"].startswith("ERR"):
                results.append(r)

    alive = [r for r in results if int(r["status_code"]) < 500]
    print(f"\n  {len(alive)} new alive domains!")

    # Merge into existing
    for r in alive:
        d = r["domain"]
        if d not in existing:
            existing[d] = r

    rows = sorted(existing.values(), key=lambda x: x.get("domain", ""))
    print(f"  Total dataset: {len(rows)}")

    # Write updated CSV
    fields = ["url", "domain", "title", "language", "status_code",
              "source", "category", "notes", "checked_at"]
    with open(complete_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

    # Write deliverable CSVs
    data_dir = os.path.join(os.path.dirname(base_dir), "data")
    os.makedirs(data_dir, exist_ok=True)
    data_all = os.path.join(data_dir, "kurdish_websites_all.csv")
    data_verified = os.path.join(data_dir, "kurdish_websites_verified.csv")

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

    # Category breakdown
    cat_counts = defaultdict(int)
    for r in rows:
        cat_counts[r.get("category", "general")] += 1

    ckb = sum(1 for r in rows if "CKB" in r.get("language", ""))
    kmr = sum(1 for r in rows if "KMR" in r.get("language", ""))
    zza = sum(1 for r in rows if "ZZA" in r.get("language", ""))
    haw = sum(1 for r in rows if "HAW" in r.get("language", ""))

    print(f"\n{'='*60}")
    print(f"  DEEP CRAWL REPORT — Blogs, Authors, Forums, Culture")
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
    print(f"\n  New entries this round: {len(alive)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
