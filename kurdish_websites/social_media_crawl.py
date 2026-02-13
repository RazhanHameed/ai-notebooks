#!/usr/bin/env python3
"""
Round 4: Social Media Deep Dive
Mines URLs discovered via Reddit, Quora, Telegram directories, Substack,
Linktree, Discord communities, podcasts, startups, dictionaries,
photography/art, and more niche content sites.
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
TIMEOUT = 12

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
        (["news", "press", "media", "rudaw", "nrt", "hawar", "nuce"], "news_media"),
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
        (["sport", "football", "fc", "club", "game", "esport"], "sports"),
        (["music", "song", "art", "culture", "dengbej", "stran"], "culture_arts"),
        (["podcast", "audio", "listen"], "podcast"),
        (["dict", "translate", "ferheng", "wergêr"], "language_tool"),
        (["wedding", "event", "party", "hall"], "services"),
        (["startup", "incubat", "accelerat", "innovat"], "startup"),
    ]
    for keywords, cat in cats:
        if any(x in d for x in keywords):
            return cat
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


def crawl_deep(url, max_subpages=10):
    """Deep crawl with more subpages for social/directory sites."""
    all_found = set()
    base_domain = extract_domain(url)
    internal_urls = [url]
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                            allow_redirects=True, verify=True)
        if resp.status_code != 200: return all_found
        soup = BeautifulSoup(resp.content, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")): continue
            full = urljoin(url, href)
            d = extract_domain(full)
            if d and not is_excluded(d):
                if d == base_domain or d == "www." + base_domain:
                    if len(internal_urls) < max_subpages + 1:
                        internal_urls.append(full)
                else:
                    all_found.add(d)
    except: pass

    for sub_url in internal_urls[1:]:
        try:
            resp = requests.get(sub_url, headers=HEADERS, timeout=8,
                                allow_redirects=True, verify=True)
            if resp.status_code != 200: continue
            soup = BeautifulSoup(resp.content, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")): continue
                full = urljoin(sub_url, href)
                d = extract_domain(full)
                if d and not is_excluded(d) and d != base_domain:
                    all_found.add(d)
        except: continue
    return all_found


def verify(url, source="social_media"):
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
# ALL URLs from social media mining
# ══════════════════════════════════════════════════════════════
SOCIAL_MEDIA_URLS = [
    # ── Substack newsletters ──
    "https://kurdistanwatch.substack.com",
    "https://kurdistaninsight.substack.com",
    "https://diasporafeedbackloops.substack.com",
    "https://danielclarkeserret.substack.com",

    # ── Podcasts ──
    "https://thekurdishedition.com",
    "https://kurdishtts.com",
    "https://podcast.feedspot.com/kurdish_podcasts/",

    # ── From Peywend.org directory ──
    "https://rojavainformationcenter.org",
    "https://npasyria.com",
    "https://hawarnews.com",
    "https://anfenglishmobile.com",
    "https://mezopotamyaajansi.net",
    "https://kurdarchive.net",
    "https://kurdishprogress.com",
    "https://dckurd.org",
    "https://kurdishinstitute.be",
    "https://aanes-rep-eu.com",
    "https://civaka-azad.org",
    "https://ceni-frauen.org",
    "https://kurdistanhumanrights.org",
    "https://hskurd.org",
    "https://kurd-akad.com",
    "https://peywend.org",

    # ── Kurdish periodicals / magazines ──
    "https://kurdistanchronicle.com",
    "https://kurdishstudies.net",
    "https://tishk.org",
    "https://hevseltimes.org",
    "https://themarkaz.org",

    # ── Kurdish apps / language tools ──
    "https://kurdim.app",
    "https://bimus.app",
    "https://ferheng.org",
    "https://lexilogos.com",
    "https://glosbe.com",
    "https://translatiz.com",
    "https://openl.io",
    "https://tirsik.com",
    "https://dictbox.com",
    "https://50languages.com",
    "https://ai.glossika.com",
    "https://weltsprachen.net",

    # ── Kurdish startups (from IraqTech) ──
    "https://iraqtech.io",
    "https://padash.app",
    "https://bonlili.com",
    "https://midient.com",
    "https://kurdivia.com",
    "https://shiffer.co",
    "https://enzziro.com",
    "https://hitex.tech",
    "https://tracxn.com",
    "https://kii.krd",
    "https://saydo.iq",

    # ── Kurdish photography / art / culture ──
    "https://kurdistan.photoshelter.com",
    "https://kcac.org",
    "https://picture-projects.com",
    "https://masonexhibitions.org",
    "https://saradistribution.com",
    "https://akakurdistan.com",

    # ── Kurdish diaspora organizations ──
    "https://demparti.org.tr",
    "https://servantgroup.org",
    "https://moving-jack.com",
    "https://videobabylon.ca",

    # ── Discord communities (websites, not discord links) ──
    "https://kurdlabs.com",
    "https://kurdscape.com",

    # ── Telegram directories ──
    "https://tdirectory.me",
    "https://telegramchannels.me",
    "https://tgstat.com",
    "https://tgdr.io",
    "https://xtea.io",
    "https://nicegram.app",

    # ── Kurdish learning platforms ──
    "https://preply.com",
    "https://kurdishcentral.org",

    # ── More from social media posts ──
    "https://kurdchronicle.com",
    "https://kurdishstruggle.com",
    "https://kurdistan-report.de",
    "https://kurdischegemeinden.de",
    "https://yek-dem.org",
    "https://civaka-azad.org",
    "https://nav-dem.com",
    "https://kongreya-star.org",
    "https://jinwar.org",
    "https://wjafreedom.org",
    "https://ronahitv.com",
    "https://zarok-tv.com",
    "https://jiyanfm.com",
    "https://sharpress.net",
    "https://hawlernews.com",
    "https://awaz.krd",
    "https://zaxo.net",
    "https://zakhopress.com",
    "https://hevaltv.com",
    "https://speda-tv.com",
    "https://kurdistanmedia.com",
    "https://peyamner.com",
    "https://pukmedia.com",

    # ── Kurdish real estate / business ──
    "https://sardam.info",
    "https://realestateiq.com",
    "https://erbilproperty.com",
    "https://makanik.iq",
    "https://miswag.com",
    "https://alsaree3.com",
    "https://zainiraq.com",
    "https://asiahawler.net",
    "https://apkurd.com",
    "https://kosar.tv",
    "https://govarivejin.com",
    "https://vejin.net",
    "https://welateparezi.org",
    "https://rizgari.com",
    "https://kurdistanpost.eu",
    "https://tofriq.com",
    "https://tapvrr.com",
    "https://sbeiy.com",
    "https://pedam.app",
    "https://snapptrip.com",
    "https://flyerbil.com",
    "https://erbilairport.net",
    "https://sulairport.com",

    # ── More diaspora blogs and community sites ──
    "https://kurdmanga.com",
    "https://mangakurd.com",
    "https://kurdishcinema.com",
    "https://mesopotamiacinema.com",
    "https://duhokiff.com",
    "https://erbilfilmfestival.com",
    "https://shofilm.co",
    "https://kurdishmovie.com",
    "https://kurdfilm.com",

    # ── Kurdish language Wikipedia projects ──
    "https://ku.wiktionary.org",
    "https://ckb.wiktionary.org",

    # ── More Kurdish cultural / community sites ──
    "https://kurdishmusicarchive.com",
    "https://kurdisharts.org",
    "https://kurdipedia.org",
    "https://mesopotamya.com",
    "https://firatajans.com",
    "https://jiyan.org",
    "https://jiyanfoundation.org",
    "https://asuda.org",
    "https://emma-organization.org",
    "https://harikar.org",
    "https://rwangafoundation.org",
    "https://reach-iraq.org",
    "https://zhianglobal.com",
    "https://dfrk.org",
    "https://pad-kurd.org",
    "https://weqfa-welat.de",
    "https://zeynelabidin.org",
    "https://kobanireconstruction.org",
    "https://haukari.de",
    "https://medico.de",

    # ── Kurdish newspapers (from Wikipedia list) ──
    "https://hawlati.co",
    "https://awene.com",
    "https://xendan.org",
    "https://sharq.net",
    "https://regay.net",
    "https://azadipress.net",
    "https://levin.media",
    "https://nubun.net",
    "https://ronahi.net",
    "https://welat.com",
    "https://yeniozgurpolitika.net",
    "https://xwebun.org",
    "https://nujiyan.net",
    "https://jiyanpress.com",
    "https://bernamegeh.com",

    # ── Kurdish influencer / content platforms ──
    "https://klear.com",
    "https://starngage.com",
    "https://favikon.com",
]

# Sites to deep-crawl for outbound links
DEEP_CRAWL_TARGETS = [
    "https://peywend.org/links-to-kurdish-websites/",
    "https://iraqtech.io/iraqi-kurdistan/",
    "https://iraqtech.io/10-startups-in-iraq-you-need-to-know-about/",
    "https://iraqtech.io/12-iraqi-and-kurdish-startups-in-takween-accelerator/",
    "https://koord.com",
    "https://c4kurd.com",
    "https://kurdistanchronicle.com",
    "https://kcac.org/en/",
    "https://rojavainformationcenter.org",
    "https://podcast.feedspot.com/kurdish_podcasts/",
    "https://lexilogos.com/english/kurdish_dictionary.htm",
    "https://themarkaz.org/contemporary-kurdish-writers-in-the-diaspora/",
    "https://hevseltimes.org",
    "https://hitex.tech",
    "https://kurdarchive.net",
    "https://anfenglishmobile.com",
    "https://thekurdishedition.com",
    "https://peywend.org",
]


def main():
    base_dir = os.path.dirname(__file__)
    complete_csv = os.path.join(base_dir, "kurdish_websites_complete.csv")

    # Load existing
    existing = {}
    if os.path.exists(complete_csv):
        with open(complete_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row.get("domain", "").lower()] = row
    print(f"[INFO] {len(existing)} existing entries.")

    # ─── Phase 1: Deep crawl social/directory sites ───
    print(f"\n[PHASE 1] Deep crawling {len(DEEP_CRAWL_TARGETS)} social/directory sites...")
    crawled_domains = set()
    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {pool.submit(crawl_deep, url, 10): url for url in DEEP_CRAWL_TARGETS}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Deep crawl"):
            crawled_domains.update(future.result())

    indicators = [
        "kurd", "hewler", "erbil", "sulaimani", "slemani", "duhok",
        "kirkuk", "halabja", "kobane", "afrin", "rojava", "newroz",
        "azadi", "peshmerga", "rudaw", "nrt", "basnews", "shafaq",
        "yezidi", "ezidi", "lalish", "zagros", "korek", "tishk",
        "helbest", "chirok", "weje", "dengbej", "govar",
        "pirtuk", "roman", "nubihar", "blog", "forum",
        "podcast", "substack", "newsletter",
        ".krd", ".iq",
    ]
    relevant_crawled = {d for d in crawled_domains - set(existing.keys())
                        if any(i in d.lower() for i in indicators)
                        or d.endswith(".krd") or d.endswith(".iq")}
    print(f"  Crawled {len(crawled_domains)} total, {len(relevant_crawled)} new Kurdish-relevant.")

    # ─── Phase 2: Collect direct URLs ───
    print(f"\n[PHASE 2] Processing {len(SOCIAL_MEDIA_URLS)} direct URLs...")
    direct_domains = set()
    for url in SOCIAL_MEDIA_URLS:
        d = extract_domain(url)
        if d and d not in existing and not is_excluded(d):
            direct_domains.add(d)
    print(f"  {len(direct_domains)} new from direct URLs.")

    # ─── Phase 3: Verify all ───
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

    # Merge
    for r in alive:
        d = r["domain"]
        if d not in existing:
            existing[d] = r

    rows = sorted(existing.values(), key=lambda x: x.get("domain", ""))
    print(f"  Total dataset: {len(rows)}")

    # Write complete CSV
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
    for r in rows:
        cat_counts[r.get("category", "general")] += 1

    ckb = sum(1 for r in rows if "CKB" in r.get("language", ""))
    kmr = sum(1 for r in rows if "KMR" in r.get("language", ""))
    zza = sum(1 for r in rows if "ZZA" in r.get("language", ""))
    haw = sum(1 for r in rows if "HAW" in r.get("language", ""))

    print(f"\n{'='*60}")
    print(f"  SOCIAL MEDIA DEEP DIVE REPORT — Round 4")
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
