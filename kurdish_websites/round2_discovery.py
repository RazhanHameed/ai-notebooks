#!/usr/bin/env python3
"""
Aggressive Kurdish Website Discovery - Round 2
Combines all search results with deep multi-page crawling,
aggressive Common Crawl mining, and more domain variations.
"""

import csv
import json
import os
import re
import sys
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
}

# ──────────────────────────────────────────────────────────────
# NEW URLs from web searches (manually extracted from results)
# ──────────────────────────────────────────────────────────────
NEW_SEARCH_URLS = [
    # Kurdish news - from search results
    "https://krd.hathalyoum.net", "https://english.hathalyoum.net",
    "https://nerinaazad2.com", "https://rojavatv.net", "https://nucetv.net",
    "https://rojname.com", "https://kurdpa.net", "https://english.kurd-online.com",
    "https://kurd-online.com", "https://knwe.org", "https://xebat.net",
    "https://sbeiy.com", "https://rupelanu.com", "https://xoybun.com",
    "https://jamawarnews.com", "https://diclehaber.com",
    "https://dengeazad.com", "https://hawarnet.com",
    "https://lotikxane.com", "https://nefel.nu", "https://netewe.com",
    "https://netkurd.com", "https://nukurd.com", "https://rewanbej.com",
    "https://serwext.com", "https://delo-duhoki.com",
    "https://amidakurd.com", "https://krd.riataza.com",
    "https://kurdi.welateme.net", "https://regaykurdistan.com",
    "https://rojevakurd.com", "https://speemedia.com",
    "https://kurdistancommentary.wordpress.com",
    "https://kurdishglobe.krd",
    # Technology / startups
    "https://iraqtech.io", "https://fiveonelabs.org", "https://jiasaz.com",
    "https://awrosoft.krd", "https://kurdai.org", "https://ovanya.io",
    "https://nasspay.com", "https://lezzoo.com", "https://zibox.co",
    "https://brsima.com", "https://suncode.co", "https://twekl.com",
    "https://h9cloud.com", "https://lucidsource.co", "https://mirotech.co",
    # E-commerce
    "https://boombeene.com", "https://ebazar.net", "https://arbela.store",
    "https://kurdshopping.com", "https://bazaryonline.com",
    "https://pelemall.com", "https://atoziraq.com", "https://amal.com",
    "https://iqcars.com",
    # Healthcare
    "https://eiherbil.com", "https://farukmedicalcity.com",
    "https://maryamanahospital.com", "https://parhospital.org",
    "https://cmcph.net", "https://hih-iq.com", "https://kurdistanclinic.com",
    # Legal
    "https://kurdishlawyers.com", "https://hawresurchi.com",
    "https://legalhand.co", "https://lawzana.com", "https://kdaloye.com",
    "https://kba.krd", "https://iraqlawyer.co.uk",
    # NGOs
    "https://kurdsngo.org", "https://americankurdsngo.org",
    "https://kohrw.org", "https://dckurd.org",
    # Education / language
    "https://blog.kurdish-archive.org", "https://kurdishcentral.org",
    "https://kurdishlessons.com", "https://knu.edu.iq",
    "https://ue.edu.krd", "https://dot.krd",
    # Culture / diaspora
    "https://kurdishpeace.org", "https://100berhemenkurdi.com",
    "https://parstimes.com/kurdish.html", "https://c4kurd.com",
    "https://koord.com", "https://kurdishproject.org",
    "https://thekurdishproject.org", "https://karwan.tv",
    "https://mixfm.live", "https://folkworks.org",
    # More radio / streaming
    "https://dengekurdistan.nu", "https://shiyarjemo.com",
    "https://allradio.net", "https://radioly.app",
    # More news/media
    "https://medyanews.net", "https://kurdistanpost.com",
    "https://yndk.com", "https://alitthad.com",
    "https://kurdistanmedia.com", "https://lemondediplo-kurdi.com",
    # Additional from search results
    "https://kurdistan-news.net", "https://soparo.com",
    "https://hathalyoum.net", "https://onlinenewspapers.com/kurdistan.shtml",
    "https://streema.com/radios/Kurdistan",
]


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


def crawl_page_links(url):
    """Crawl a single page for all outbound domains."""
    found = set()
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10,
                            allow_redirects=True, verify=True)
        if resp.status_code != 200:
            return found
        soup = BeautifulSoup(resp.content, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            full = urljoin(url, href)
            d = extract_domain(full)
            if d and not is_excluded(d):
                found.add(d)
    except:
        pass
    return found


def crawl_deep(url, max_subpages=5):
    """Crawl a site's homepage + up to max_subpages internal pages for outbound links."""
    all_found = set()
    base_domain = extract_domain(url)
    internal_urls = [url]

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10,
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
                    # Internal link - add to crawl queue
                    if len(internal_urls) < max_subpages + 1:
                        internal_urls.append(full)
                else:
                    all_found.add(d)
    except:
        pass

    # Crawl internal subpages for more outbound links
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


def query_cc_aggressive(query, index="CC-MAIN-2024-46"):
    """Query Common Crawl for domains matching a query pattern."""
    domains = set()
    try:
        api_url = f"https://index.commoncrawl.org/{index}-index"
        params = {"url": query, "output": "json", "limit": 500}
        resp = requests.get(api_url, params=params, timeout=30)
        if resp.status_code == 200:
            for line in resp.text.strip().split("\n"):
                if line:
                    try:
                        rec = json.loads(line)
                        d = extract_domain(rec.get("url", ""))
                        if d and not is_excluded(d):
                            domains.add(d)
                    except:
                        pass
    except:
        pass
    return domains


def generate_more_variations():
    """Generate additional domain variations not covered before."""
    domains = set()

    # Iraqi cities and towns with known Kurdish population
    iraqi_places = [
        "sulaymaniyah", "sulaymani", "sulimani", "suli",
        "hawleri", "hewleri", "arbella", "arbella",
        "dahuk", "dhouk", "zakhu", "barzan",
        "sharezur", "shahrazur", "bazyan", "taqtaq",
        "saruchawa", "biyara", "twella", "nawsud",
        "darband", "shorsh", "bestansur", "sharezor",
        "kafruk", "sangasar", "mawat", "qalat",
        "khanaqin", "mandali", "jalawla", "saadiya",
        "tuz", "tuzhurmatu", "laylan",
        # More Erbil subdistricts
        "ankawa", "kasnazan", "masif", "pirmam",
        "qushtapa", "shaqlawa2", "harir", "bnaslawa",
        # Sulaimani subdistricts
        "bakrajo", "tanjaro", "raparin2", "azadi2",
        "tasluja", "bazian", "qaradax", "warmawa",
    ]

    # Additional Kurdish organizations / media names
    org_names = [
        "asayish", "parastin", "zanyari", "zaniari",
        "destur", "komela", "hizb", "yeketi",
        "berhem", "chawder", "peyman", "rasan",
        "shar", "gund", "deh", "nawche",
        "nokan", "rawezh", "ronahì", "ster",
        "goran", "lalezar", "gul", "gulbehar",
        "saman", "bahoz", "barham", "dilan",
        "cira", "ciya", "berz", "bilind",
        "pishtiwan", "hevkar", "hawkar", "yari",
        "pewendì", "ragihandin", "agadarì", "zanayari",
        # Kurdish food / lifestyle
        "dolma", "kebab", "naan", "tandoor",
        "biryani", "tepsi", "quzi", "masgouf",
        # Kurdish concepts
        "peshraw", "helbest", "chirok", "roman",
        "gorani2", "stran", "dilan2", "govend",
    ]

    tlds = [".com", ".net", ".org", ".krd", ".info", ".io", ".tv"]

    for place in iraqi_places:
        p = place.replace("2", "")
        for tld in [".com", ".net", ".org", ".krd"]:
            domains.add(p + tld)
        for suffix in ["news", "online", "today", "tv", "media"]:
            domains.add(p + suffix + ".com")

    for name in org_names:
        n = name.replace("2", "").replace("ì", "i")
        for tld in tlds:
            domains.add(n + tld)

    # Subdomain exploration for key Kurdish domains
    subdomains = [
        "www", "mail", "portal", "blog", "news", "shop", "store",
        "app", "api", "cdn", "img", "media", "video", "live",
        "tv", "radio", "m", "mobile", "ar", "en", "ku", "ckb",
        "old", "new", "beta", "test", "dev", "admin", "panel",
        "jobs", "career", "hr", "apply", "register", "login",
        "support", "help", "faq", "contact", "about",
        "library", "archive", "docs", "wiki", "forum",
        "sport", "health", "edu", "research", "lab",
    ]

    key_domains = [
        "rudaw.net", "kurdistan24.net", "nrttv.com", "basnews.com",
        "shafaq.com", "xendan.org", "gov.krd", "parliament.krd",
        "kurdipedia.org", "hawlati.co", "awene.com",
        "peyamner.com", "pukmedia.com",
    ]

    for sub in subdomains:
        for base in key_domains:
            domains.add(f"{sub}.{base}")

    return domains


def check_alive(domain):
    url = f"https://{domain}"
    try:
        resp = requests.head(url, headers=HEADERS, timeout=6,
                             allow_redirects=True, verify=True)
        return (domain, resp.status_code, True)
    except requests.exceptions.SSLError:
        try:
            resp = requests.head(url, headers=HEADERS, timeout=6,
                                 allow_redirects=True, verify=False)
            return (domain, resp.status_code, True)
        except:
            return (domain, 0, False)
    except:
        return (domain, 0, False)


def main():
    base_dir = os.path.dirname(__file__)
    complete_csv = os.path.join(base_dir, "kurdish_websites_complete.csv")
    output_file = os.path.join(base_dir, "round2_discovered.txt")

    # Load known domains
    known_domains = set()
    live_urls = []
    if os.path.exists(complete_csv):
        with open(complete_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = row.get("domain", "").lower()
                known_domains.add(d)
                sc = row.get("status_code", "")
                if sc.startswith("2"):
                    live_urls.append(row["url"])

    print(f"[INFO] {len(known_domains)} known, {len(live_urls)} live.")

    # ─── Phase 1: Add URLs from web search results ───
    print(f"\n[PHASE 1] Adding {len(NEW_SEARCH_URLS)} URLs from web searches...")
    search_domains = set()
    for url in NEW_SEARCH_URLS:
        d = extract_domain(url)
        if d and d not in known_domains and not is_excluded(d):
            search_domains.add(d)
    print(f"  {len(search_domains)} new domains from searches.")

    # ─── Phase 2: Deep multi-page crawl of live sites ───
    print(f"\n[PHASE 2] Deep crawling {len(live_urls)} live sites (up to 5 subpages each)...")
    crawled = set()
    with ThreadPoolExecutor(max_workers=30) as pool:
        futures = {pool.submit(crawl_deep, url, 5): url for url in live_urls}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Deep crawling"):
            crawled.update(future.result())

    new_crawled = crawled - known_domains - search_domains
    # Filter for Kurdish relevance
    indicators = [
        "kurd", "hewler", "erbil", "sulaimani", "slemani", "duhok",
        "kirkuk", "halabja", "kobane", "afrin", "rojava", "newroz",
        "azadi", "peshmerga", "rudaw", "nrt", "basnews", "shafaq",
        "yezidi", "ezidi", "lalish", "zagros", "korek", "tishk",
        ".krd", ".iq",
    ]
    relevant = {d for d in new_crawled
                if any(i in d.lower() for i in indicators)
                or d.endswith(".krd") or d.endswith(".iq")}
    print(f"  Found {len(crawled)} total, {len(relevant)} Kurdish-relevant new.")

    # ─── Phase 3: Aggressive Common Crawl mining ───
    print(f"\n[PHASE 3] Mining Common Crawl indices aggressively...")
    cc_queries = [
        "*.krd/*", "*.edu.krd/*", "*.gov.krd/*",
        "*.kurdistan*/*", "*.kurd*/*",
        "*.hewler*/*", "*.erbil*/*",
        "*.sulaimani*/*", "*.duhok*/*",
        "*.rojava*/*", "*.kobane*/*",
    ]
    cc_indices = [
        "CC-MAIN-2024-46", "CC-MAIN-2024-33", "CC-MAIN-2024-22",
        "CC-MAIN-2024-10", "CC-MAIN-2023-50", "CC-MAIN-2023-40",
    ]

    cc_found = set()
    total_queries = len(cc_queries) * len(cc_indices)
    with tqdm(total=total_queries, desc="CC queries") as pbar:
        for query in cc_queries:
            for idx in cc_indices:
                results = query_cc_aggressive(query, idx)
                cc_found.update(results)
                pbar.update(1)
                time.sleep(0.3)

    new_cc = cc_found - known_domains - search_domains - relevant
    print(f"  Found {len(cc_found)} total, {len(new_cc)} new from Common Crawl.")

    # ─── Phase 4: More domain variations ───
    print(f"\n[PHASE 4] Generating additional variations...")
    variations = generate_more_variations()
    new_vars = variations - known_domains - search_domains - relevant - new_cc
    print(f"  {len(new_vars)} new variations to check.")

    # ─── Phase 5: Check everything ───
    all_new = search_domains | relevant | new_cc | new_vars
    print(f"\n[PHASE 5] Checking {len(all_new)} domains...")

    alive = []
    with ThreadPoolExecutor(max_workers=80) as pool:
        futures = {pool.submit(check_alive, d): d for d in all_new}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking"):
            domain, status, ok = future.result()
            if ok and 0 < status < 500:
                source = "search" if domain in search_domains else \
                         "crawl" if domain in relevant else \
                         "common_crawl" if domain in new_cc else "variation"
                alive.append((domain, status, source))

    print(f"\n  {len(alive)} new alive domains!")

    with open(output_file, "w", encoding="utf-8") as f:
        for domain, status, source in sorted(alive):
            f.write(f"https://{domain}\t{domain}\t{status}\t{source}\n")

    print(f"\n[SAVED] {len(alive)} to {output_file}")
    for src in ["search", "crawl", "common_crawl", "variation"]:
        cnt = sum(1 for _, _, s in alive if s == src)
        print(f"  {src}: {cnt}")
    print(f"  Previous: {len(known_domains)}")
    print(f"  New total: {len(known_domains) + len(alive)}")


if __name__ == "__main__":
    main()
