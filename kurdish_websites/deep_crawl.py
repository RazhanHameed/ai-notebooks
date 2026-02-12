#!/usr/bin/env python3
"""
Deep Crawl: Multi-phase Kurdish website discovery.
Phase 1: Crawl all reachable sites (including newly discovered) for outbound links
Phase 2: Query Common Crawl aggressively for Kurdish domains
Phase 3: Generate even more domain variations
Phase 4: Verify and merge
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

TIMEOUT = 10

EXCLUDE_DOMAINS = {
    "google.com", "youtube.com", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "tiktok.com", "linkedin.com", "reddit.com",
    "whatsapp.com", "telegram.org", "t.me", "apple.com", "play.google.com",
    "apps.apple.com", "amazon.com", "wikipedia.org", "wikimedia.org",
    "wordpress.com", "blogspot.com", "blogger.com", "medium.com",
    "soundcloud.com", "spotify.com", "pinterest.com", "flickr.com",
    "vimeo.com", "dailymotion.com", "archive.org", "web.archive.org",
    "bit.ly", "goo.gl", "t.co", "ow.ly", "tinyurl.com",
    "cloudflare.com", "googleapis.com", "gstatic.com", "cdnjs.com",
    "jquery.com", "bootstrapcdn.com", "w3.org", "schema.org",
    "addthis.com", "sharethis.com", "disqus.com", "wp.com",
    "paypal.com", "stripe.com", "gravatar.com", "wp.me",
    "googlesyndication.com", "doubleclick.net", "googletagmanager.com",
    "google-analytics.com", "fontawesome.com", "fonts.googleapis.com",
    "maxcdn.bootstrapcdn.com", "ajax.googleapis.com",
    "creativecommons.org", "feedburner.com", "statcounter.com",
    "microsoft.com", "office.com", "live.com", "outlook.com",
    "yahoo.com", "bing.com", "baidu.com",
    "github.com", "gitlab.com", "bitbucket.org",
    "npmjs.com", "pypi.org", "rubygems.org",
}

KURDISH_INDICATORS = [
    "kurd", "kurdi", "kurdish", "kurdistan", "sorani", "kurmanji",
    "ckb", "hewler", "erbil", "sulaimani", "slemani", "duhok",
    "rojava", "rojhelat", "bashur", "bakur", "peshmerga", "newroz",
    "azadi", "welat", "nuce", "hawar", "gelawej", "rudaw",
    "halabja", "kirkuk", "kobane", "afrin", "qamishli", "diyarbakir",
    "amed", "zaxo", "shaqlawa", "rania", "koya", "chamchamal",
    "penjwen", "zakho", "akre", "barzan", "hawrami", "zazaki",
    ".krd", "gov.krd", "edu.krd",
    "yezidi", "ezidi", "lalish", "ankawa",
    "korek", "zagros", "gali", "tishk", "speda",
    "dengkurd", "dengikurd", "payam", "xendan",
    "sharpress", "pukmedia", "kdp", "puk", "gorran",
    "hmu", "uos", "lfu", "ukh", "auis", "charmo",
    "peshawa", "sakar", "bardarash", "mergasur", "sharezoor",
    "raperin", "garmian", "kalar", "kifri", "chamchamal",
    "soran", "choman", "sidakan", "khalifan", "shaqlaw",
    "rawanduz", "diana", "koysinjaq", "dukan", "qaladze",
    "said", "piramagrun", "darbandikhan",
]


def extract_domain(url):
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        host = host.lower().strip(".")
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def is_excluded(domain):
    for excl in EXCLUDE_DOMAINS:
        if domain == excl or domain.endswith("." + excl):
            return True
    return False


def is_kurdish_related(url, domain):
    check = (url + " " + domain).lower()
    for indicator in KURDISH_INDICATORS:
        if indicator in check:
            return True
    if domain.endswith(".krd"):
        return True
    # Check for .iq domains (Iraqi, often Kurdish)
    if domain.endswith(".iq") or domain.endswith(".edu.iq"):
        return True
    return False


def crawl_page_deep(url):
    """Fetch a page and extract all outbound domains with Kurdish filtering."""
    discovered = set()
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                            allow_redirects=True, verify=True)
        if resp.status_code != 200:
            return discovered

        soup = BeautifulSoup(resp.content, "html.parser")

        # Extract links from <a> tags
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            full_url = urljoin(url, href)
            parsed = urlparse(full_url)
            if parsed.scheme not in ("http", "https"):
                continue
            domain = extract_domain(full_url)
            if domain and not is_excluded(domain):
                discovered.add(domain)

        # Also check for links in iframes, scripts src
        for iframe in soup.find_all("iframe", src=True):
            d = extract_domain(iframe["src"])
            if d and not is_excluded(d):
                discovered.add(d)

    except Exception:
        pass
    return discovered


def query_cc(query_domain, index="CC-MAIN-2024-46"):
    """Query Common Crawl CDX for a domain."""
    urls = set()
    try:
        api_url = f"https://index.commoncrawl.org/{index}-index"
        params = {
            "url": f"*.{query_domain}/*",
            "output": "json",
            "limit": 200,
        }
        resp = requests.get(api_url, params=params, timeout=25)
        if resp.status_code == 200:
            for line in resp.text.strip().split("\n"):
                if line:
                    try:
                        rec = json.loads(line)
                        d = extract_domain(rec.get("url", ""))
                        if d and not is_excluded(d):
                            urls.add(d)
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass
    return urls


def generate_extended_variations():
    """Generate a massive set of Kurdish domain variations."""
    # City and region names in Kurdistan
    cities = [
        "hewler", "erbil", "hawler", "sulaimani", "slemani", "duhok",
        "halabja", "kirkuk", "kerkuk", "kobane", "kobani", "afrin", "efrin",
        "qamishli", "qamishlo", "hasaka", "hasakeh", "derik", "amuda",
        "tirbespi", "gire spi", "manbij", "raqqa", "tabqa",
        "diyarbakir", "amed", "batman", "siirt", "sirnak", "hakkari",
        "van", "bitlis", "mus", "bingol", "tunceli", "dersim",
        "elazig", "malatya", "urfa", "mardin", "nusaybin", "cizre",
        "mahabad", "sanandaj", "sine", "ilam", "kermanshah",
        "paveh", "javanrud", "kamyaran", "bijar", "saghez", "baneh",
        "sardasht", "oshnavieh", "piranshahr", "bukan", "miandoab",
        "nagadeh", "shahin dezh", "divandareh",
        "zaxo", "zakho", "shaqlawa", "rawanduz", "soran", "mergasor",
        "choman", "diana", "akre", "bardarash", "shekhan",
        "kalar", "kifri", "chamchamal", "darbandikhan", "penjwen",
        "said sadiq", "qaradagh", "ranya", "dukan", "qaladze",
        "koysinjaq", "makhmur", "pirmam", "gwer", "bashiqa",
    ]

    # Kurdish media/content words
    content_words = [
        "kurd", "kurdi", "kurdish", "kurdistan", "newroz", "azadi",
        "rojava", "rojhelat", "bashur", "bakur", "welat", "nuce",
        "hawar", "deng", "roj", "jin", "jiyan", "berxwedan",
        "peshmerga", "xebat", "rizgari", "serhildan",
        "gelawej", "sardam", "zheen", "nefel", "penase",
        "zanist", "perwerde", "tendrusti", "aburi",
    ]

    # Common Kurdish website patterns
    suffixes = [
        "news", "press", "media", "tv", "radio", "fm",
        "online", "net", "web", "site", "info",
        "post", "times", "daily", "today", "now",
        "blog", "mag", "magazine",
    ]

    tlds = [".com", ".net", ".org", ".info", ".tv", ".news",
            ".media", ".krd", ".press", ".io", ".me", ".xyz"]

    variations = set()

    # City + suffix combos
    for city in cities:
        city_clean = city.replace(" ", "")
        for tld in [".com", ".net", ".org", ".info", ".krd"]:
            variations.add(city_clean + tld)
        for suffix in ["news", "press", "media", "tv", "online", "today", "post"]:
            for tld in [".com", ".net", ".org"]:
                variations.add(city_clean + suffix + tld)

    # Content word + suffix combos
    for word in content_words:
        for tld in tlds:
            variations.add(word + tld)
        for suffix in suffixes:
            for tld in [".com", ".net", ".org"]:
                variations.add(word + suffix + tld)

    # Kurdish + city combos
    for city in cities[:20]:
        city_clean = city.replace(" ", "")
        for prefix in ["kurd", "kurdish", "kurdistan"]:
            for tld in [".com", ".net", ".org"]:
                variations.add(prefix + city_clean + tld)
                variations.add(city_clean + prefix + tld)

    # Additional specific patterns
    specific = [
        # More domain patterns found in Kurdish web ecosystem
        "kurdistannet.org", "kurdsite.com", "kurdlink.com",
        "kurdinfo.com", "kurdnews.net", "kurddaily.com",
        "kurdweekly.com", "kurdmonthly.com", "kurdvoice.com",
        "voicekurd.com", "voiceofkurdistan.com",
        "kurdishvoice.com", "kurdishvoice.net",
        "radiokurdistan.com", "tvkurdistan.com",
        "kurdishradio.com", "kurdishonline.com",
        "kurdonline.com", "kurdonline.net",
        "kurdweb.com", "kurdweb.net",
        "kurdistan.com", "kurdistan.net", "kurdistan.org",
        "kurdistan.info", "kurdistan.tv", "kurdistan.press",
        "kurdistan.news", "kurdistan.media",
        "kurd.com", "kurd.net", "kurd.org", "kurd.info", "kurd.tv",
        "kurds.com", "kurds.net", "kurds.org",
        "rojava.com", "rojava.net", "rojava.org", "rojava.info",
        "rojava.news", "rojava.tv", "rojava.media",
        "kobanê.com", "kobane.net", "kobane.org",
        "afrin.com", "afrin.net", "afrin.org", "afrin.info",
        "amed.com", "amed.net", "amed.org", "amed.info",
        "hewler.com", "hewler.net", "hewler.org",
        "hawler.com", "hawler.net", "hawler.org",
        # Education institutions
        "spu.edu.iq", "su.edu.krd",
        "uoh.edu.iq", "polytechnic.edu.iq",
        "hmu.edu.krd", "su.edu.krd",
        # More Kurdish media
        "spartanews.com", "vendpress.com", "kawapress.com",
        "dengnews.com", "dengpress.com",
        "hawlernews.com", "sloganpress.com",
        "komalnews.com", "gorannews.com",
        "yekitunews.com", "islamicunionnews.com",
        "newgenerationnews.com",
        # Kurdish food/lifestyle
        "kurdfood.com", "kurdcuisine.com",
        "kurdishfood.com", "kurdishrecipes.com",
        "kurdishcooking.com",
        # Kurdish diaspora
        "kurdistanlondon.com", "kurdistanberlin.com",
        "kurdistanparis.com", "kurdistanstockholm.com",
        "kurdsweden.com", "kurdgermany.com",
        "kurdfrance.com", "kurduk.com",
        "kurdusa.com", "kurdcanada.com",
        "kurdaustralia.com", "kurdnorway.com",
        "kurddenmark.com", "kurdfinland.com",
        "kurdbrazil.com", "kurdiran.com",
        # Kurdish sports
        "erbilfc.com", "slemanifc.com", "duhokfc.com",
        "erbilsport.com", "slemanisport.com",
        "kurdistanfootball.com", "kurdfootball.com",
        # More political
        "kdpinfo.com", "pukinfo.com", "gorraninfo.com",
        "newgeneration.krd", "islamicunion.krd",
        # Healthcare
        "healthkurdistan.com", "kurdistanhealth.com",
        "kurdistanhospital.com",
        # Technology
        "techkurd.com", "techkurdistan.com",
        "startuphewler.com", "startuperbil.com",
        "kurditech.com", "techsorani.com",
        # Travel
        "travelkurdistan.com", "visiterbil.com",
        "visitsulaimani.com", "visitduhok.com",
        "kurdistantravel.com", "kurdtourism.com",
    ]
    variations.update(specific)

    return variations


def check_domain_alive(domain):
    """Quick check if a domain is alive."""
    url = f"https://{domain}"
    try:
        resp = requests.head(url, headers=HEADERS, timeout=8,
                             allow_redirects=True, verify=True)
        return (domain, resp.status_code, True)
    except requests.exceptions.SSLError:
        try:
            resp = requests.head(url, headers=HEADERS, timeout=8,
                                 allow_redirects=True, verify=False)
            return (domain, resp.status_code, True)
        except Exception:
            return (domain, 0, False)
    except Exception:
        return (domain, 0, False)


def main():
    base_dir = os.path.dirname(__file__)
    full_csv = os.path.join(base_dir, "kurdish_websites_full.csv")
    output_file = os.path.join(base_dir, "deep_discovered.txt")

    # Load all known domains
    known_domains = set()
    live_urls = []
    if os.path.exists(full_csv):
        with open(full_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = row.get("domain", "").lower()
                known_domains.add(d)
                sc = row.get("status_code", "")
                if sc.startswith("2"):
                    live_urls.append(row["url"])

    print(f"[INFO] {len(known_domains)} known domains, {len(live_urls)} live sites.")

    # ─── Phase 1: Deep crawl all reachable sites ───
    print(f"\n[PHASE 1] Deep crawling {len(live_urls)} live sites for outbound links...")
    all_found_domains = set()

    with ThreadPoolExecutor(max_workers=30) as pool:
        futures = {pool.submit(crawl_page_deep, url): url for url in live_urls}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Deep crawling"):
            domains = future.result()
            all_found_domains.update(domains)

    new_from_crawl = all_found_domains - known_domains
    # Filter for Kurdish-related
    kurdish_from_crawl = {d for d in new_from_crawl
                          if is_kurdish_related(f"https://{d}", d)}

    print(f"  Found {len(all_found_domains)} total domains from crawling.")
    print(f"  {len(new_from_crawl)} are new (not in existing list).")
    print(f"  {len(kurdish_from_crawl)} appear Kurdish-related.")

    # ─── Phase 2: Common Crawl queries ───
    print(f"\n[PHASE 2] Querying Common Crawl for Kurdish domains...")
    cc_query_domains = [
        # Search for various .krd domains
        "gov.krd", "edu.krd", "krd",
        # Kurdish news domains
        "rudaw.net", "kurdistan24.net", "nrttv.com",
        "basnews.com", "shafaq.com",
        # Kurdish organizations
        "kurdipedia.org", "kurdishproject.org",
    ]

    cc_indices = ["CC-MAIN-2024-46", "CC-MAIN-2024-33", "CC-MAIN-2024-22"]

    cc_found = set()
    for domain in tqdm(cc_query_domains, desc="CC queries"):
        for idx in cc_indices:
            results = query_cc(domain, idx)
            cc_found.update(results)
            time.sleep(0.3)

    new_from_cc = cc_found - known_domains - kurdish_from_crawl
    print(f"  Found {len(cc_found)} domains from Common Crawl.")
    print(f"  {len(new_from_cc)} are new.")

    # ─── Phase 3: Domain variations ───
    print(f"\n[PHASE 3] Generating extended domain variations...")
    variations = generate_extended_variations()
    new_variations = variations - known_domains - kurdish_from_crawl - new_from_cc
    print(f"  Generated {len(new_variations)} new domain variations to probe.")

    # ─── Phase 4: Check all new domains ───
    all_new = kurdish_from_crawl | new_from_cc | new_variations
    print(f"\n[PHASE 4] Checking {len(all_new)} new domains for liveness...")

    alive_domains = []
    with ThreadPoolExecutor(max_workers=60) as pool:
        futures = {pool.submit(check_domain_alive, d): d for d in all_new}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Checking domains"):
            domain, status, is_alive = future.result()
            if is_alive and 0 < status < 500:
                source = "crawl" if domain in kurdish_from_crawl else \
                         "common_crawl" if domain in new_from_cc else "variation"
                alive_domains.append((domain, status, source))

    print(f"\n  {len(alive_domains)} new domains are ALIVE!")

    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        for domain, status, source in sorted(alive_domains, key=lambda x: x[0]):
            f.write(f"https://{domain}\t{domain}\t{status}\t{source}\n")

    print(f"\n[SAVED] {len(alive_domains)} new URLs to {output_file}")
    print(f"\n--- Deep Discovery Summary ---")
    print(f"  From deep crawl:       {sum(1 for _, _, s in alive_domains if s == 'crawl')}")
    print(f"  From Common Crawl:     {sum(1 for _, _, s in alive_domains if s == 'common_crawl')}")
    print(f"  From variations:       {sum(1 for _, _, s in alive_domains if s == 'variation')}")
    print(f"  Total new alive:       {len(alive_domains)}")
    print(f"  Previous known:        {len(known_domains)}")
    print(f"  New total potential:    {len(known_domains) + len(alive_domains)}")


if __name__ == "__main__":
    main()
