#!/usr/bin/env python3
"""
Kurdish Website Discovery Crawler
Discovers new Kurdish websites by:
1. Crawling outbound links from known live Kurdish sites
2. Querying Common Crawl index for Kurdish domains
3. Generating domain variations (subdomains, related TLDs)
"""

import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
MAX_WORKERS = 30

# Kurdish-related keywords for filtering discovered links
KURDISH_INDICATORS = [
    "kurd", "kurdi", "kurdish", "kurdistan", "sorani", "kurmanji",
    "ckb", "kmr", "hewler", "erbil", "sulaimani", "slemani", "duhok",
    "rojava", "rojhelat", "bashur", "bakur", "peshmerga", "newroz",
    "azadi", "welat", "nuce", "hawar", "gelawej", "rudaw",
    "halabja", "kirkuk", "kobane", "afrin", "qamishli", "diyarbakir",
    "amed", "zaxo", "shaqlawa", "rania", "koya", "chamchamal",
    "penjwen", "said sadiq", "zakho", "akre", "barzan",
    ".krd", "gov.krd", "edu.krd",
    # Kurdish script characters in domain
    "کورد", "هەواڵ", "نووچە",
]

# TLDs commonly used by Kurdish sites
RELEVANT_TLDS = {
    ".krd", ".iq", ".tr", ".ir", ".sy", ".com", ".net", ".org",
    ".tv", ".info", ".me", ".eu", ".de", ".se", ".nl", ".uk",
    ".news", ".media", ".press",
}

# Domains to exclude (too generic/large)
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
    "addthis.com", "sharethis.com", "disqus.com",
    "paypal.com", "stripe.com",
}


def extract_domain(url):
    """Extract the registrable domain from a URL."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        host = host.lower().strip(".")
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def is_kurdish_related(url, domain):
    """Check if a URL or domain appears to be Kurdish-related."""
    check = (url + " " + domain).lower()
    for indicator in KURDISH_INDICATORS:
        if indicator in check:
            return True
    # Check for .krd TLD
    if domain.endswith(".krd"):
        return True
    return False


def is_excluded(domain):
    """Check if domain should be excluded."""
    for excl in EXCLUDE_DOMAINS:
        if domain == excl or domain.endswith("." + excl):
            return True
    return False


def crawl_page_for_links(url):
    """Fetch a page and extract all outbound links."""
    discovered = set()
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                            allow_redirects=True, verify=True)
        if resp.status_code != 200:
            return discovered
        soup = BeautifulSoup(resp.content, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            full_url = urljoin(url, href)
            parsed = urlparse(full_url)
            if parsed.scheme not in ("http", "https"):
                continue
            domain = extract_domain(full_url)
            if not domain or is_excluded(domain):
                continue
            # Normalize to just the domain homepage
            clean_url = f"https://{domain}"
            discovered.add((clean_url, domain))
    except Exception:
        pass
    return discovered


def load_live_sites(csv_path):
    """Load live sites from existing CSV."""
    sites = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sc = row.get("status_code", "")
                if sc.startswith("2") or sc.startswith("3"):
                    sites.append(row["url"])
    except FileNotFoundError:
        pass
    return sites


def query_common_crawl_index(domain, index="CC-MAIN-2024-46"):
    """Query Common Crawl CDX index for pages from a domain."""
    urls = set()
    try:
        api_url = f"https://index.commoncrawl.org/{index}-index"
        params = {
            "url": f"*.{domain}",
            "output": "json",
            "limit": 100,
        }
        resp = requests.get(api_url, params=params, timeout=20)
        if resp.status_code == 200:
            for line in resp.text.strip().split("\n"):
                if line:
                    try:
                        rec = json.loads(line)
                        url = rec.get("url", "")
                        d = extract_domain(url)
                        if d and not is_excluded(d):
                            urls.add((f"https://{d}", d))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass
    return urls


def generate_domain_variations():
    """Generate likely Kurdish domain variations to probe."""
    prefixes = [
        "kurd", "kurdi", "kurdish", "kurdistan", "kordestan", "sorani",
        "kurmanji", "hawrami", "zazaki", "dimli", "gorani", "zaza",
        "rojava", "rojhelat", "bashur", "bakur", "hewler", "erbil",
        "sulaimani", "slemani", "duhok", "halabja", "kirkuk", "kobane",
        "afrin", "qamishli", "diyarbakir", "amed", "mahabad", "sanandaj",
        "peshmerga", "newroz", "azadi", "welat", "nuce", "hawar",
        "rudaw", "gelawej", "dengkurd", "dengikurd", "voakurd",
        "trtkurd", "kurdsport", "kurdnews", "kurdmedia", "kurdpress",
        "kurdfilm", "kurdcinema", "kurdblog", "kurdtv", "kurdradio",
        "kurdmax", "kurdsat", "zagrostv", "vintv", "newroztv",
        "yezidi", "ezidi", "lalish", "ankawa", "chaldean",
        "kurddev", "kurdtech", "kurdstore", "kurdbazar", "kurdshop",
        "kurdfood", "kurdhealth", "kurdlaw", "kurdart", "kurdmusic",
        "xebat", "peyam", "xendan", "sharpress", "pukpress",
        "kdppress", "gorranpress", "komalpress", "yekgirtu",
        "hfrk", "khrp", "kohrw", "kmewo",
        "zheen", "zhian", "jwan", "spee", "penase", "binav",
        "zardalyan", "nefel", "welatparez", "difraq", "sardam",
        "tishk", "speda", "payamtv", "galitv", "korek",
        "kurd1", "kurd2", "kurd3", "kurd24", "kurd4all",
        "kurdistan24", "kurdistantv", "kurdistanpress",
        "rojnews", "rojpress", "rojmedia", "rojtv",
        "hawarnews", "hawarmedia", "hawarpress",
        "nuceciwan", "dengciwan", "ciwan",
        "malperkurd", "malpera", "malper",
        "rizgari", "azadiya", "serhildan", "berxwedan",
        "jinnews", "jinlife", "jinpress", "jinmedia",
        "zanist", "zanko", "zankoy", "perwerde",
        "komela", "komkar", "tevger", "tevdem",
    ]
    tlds = [".com", ".net", ".org", ".info", ".tv", ".news",
            ".media", ".krd", ".press", ".io", ".me"]
    variations = set()
    for prefix in prefixes:
        for tld in tlds:
            domain = prefix + tld
            variations.add((f"https://{domain}", domain))
    return variations


def check_domain_alive(url_domain_tuple):
    """Quick check if a domain is alive."""
    url, domain = url_domain_tuple
    try:
        resp = requests.head(url, headers=HEADERS, timeout=8,
                             allow_redirects=True, verify=True)
        return (url, domain, resp.status_code, True)
    except requests.exceptions.SSLError:
        try:
            resp = requests.head(url, headers=HEADERS, timeout=8,
                                 allow_redirects=True, verify=False)
            return (url, domain, resp.status_code, True)
        except Exception:
            return (url, domain, 0, False)
    except Exception:
        return (url, domain, 0, False)


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "kurdish_websites.csv")
    output_path = os.path.join(os.path.dirname(__file__), "discovered_urls.txt")

    # Load already known domains
    known_domains = set()
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                known_domains.add(row.get("domain", "").lower())
    except FileNotFoundError:
        pass

    print(f"[INFO] {len(known_domains)} already-known domains from CSV.")

    # Phase 1: Crawl live Kurdish sites for outbound links
    print("\n[PHASE 1] Crawling live Kurdish sites for outbound links...")
    live_sites = load_live_sites(csv_path)
    print(f"  Found {len(live_sites)} live sites to crawl.")

    all_discovered = set()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(crawl_page_for_links, url): url for url in live_sites}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Crawling sites"):
            links = future.result()
            all_discovered.update(links)

    # Filter: only keep Kurdish-related or from known Kurdish TLDs
    crawled_kurdish = set()
    for url, domain in all_discovered:
        if domain not in known_domains and is_kurdish_related(url, domain):
            crawled_kurdish.add((url, domain))

    print(f"  Discovered {len(all_discovered)} unique domains total.")
    print(f"  {len(crawled_kurdish)} appear Kurdish-related (new).")

    # Phase 2: Generate domain variations
    print("\n[PHASE 2] Generating domain name variations...")
    variations = generate_domain_variations()
    new_variations = {(u, d) for u, d in variations
                      if d not in known_domains and d not in {d2 for _, d2 in crawled_kurdish}}
    print(f"  Generated {len(new_variations)} new domain variations to probe.")

    # Phase 3: Query Common Crawl for Kurdish domains
    print("\n[PHASE 3] Querying Common Crawl index for Kurdish domains...")
    cc_seeds = [
        "kurd", "kurdish", "kurdistan", "sorani", "kurmanji",
        "rojava", "hewler", "erbil", "sulaimani", "duhok",
        "peshmerga", "newroz", "kobane", "afrin",
    ]
    cc_discovered = set()
    for seed in tqdm(cc_seeds, desc="Common Crawl queries"):
        results = query_common_crawl_index(seed + ".com")
        cc_discovered.update(results)
        results = query_common_crawl_index(seed + ".net")
        cc_discovered.update(results)
        results = query_common_crawl_index(seed + ".org")
        cc_discovered.update(results)
        time.sleep(0.5)  # Rate limit

    new_cc = {(u, d) for u, d in cc_discovered if d not in known_domains}
    print(f"  Found {len(new_cc)} new domains from Common Crawl.")

    # Combine all newly discovered domains
    all_new = crawled_kurdish | new_variations | new_cc
    print(f"\n[TOTAL] {len(all_new)} new candidate domains to check.")

    # Phase 4: Check which new domains are alive
    print("\n[PHASE 4] Checking which new domains are alive...")
    alive_domains = []

    with ThreadPoolExecutor(max_workers=50) as pool:
        futures = {pool.submit(check_domain_alive, item): item for item in all_new}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Checking domains"):
            url, domain, status, is_alive = future.result()
            if is_alive and status > 0 and status < 500:
                alive_domains.append((url, domain, status))

    print(f"\n  {len(alive_domains)} new domains are ALIVE!")

    # Save discovered URLs
    with open(output_path, "w", encoding="utf-8") as f:
        for url, domain, status in sorted(alive_domains, key=lambda x: x[1]):
            f.write(f"{url}\t{domain}\t{status}\n")

    print(f"\n[SAVED] {len(alive_domains)} discovered URLs to {output_path}")

    # Summary
    print(f"\n--- Discovery Summary ---")
    print(f"  From crawling live sites: {len(crawled_kurdish)}")
    print(f"  From domain variations:   {len(new_variations)}")
    print(f"  From Common Crawl:        {len(new_cc)}")
    print(f"  Total candidates:         {len(all_new)}")
    print(f"  Alive & responding:       {len(alive_domains)}")


if __name__ == "__main__":
    main()
