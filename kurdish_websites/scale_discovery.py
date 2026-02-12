#!/usr/bin/env python3
"""
Scale discovery: Generate and check thousands more Kurdish-related domains.
Strategy:
1. Exhaustive city/town name + TLD combinations
2. Kurdish word combinations
3. Subdomain enumeration on known .krd, .edu.krd, .gov.krd
4. Deep crawl of reachable sites for ALL outbound links (relaxed Kurdish filter)
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
    "amazonaws.com", "azurewebsites.net", "herokuapp.com",
}


def is_excluded(domain):
    for excl in EXCLUDE_DOMAINS:
        if domain == excl or domain.endswith("." + excl):
            return True
    return False


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


def generate_massive_variations():
    """Generate thousands of Kurdish-related domain variations."""
    domains = set()

    # ─── All Kurdistan cities and towns ───
    places = [
        # Iraqi Kurdistan
        "hewler", "erbil", "hawler", "arbil", "irbil",
        "sulaimani", "slemani", "sulaymaniyah", "sulaymani",
        "duhok", "dahuk", "dohuk", "zaxo", "zakho",
        "halabja", "halabjah", "shaqlawa", "shaqlaw",
        "rawanduz", "rawandiz", "soran", "choman",
        "diana", "akre", "aqra", "bardarash", "shekhan",
        "kalar", "kalar", "kifri", "chamchamal",
        "darbandikhan", "penjwen", "panjwin",
        "saidsadiq", "qaradagh", "ranya", "raparin",
        "dukan", "dokan", "qaladze", "qaladiza",
        "koysinjaq", "koisanjaq", "makhmur", "makhmour",
        "pirmam", "gwer", "gweir", "bashiqa",
        "mergasor", "mergasur", "sidakan", "khalifan",
        "sharezoor", "sharazoor", "piramagrun",
        "taqtaq", "harir", "shaqlawa", "barzan",
        "amadiya", "amedi", "semel", "simele",
        # Kirkuk
        "kirkuk", "kerkuk", "tuzkhormato", "daquq",
        "hawija", "altuncopru", "dibis",
        # Rojava / NE Syria
        "kobane", "kobani", "afrin", "efrin",
        "qamishli", "qamishlo", "qamishle",
        "hasaka", "hasakeh", "derik",
        "amuda", "amude", "tirbespi", "tirbespiye",
        "girespi", "telabyed", "manbij",
        "raqqa", "tabqa", "tabqah",
        "serekaniye", "rasalayn",
        "derbasi", "darbasiyah",
        "ain issa", "ainissa",
        # Turkish Kurdistan (Bakur)
        "diyarbakir", "amed", "batman", "siirt",
        "sirnak", "hakkari", "van", "bitlis",
        "mus", "bingol", "tunceli", "dersim",
        "elazig", "malatya", "urfa", "sanliurfa",
        "mardin", "nusaybin", "cizre", "cizir",
        "silopi", "idil", "midyat", "savur",
        "kiziltepe", "derik", "mazidagi",
        "lice", "silvan", "bismil", "kulp",
        "hazro", "dicle", "hani", "ergani",
        "sur", "yenisehir", "kayapinar",
        "viransehir", "bozova", "siverek", "hilvan",
        "pervari", "eruh", "beytussebap",
        "semdinli", "yuksekova", "colemerik",
        "catak", "baskale", "ozalp", "gurpinar",
        "tatvan", "ahlat", "hizan", "mutki",
        "bulanik", "malazgirt", "varto", "sasun",
        "genç", "karliova", "solhan", "adakli",
        "pulumur", "hozat", "mazgirt", "ovacik",
        "pertek", "cemisgezek", "nazimiye",
        # Iranian Kurdistan (Rojhelat)
        "mahabad", "mehabad", "sanandaj", "sine",
        "ilam", "kermanshah", "kirmashan",
        "paveh", "javanrud", "javanroud",
        "kamyaran", "bijar", "saghez", "saqqez",
        "baneh", "sardasht", "oshnavieh", "oshnaviyeh",
        "piranshahr", "bukan", "miandoab",
        "nagadeh", "naqadeh", "shahindezh",
        "divandareh", "marivan", "sarvabad",
        "dalahu", "kangavar", "sonqor", "sahneh",
        "harsin", "qasr-e-shirin", "gilangharb",
        "eslam abad", "ravansar",
    ]

    # Kurdish words for domain generation
    kurdish_words = [
        "kurd", "kurdi", "kurdish", "kurdistan", "kurds",
        "newroz", "azadi", "azadiya", "welat", "welatparez",
        "nuce", "hawar", "deng", "dengi", "roj", "rojava",
        "gelawej", "sardam", "zheen", "nefel", "penase",
        "zanist", "perwerde", "tendrusti", "aburi",
        "serhildan", "berxwedan", "rizgari", "xebat",
        "jin", "jiyan", "hevallo", "hewal", "peshmerga",
        "yezidi", "ezidi", "lalish",
        "korek", "zagros", "gali", "tishk", "speda",
        "payam", "xendan", "sharpress", "pukmedia",
        "rudaw", "basnews", "kirkuknow", "shafaq",
        "gulan", "jiyan", "binav", "zanko",
        "sorani", "kurmanji", "hawrami", "zazaki",
        "komala", "komal", "gorran", "yekgirtu",
        "chinarok", "rengin", "reng", "gulistan",
        "ruwange", "hawpshti", "warvin",
        "chinar", "nardin", "baran", "shamal",
        "bahar", "payiz", "zivistan", "havin",
        "chia", "gund", "shar", "bazar",
        "ashti", "heshti", "dostani",
    ]

    suffixes = [
        "", "news", "press", "media", "tv", "radio",
        "online", "net", "web", "info", "post",
        "times", "daily", "today", "now",
        "blog", "mag", "hub", "center",
        "group", "tech", "dev", "store",
        "shop", "market", "food", "travel",
        "health", "sport", "music", "film",
    ]

    tlds = [".com", ".net", ".org", ".info", ".tv",
            ".news", ".media", ".krd", ".press", ".io",
            ".me", ".xyz", ".online", ".site"]

    # Generate place + TLD
    for place in places:
        place_clean = place.replace(" ", "").replace("-", "")
        for tld in [".com", ".net", ".org", ".info", ".krd"]:
            domains.add(place_clean + tld)
        for suffix in ["news", "press", "media", "tv", "online", "today", "post"]:
            for tld in [".com", ".net"]:
                domains.add(place_clean + suffix + tld)

    # Generate word + TLD and word+suffix+TLD
    for word in kurdish_words:
        for tld in tlds:
            domains.add(word + tld)
        for suffix in suffixes:
            if suffix:
                for tld in [".com", ".net", ".org"]:
                    domains.add(word + suffix + tld)

    # Known subdomains for .krd, .edu.krd, .gov.krd
    gov_subdomains = [
        "www", "mail", "portal", "e-gov", "hr", "finance",
        "planning", "agriculture", "electricity", "water",
        "transport", "trade", "industry", "culture", "youth",
        "sport", "tourism", "peshmerga", "asayish",
    ]
    krd_bases = ["gov.krd", "edu.krd"]
    for sub in gov_subdomains:
        for base in krd_bases:
            domains.add(f"{sub}.{base}")

    edu_subdomains = [
        "www", "portal", "lms", "library", "admission",
        "hr", "research", "student", "faculty", "admin",
        "e-learning", "moodle", "scholarship",
    ]
    edu_bases = [
        "uos.edu.krd", "su.edu.krd", "hmu.edu.krd",
        "lfu.edu.krd", "ukh.edu.krd", "charmo.edu.krd",
        "raperin.edu.krd", "garmian.edu.krd",
        "halabjauni.edu.krd", "knowledge.edu.krd",
    ]
    for sub in edu_subdomains:
        for base in edu_bases:
            domains.add(f"{sub}.{base}")

    return domains


def crawl_all_links(url):
    """Crawl a page for ALL outbound domains (not just Kurdish-related)."""
    discovered = set()
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10,
                            allow_redirects=True, verify=True)
        if resp.status_code != 200:
            return discovered
        soup = BeautifulSoup(resp.content, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            full_url = urljoin(url, href)
            domain = extract_domain(full_url)
            if domain and not is_excluded(domain):
                discovered.add(domain)
    except:
        pass
    return discovered


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
    final_csv = os.path.join(base_dir, "kurdish_websites_final.csv")
    output_file = os.path.join(base_dir, "scale_discovered.txt")

    # Load known domains
    known_domains = set()
    live_urls = []
    if os.path.exists(final_csv):
        with open(final_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = row.get("domain", "").lower()
                known_domains.add(d)
                sc = row.get("status_code", "")
                if sc.startswith("2"):
                    live_urls.append(row["url"])

    print(f"[INFO] {len(known_domains)} known domains, {len(live_urls)} live sites.")

    # ─── Phase 1: Deep crawl all live sites (relaxed) ───
    print(f"\n[PHASE 1] Deep crawling {len(live_urls)} live sites (relaxed filter)...")
    crawled_domains = set()

    with ThreadPoolExecutor(max_workers=40) as pool:
        futures = {pool.submit(crawl_all_links, url): url for url in live_urls}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Crawling"):
            domains = future.result()
            crawled_domains.update(domains)

    new_from_crawl = crawled_domains - known_domains
    print(f"  Found {len(crawled_domains)} total, {len(new_from_crawl)} new domains.")

    # Keep domains that have Kurdish-related indicators OR are from .iq/.krd
    def is_relevant(d):
        if d.endswith(".krd") or d.endswith(".iq"):
            return True
        check = d.lower()
        indicators = [
            "kurd", "hewler", "erbil", "sulaimani", "slemani",
            "duhok", "kirkuk", "halabja", "kobane", "afrin",
            "rojava", "newroz", "azadi", "peshmerga",
            "rudaw", "nrt", "basnews", "shafaq",
            "yezidi", "ezidi", "lalish",
            "zagros", "korek", "tishk", "speda",
        ]
        return any(ind in check for ind in indicators)

    relevant_crawled = {d for d in new_from_crawl if is_relevant(d)}
    print(f"  {len(relevant_crawled)} are Kurdish-related.")

    # ─── Phase 2: Massive domain variations ───
    print(f"\n[PHASE 2] Generating massive domain variations...")
    variations = generate_massive_variations()
    new_variations = variations - known_domains - relevant_crawled
    print(f"  Generated {len(new_variations)} new variations to check.")

    # ─── Phase 3: Check all ───
    all_to_check = relevant_crawled | new_variations
    print(f"\n[PHASE 3] Checking {len(all_to_check)} domains for liveness...")

    alive = []
    with ThreadPoolExecutor(max_workers=80) as pool:
        futures = {pool.submit(check_alive, d): d for d in all_to_check}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Checking"):
            domain, status, ok = future.result()
            if ok and 0 < status < 500:
                source = "crawl" if domain in relevant_crawled else "variation"
                alive.append((domain, status, source))

    print(f"\n  {len(alive)} new alive domains!")

    with open(output_file, "w", encoding="utf-8") as f:
        for domain, status, source in sorted(alive, key=lambda x: x[0]):
            f.write(f"https://{domain}\t{domain}\t{status}\t{source}\n")

    print(f"\n[SAVED] {len(alive)} to {output_file}")
    crawl_cnt = sum(1 for _, _, s in alive if s == "crawl")
    var_cnt = sum(1 for _, _, s in alive if s == "variation")
    print(f"  From crawl: {crawl_cnt}")
    print(f"  From variations: {var_cnt}")
    print(f"  Previous: {len(known_domains)}")
    print(f"  New total: {len(known_domains) + len(alive)}")


if __name__ == "__main__":
    main()
