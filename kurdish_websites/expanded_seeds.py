"""
Comprehensive list of Kurdish-related website URLs organized by category.

This module provides a large, deduplicated collection of Kurdish websites
spanning news, media, government, education, culture, diaspora, and more.
"""


def get_all_expanded_urls():
    """
    Return a list of dicts with keys: url, source, category, notes.

    Each entry represents a known Kurdish-related website, organized by
    category and tagged with its source type.
    """

    urls = []

    # =========================================================================
    # NEWS & MEDIA
    # =========================================================================
    news_media = [
        ("https://rudaw.net", "known_media", "news_media", "Major Kurdish news outlet"),
        ("https://kurdistan24.net", "known_media", "news_media", "Kurdistan 24 news network"),
        ("https://nrttv.com", "known_media", "news_media", "NRT TV news"),
        ("https://kirkuknow.com", "known_media", "news_media", "Kirkuk-focused news"),
        ("https://basnews.com", "known_media", "news_media", "BasNews Kurdish media"),
        ("https://shafaq.com", "known_media", "news_media", "Shafaq News"),
        ("https://xendan.org", "known_media", "news_media", "Xendan news agency"),
        ("https://awene.com", "known_media", "news_media", "Awene newspaper"),
        ("https://hawlati.co", "known_media", "news_media", "Hawlati newspaper"),
        ("https://peyamner.com", "known_media", "news_media", "Peyamner news agency"),
        ("https://pukmedia.com", "known_media", "news_media", "PUK-affiliated media"),
        ("https://gulan-media.com", "known_media", "news_media", "Gulan Media"),
        ("https://kurdpress.com", "known_media", "news_media", "KurdPress news"),
        ("https://dengiamerica.com", "known_media", "news_media", "Voice of America Kurdish"),
        ("https://ronahi.tv", "known_media", "news_media", "Ronahi TV"),
        ("https://hawar.news", "known_media", "news_media", "Hawar News Agency"),
        ("https://hawarnews.com", "known_media", "news_media", "ANHA news agency"),
        ("https://rojavanews.com", "known_media", "news_media", "Rojava News"),
        ("https://anfenglish.com", "known_media", "news_media", "ANF English edition"),
        ("https://anfkurdi.com", "known_media", "news_media", "ANF Kurdish edition"),
        ("https://anfarabic.com", "known_media", "news_media", "ANF Arabic edition"),
        ("https://kurdistanpress.com", "known_media", "news_media", "Kurdistan Press"),
        ("https://ekurd.net", "known_media", "news_media", "eKurd.net daily news"),
        ("https://kurdistantribune.com", "known_media", "news_media", "Kurdistan Tribune"),
        ("https://kurdishglobe.net", "known_media", "news_media", "Kurdish Globe"),
        ("https://firatajans.com", "known_media", "news_media", "Firat News Agency"),
        ("https://firatnews.com", "known_media", "news_media", "Firat News"),
        ("https://jiyanpress.com", "known_media", "news_media", "Jiyan Press"),
        ("https://rozhnamavan.com", "known_media", "news_media", "Rozhnamavan news portal"),
        ("https://kurdiu.org", "known_media", "news_media", "Kurdiu media"),
        ("https://sharpress.net", "known_media", "news_media", "Sharpress news"),
        ("https://knnc.net", "known_media", "news_media", "KNN news channel"),
        ("https://payam.tv", "known_media", "news_media", "Payam TV"),
        ("https://gelawej.net", "known_media", "news_media", "Gelawej news"),
        ("https://bianet.org", "known_media", "news_media", "Bianet independent news"),
        ("https://mezopotamyaajansi.com", "known_media", "news_media", "Mezopotamya Agency"),
        ("https://rojakurdistan.com", "known_media", "news_media", "Roja Kurdistan"),
        ("https://kurdistannews.net", "known_media", "news_media", "Kurdistan News"),
        ("https://dengikurdistan.net", "known_media", "news_media", "Dengi Kurdistan"),
        ("https://radiokurdistan.net", "known_media", "news_media", "Radio Kurdistan"),
        ("https://dengikurdistan.com", "known_media", "news_media", "Dengi Kurdistan (com)"),
        ("https://voakurdish.com", "known_media", "news_media", "VOA Kurdish service"),
        ("https://bbckurdi.com", "known_media", "news_media", "BBC Kurdish"),
        ("https://bbc.com/kurdish", "known_media", "news_media", "BBC Kurdish section"),
        ("https://dw.com/ku", "known_media", "news_media", "Deutsche Welle Kurmanji"),
        ("https://dw.com/ckb", "known_media", "news_media", "Deutsche Welle Sorani"),
        ("https://france24.com/ku", "known_media", "news_media", "France 24 Kurdish"),
        ("https://trtkurdi.com.tr", "known_media", "news_media", "TRT Kurdi"),
        ("https://medyanews.com", "known_media", "news_media", "Medya News"),
        ("https://medyanews.net", "known_media", "news_media", "Medya News (net)"),
        ("https://rohanipress.com", "known_media", "news_media", "Rohani Press"),
        ("https://evinkurd.com", "known_media", "news_media", "Evin Kurd"),
        ("https://jwanpress.com", "known_media", "news_media", "Jwan Press"),
        ("https://khakpress.com", "known_media", "news_media", "Khak Press"),
        ("https://aweza.net", "known_media", "news_media", "Aweza news"),
        ("https://kurdpress.ir", "known_media", "news_media", "KurdPress Iran"),
        ("https://aknews.com", "known_media", "news_media", "AK News"),
        ("https://chaknews.com", "known_media", "news_media", "Chak News"),
        ("https://kurdistanchronicle.com", "known_media", "news_media", "Kurdistan Chronicle"),
        ("https://kurdpa.net", "known_media", "news_media", "Kurdpa news agency"),
        ("https://theinsightinternational.com", "known_media", "news_media", "The Insight International"),
        ("https://inkdrop.net/news/kurdistan", "known_media", "news_media", "Inkdrop Kurdistan news"),
        ("https://manaramagazine.org", "known_media", "news_media", "Manara Magazine"),
        ("https://middleeasteye.net/topics/kurds", "known_media", "news_media", "Middle East Eye Kurdish topics"),
        ("https://newsnow.com/us/World/Asia/Kurdistan", "known_media", "news_media", "NewsNow Kurdistan aggregator"),
        ("https://arabmediasociety.com", "known_media", "news_media", "Arab Media Society"),
        ("https://rojnews.news", "known_media", "news_media", "Roj News"),
        ("https://sbeiy.com", "known_media", "news_media", "Sbeiy media"),
        ("https://xoybun.com", "known_media", "news_media", "Xoybun media"),
        ("https://roj.tv", "known_media", "news_media", "Roj TV"),
        ("https://sterk.tv", "known_media", "news_media", "Sterk TV"),
        ("https://nuceciwan.info", "known_media", "news_media", "Nuceciwan news"),
        ("https://ciwannews.com", "known_media", "news_media", "Ciwan News"),
        ("https://navenda-nuceyan.com", "known_media", "news_media", "Navenda Nuceyan"),
        ("https://rizgarinews.com", "known_media", "news_media", "Rizgari News"),
        ("https://aramnews.com", "known_media", "news_media", "Aram News"),
        ("https://dangiiran.com", "known_media", "news_media", "Dangi Iran"),
        ("https://danginwe.com", "known_media", "news_media", "Dangi Nwe"),
        ("https://payamnwe.com", "known_media", "news_media", "Payam Nwe"),
        ("https://xelk.org", "known_media", "news_media", "Xelk media"),
        ("https://penase.com", "known_media", "news_media", "Penase news"),
        ("https://speemedia.com", "known_media", "news_media", "Spee Media"),
        ("https://newrozpost.com", "known_media", "news_media", "Newroz Post"),
        ("https://smartnews.agency", "known_media", "news_media", "Smart News Agency"),
        ("https://zhiannews.com", "known_media", "news_media", "Zhian News"),
        ("https://livancn.com", "known_media", "news_media", "Livancn news"),
    ]

    # =========================================================================
    # TV CHANNELS
    # =========================================================================
    tv_channels = [
        ("https://kurdistantv.net", "tv_channels", "tv_channels", "Kurdistan TV"),
        ("https://kurdsat.tv", "tv_channels", "tv_channels", "KurdSat TV"),
        ("https://zagros.tv", "tv_channels", "tv_channels", "Zagros TV"),
        ("https://gksat.tv", "tv_channels", "tv_channels", "GK Sat TV"),
        ("https://vintv.net", "tv_channels", "tv_channels", "Vin TV"),
        ("https://newroz.tv", "tv_channels", "tv_channels", "Newroz TV"),
        ("https://mmc.tv", "tv_channels", "tv_channels", "MMC TV"),
        ("https://kurditv.com", "tv_channels", "tv_channels", "Kurdi TV"),
        ("https://kurdtvs.net", "tv_channels", "tv_channels", "Kurd TVs directory"),
        ("https://kurd1.tv", "tv_channels", "tv_channels", "Kurd1 TV"),
        ("https://kurdmax.tv", "tv_channels", "tv_channels", "KurdMax TV"),
        ("https://koreksat.tv", "tv_channels", "tv_channels", "Korek Sat TV"),
        ("https://tishk.tv", "tv_channels", "tv_channels", "Tishk TV"),
        ("https://speda.tv", "tv_channels", "tv_channels", "Speda TV"),
        ("https://zaroktv.com", "tv_channels", "tv_channels", "Zarok TV children's channel"),
        ("https://jintv.org", "tv_channels", "tv_channels", "Jin TV"),
        ("https://nrt.tv", "tv_channels", "tv_channels", "NRT TV"),
        ("https://waar.tv", "tv_channels", "tv_channels", "Waar TV"),
        ("https://zoomnews.tv", "tv_channels", "tv_channels", "Zoom News TV"),
        ("https://karwan.tv", "tv_channels", "tv_channels", "Karwan TV"),
        ("https://kurdchannel.tv", "tv_channels", "tv_channels", "Kurd Channel TV"),
    ]

    # =========================================================================
    # RADIO
    # =========================================================================
    radio_stations = [
        ("https://radionawa.com", "radio_stations", "radio", "Radio Nawa"),
        ("https://azadifm.com", "radio_stations", "radio", "Azadi FM"),
        ("https://dengewelat.com", "radio_stations", "radio", "Denge Welat radio"),
        ("https://havin.fm", "radio_stations", "radio", "Havin FM"),
        ("https://kurd1fm.com", "radio_stations", "radio", "Kurd1 FM"),
        ("https://rudawnewsradio.com", "radio_stations", "radio", "Rudaw News Radio"),
        ("https://trtradiokurdi.com", "radio_stations", "radio", "TRT Radio Kurdi"),
        ("https://dengeradio.com", "radio_stations", "radio", "Denge Radio"),
        ("https://kurdmix.com", "radio_stations", "radio", "KurdMix radio"),
        ("https://radiogorran.com", "radio_stations", "radio", "Radio Gorran"),
        ("https://nawamedia.com", "radio_stations", "radio", "Nawa Media"),
    ]

    # =========================================================================
    # GOVERNMENT & INSTITUTIONS
    # =========================================================================
    government = [
        ("https://gov.krd", "government", "government", "KRG official portal"),
        ("https://krg.org", "government", "government", "KRG international site"),
        ("https://parliament.krd", "government", "government", "Kurdistan Parliament"),
        ("https://presidency.krd", "government", "government", "Kurdistan Region Presidency"),
        ("https://cabinet.gov.krd", "government", "government", "KRG Cabinet"),
        ("https://mof.gov.krd", "government", "government", "Ministry of Finance"),
        ("https://mofa.gov.krd", "government", "government", "Ministry of Foreign Affairs"),
        ("https://moi.gov.krd", "government", "government", "Ministry of Interior"),
        ("https://moj.gov.krd", "government", "government", "Ministry of Justice"),
        ("https://krso.gov.krd", "government", "government", "Kurdistan Region Statistics Office"),
        ("https://moe-krg.com", "government", "government", "Ministry of Education"),
        ("https://health.gov.krd", "government", "government", "Ministry of Health"),
        ("https://mnr.gov.krd", "government", "government", "Ministry of Natural Resources"),
        ("https://mop.gov.krd", "government", "government", "Ministry of Planning"),
        ("https://mot.gov.krd", "government", "government", "Ministry of Transport"),
        ("https://mol.gov.krd", "government", "government", "Ministry of Labour"),
        ("https://mohe.gov.krd", "government", "government", "Ministry of Higher Education"),
        ("https://moe.gov.krd", "government", "government", "Ministry of Education (gov.krd)"),
        ("https://gov.krd/moh-en", "government", "government", "Ministry of Health English page"),
        ("https://gov.krd/dngo-en", "government", "government", "Dept of NGO Affairs"),
        ("https://hawlergov.org", "government", "government", "Hawler Governorate"),
        ("https://visitkurdistan.krd", "government", "government", "Kurdistan tourism board"),
        ("https://syriandemocraticcouncil.us", "government", "government", "Syrian Democratic Council"),
        ("https://dfns.net", "government", "government", "Democratic Federation of Northern Syria"),
        ("https://sdf-press.com", "government", "government", "SDF Press"),
        ("https://afrinkanton.com", "government", "government", "Afrin Canton"),
        ("https://tevdem.org", "government", "government", "TEV-DEM"),
    ]

    # =========================================================================
    # POLITICAL PARTIES
    # =========================================================================
    political_parties = [
        ("https://kdp.info", "political_parties", "political_parties", "Kurdistan Democratic Party"),
        ("https://puk.org", "political_parties", "political_parties", "Patriotic Union of Kurdistan"),
        ("https://gorran.net", "political_parties", "political_parties", "Gorran Movement"),
        ("https://gorannews.net", "political_parties", "political_parties", "Gorran Movement news"),
        ("https://yekgirtu.org", "political_parties", "political_parties", "Kurdistan Islamic Union"),
        ("https://komal.org", "political_parties", "political_parties", "Kurdistan Islamic Group"),
        ("https://nfrtkurdistan.com", "political_parties", "political_parties", "New Generation Movement"),
        ("https://kdsp.org", "political_parties", "political_parties", "Kurdistan Democratic Solution Party"),
        ("https://pdk-iran.org", "political_parties", "political_parties", "PDK-Iran"),
        ("https://pdki.org", "political_parties", "political_parties", "PDKI"),
        ("https://komelakurdistan.org", "political_parties", "political_parties", "Komala Party of Iranian Kurdistan"),
        ("https://komala.org", "political_parties", "political_parties", "Komala Party"),
        ("https://pjak.org", "political_parties", "political_parties", "PJAK"),
        ("https://enks.info", "political_parties", "political_parties", "ENKS Syrian Kurdish National Council"),
        ("https://pkkonline.com", "political_parties", "political_parties", "PKK online"),
        ("https://kongrakurdistan.net", "political_parties", "political_parties", "Kongra Kurdistan"),
        ("https://ypgrojava.org", "political_parties", "political_parties", "YPG Rojava"),
        ("https://kck-info.com", "political_parties", "political_parties", "KCK info"),
        ("https://hpg-online.com", "political_parties", "political_parties", "HPG online"),
        ("https://pydrojava.com", "political_parties", "political_parties", "PYD Rojava"),
        ("https://sfrk.info", "political_parties", "political_parties", "SFRK info"),
        ("https://newgeneration.krd", "political_parties", "political_parties", "New Generation Movement (krd)"),
        ("https://azadiyawelat.com", "political_parties", "political_parties", "Azadiya Welat"),
    ]

    # =========================================================================
    # EDUCATION (UNIVERSITIES)
    # =========================================================================
    universities = [
        ("https://uos.edu.krd", "universities", "education", "University of Sulaimani"),
        ("https://su.edu.krd", "universities", "education", "Salahaddin University"),
        ("https://hmu.edu.krd", "universities", "education", "Hawler Medical University"),
        ("https://uod.ac", "universities", "education", "University of Duhok"),
        ("https://epu.edu.iq", "universities", "education", "Erbil Polytechnic University"),
        ("https://uhd.edu.iq", "universities", "education", "University of Human Development"),
        ("https://auis.edu.krd", "universities", "education", "American University of Iraq Sulaimani"),
        ("https://cihanuniversity.edu.iq", "universities", "education", "Cihan University Erbil"),
        ("https://koyauniversity.org", "universities", "education", "Koya University"),
        ("https://uor.edu.krd", "universities", "education", "University of Raparin"),
        ("https://tiu.edu.iq", "universities", "education", "Tishk International University"),
        ("https://lfu.edu.krd", "universities", "education", "Lebanese French University"),
        ("https://ukh.edu.krd", "universities", "education", "University of Kurdistan Hewler"),
        ("https://ishik.edu.iq", "universities", "education", "Ishik University"),
        ("https://charmo.edu.krd", "universities", "education", "Charmo University"),
        ("https://halabjauni.edu.krd", "universities", "education", "University of Halabja"),
        ("https://garmian.edu.krd", "universities", "education", "University of Garmian"),
        ("https://raperin.edu.krd", "universities", "education", "University of Raparin (raperin)"),
        ("https://komar.edu.iq", "universities", "education", "Komar University"),
        ("https://bnu.edu.iq", "universities", "education", "BNU Iraq"),
        ("https://knowledge.edu.krd", "universities", "education", "Knowledge University"),
        ("https://cihan.edu.iq", "universities", "education", "Cihan University"),
        ("https://knu.edu.iq", "universities", "education", "Kurdistan National University"),
        ("https://main.soran.edu.iq", "universities", "education", "Soran University"),
        ("https://su.edu.iq", "universities", "education", "Sulaimani University (iq)"),
    ]

    # =========================================================================
    # CULTURAL & REFERENCE
    # =========================================================================
    cultural = [
        ("https://kurdishacademy.org", "cultural", "cultural", "Kurdish Academy of Language"),
        ("https://institutkurde.org", "cultural", "cultural", "Institut Kurde de Paris"),
        ("https://kurdipedia.org", "cultural", "cultural", "Kurdipedia encyclopedia"),
        ("https://pertukal.com", "cultural", "cultural", "Pertuk-al Kurdish books"),
        ("https://amude.net", "cultural", "cultural", "Amude cultural portal"),
        ("https://kurdistanica.com", "cultural", "cultural", "Kurdistanica encyclopedia"),
        ("https://bnk.gov.krd", "cultural", "cultural", "Kurdistan National Library"),
        ("https://ckb.wikipedia.org", "cultural", "cultural", "Wikipedia Sorani Kurdish"),
        ("https://ku.wikipedia.org", "cultural", "cultural", "Wikipedia Kurmanji Kurdish"),
        ("https://ku.wiktionary.org", "cultural", "cultural", "Wiktionary Kurmanji Kurdish"),
        ("https://ckb.wiktionary.org", "cultural", "cultural", "Wiktionary Sorani Kurdish"),
        ("https://ferheng.org", "cultural", "cultural", "Ferheng Kurdish dictionary (org)"),
        ("https://ferheng.com", "cultural", "cultural", "Ferheng Kurdish dictionary (com)"),
        ("https://kurdi.com", "cultural", "cultural", "Kurdi.com cultural portal"),
        ("https://kurd.cc", "cultural", "cultural", "Kurd.cc"),
        ("https://kurdishlyrics.com", "cultural", "cultural", "Kurdish song lyrics"),
        ("https://kurdishmusic.eu", "cultural", "cultural", "Kurdish music archive"),
        ("https://kurdishcinema.com", "cultural", "cultural", "Kurdish cinema"),
        ("https://kurdishfilm.com", "cultural", "cultural", "Kurdish film"),
        ("https://kurditgroup.org", "cultural", "cultural", "Kurdit Group"),
        ("https://krmanj.org", "cultural", "cultural", "Krmanj.org Kurmanji resources"),
        ("https://sardam.info", "cultural", "cultural", "Sardam cultural magazine"),
        ("https://amako.info", "cultural", "cultural", "Amako info"),
        ("https://welatparez.com", "cultural", "cultural", "Welat Parez"),
        ("https://zivmagazine.com", "cultural", "cultural", "Ziv Magazine"),
        ("https://kurdistanmemory.com", "cultural", "cultural", "Kurdistan Memory Programme"),
        ("https://kurd.org", "cultural", "cultural", "Kurd.org"),
        ("https://kurdishculture.com", "cultural", "cultural", "Kurdish Culture"),
        ("https://kurdishheritage.org", "cultural", "cultural", "Kurdish Heritage Foundation"),
        ("https://kurdishproject.org", "cultural", "cultural", "The Kurdish Project"),
        ("https://thekurdishproject.org", "cultural", "cultural", "The Kurdish Project (the)"),
        ("https://kurdishstudies.net", "cultural", "cultural", "Kurdish Studies journal"),
        ("https://kurdishcentre.com", "cultural", "cultural", "Kurdish Centre"),
        ("https://kurdishlobby.com", "cultural", "cultural", "Kurdish Lobby"),
        ("https://kurdishrights.org", "cultural", "cultural", "Kurdish Rights"),
        ("https://khrp.org", "cultural", "cultural", "Kurdish Human Rights Project"),
        ("https://kurdistanhumanrights.org", "cultural", "cultural", "Kurdistan Human Rights"),
        ("https://kurdistancommentary.com", "cultural", "cultural", "Kurdistan Commentary"),
        ("https://zheen.com", "cultural", "cultural", "Zheen cultural center"),
        ("https://regaonline.com", "cultural", "cultural", "Rega Online"),
        ("https://kanipedia.org", "cultural", "cultural", "Kanipedia"),
        ("https://kurdophone.com", "cultural", "cultural", "Kurdophone"),
        ("https://100berhemenkurdi.com", "cultural", "cultural", "100 Kurdish works"),
        ("https://sinaahmadi.github.io", "cultural", "cultural", "Sina Ahmadi Kurdish NLP research"),
        ("https://kurdishcentral.org", "cultural", "cultural", "Kurdish Central"),
        ("https://kurdishlessons.com", "cultural", "cultural", "Kurdish Lessons"),
        ("https://kurdishpeople.org", "cultural", "cultural", "Kurdish People"),
        ("https://zimannas.wordpress.com", "cultural", "cultural", "Zimannas Kurdish language blog"),
        ("https://kurdilit.net", "cultural", "cultural", "Kurdish literature"),
        ("https://c4kurd.com", "cultural", "cultural", "C4Kurd"),
        ("https://koord.com", "cultural", "cultural", "Koord.com"),
        ("https://parstimes.com/kurdish.html", "cultural", "cultural", "ParsTimes Kurdish section"),
    ]

    # =========================================================================
    # RELIGIOUS & MINORITY
    # =========================================================================
    religious = [
        ("https://ezidikhan.com", "religious", "religious", "Ezidikhan Yezidi governance"),
        ("https://lalish.info", "religious", "religious", "Lalish Yezidi temple info"),
        ("https://yezidi.org", "religious", "religious", "Yezidi international"),
        ("https://yezidipost.com", "religious", "religious", "Yezidi Post news"),
        ("https://ezidikan.com", "religious", "religious", "Ezidikan Yezidi portal"),
        ("https://lalish.org", "religious", "religious", "Lalish cultural center"),
        ("https://yezidism.com", "religious", "religious", "Yezidism information"),
        ("https://shareyezidi.com", "religious", "religious", "Share Yezidi"),
        ("https://ezidpress.com", "religious", "religious", "Ezid Press news"),
        ("https://dengeyezidi.com", "religious", "religious", "Denge Yezidi"),
        ("https://shingalmedia.com", "religious", "religious", "Shingal Media"),
        ("https://ankawa.com", "religious", "religious", "Ankawa Assyrian-Kurdish forum"),
        ("https://chaldean.org", "religious", "religious", "Chaldean community"),
        ("https://christiankurdistan.com", "religious", "religious", "Christian Kurdistan"),
        ("https://qurankurdi.com", "religious", "religious", "Quran Kurdish translation"),
        ("https://islamkurd.com", "religious", "religious", "Islam Kurd"),
        ("https://difraq.com", "religious", "religious", "Difraq"),
        ("https://ehlibeyt.info", "religious", "religious", "Ehlibeyt info"),
        ("https://kaniyanews.com", "religious", "religious", "Kaniya News"),
        ("https://xudanews.com", "religious", "religious", "Xuda News"),
        ("https://hawramani.com", "religious", "religious", "Hawramani Islamic studies"),
    ]

    # =========================================================================
    # E-COMMERCE & BUSINESS
    # =========================================================================
    ecommerce = [
        ("https://boombeene.com", "ecommerce", "ecommerce", "Boombeene online store"),
        ("https://ebazar.net", "ecommerce", "ecommerce", "eBazar marketplace"),
        ("https://kurdistan-store.com", "ecommerce", "ecommerce", "Kurdistan Store"),
        ("https://kurdshopping.com", "ecommerce", "ecommerce", "Kurd Shopping"),
        ("https://pelemall.com", "ecommerce", "ecommerce", "Pelemall online mall"),
        ("https://kurdistansupermarket.com", "ecommerce", "ecommerce", "Kurdistan Supermarket"),
        ("https://arbela.store", "ecommerce", "ecommerce", "Arbela Store"),
        ("https://mybestshopping.net", "ecommerce", "ecommerce", "My Best Shopping"),
        ("https://bazarekurd.com", "ecommerce", "ecommerce", "Bazare Kurd"),
        ("https://slemani.com", "ecommerce", "ecommerce", "Slemani city portal"),
        ("https://erbilcitadel.org", "ecommerce", "ecommerce", "Erbil Citadel"),
        ("https://sulaymaniyah.com", "ecommerce", "ecommerce", "Sulaymaniyah city portal"),
        ("https://duhok.org", "ecommerce", "ecommerce", "Duhok city portal"),
        ("https://hawler.info", "ecommerce", "ecommerce", "Hawler city info"),
        ("https://kurd.store", "ecommerce", "ecommerce", "Kurd Store"),
    ]

    # =========================================================================
    # DIASPORA & INTERNATIONAL
    # =========================================================================
    diaspora = [
        ("https://kurdistan.co.uk", "diaspora", "diaspora", "Kurdistan UK"),
        ("https://kurdistan-report.de", "diaspora", "diaspora", "Kurdistan Report Germany"),
        ("https://kurdishmedia.com", "diaspora", "diaspora", "Kurdish Media"),
        ("https://kurdishquestion.com", "diaspora", "diaspora", "The Kurdish Question"),
        ("https://kurdishinstitute.be", "diaspora", "diaspora", "Kurdish Institute Brussels"),
        ("https://navkurd.de", "diaspora", "diaspora", "Navkurd Germany"),
        ("https://fek-online.com", "diaspora", "diaspora", "FEK Online"),
        ("https://kurdistanweekly.com", "diaspora", "diaspora", "Kurdistan Weekly"),
        ("https://yxkonline.com", "diaspora", "diaspora", "YXK Online"),
        ("https://kurdistan.nl", "diaspora", "diaspora", "Kurdistan Netherlands"),
        ("https://kurdistan.se", "diaspora", "diaspora", "Kurdistan Sweden"),
        ("https://kurdistanforum.de", "diaspora", "diaspora", "Kurdistan Forum Germany"),
        ("https://kurdwatch.org", "diaspora", "diaspora", "KurdWatch Syria monitoring"),
        ("https://civaka-azad.org", "diaspora", "diaspora", "Civaka Azad Kurdish centre"),
        ("https://ronahi.net", "diaspora", "diaspora", "Ronahi net"),
        ("https://amed.to", "diaspora", "diaspora", "Amed portal"),
        ("https://efrin.net", "diaspora", "diaspora", "Efrin community"),
        ("https://kobane.com", "diaspora", "diaspora", "Kobane community"),
        ("https://kurdistanamericalatina.org", "diaspora", "diaspora", "Kurdistan Latin America"),
        ("https://kurdishinstitute.org", "diaspora", "diaspora", "Kurdish Institute"),
        ("https://kurdistanfirst.com", "diaspora", "diaspora", "Kurdistan First"),
        ("https://kurdishnews.com", "diaspora", "diaspora", "Kurdish News"),
        ("https://kurdishjournalists.com", "diaspora", "diaspora", "Kurdish Journalists"),
        ("https://kurdistancenter.com", "diaspora", "diaspora", "Kurdistan Center"),
        ("https://diakurdistan.com", "diaspora", "diaspora", "Dia Kurdistan"),
        ("https://kurdistangeneration.com", "diaspora", "diaspora", "Kurdistan Generation"),
        ("https://defendrojava.org", "diaspora", "diaspora", "Defend Rojava"),
        ("https://kmewo.com", "diaspora", "diaspora", "KMEWO Kurdish women"),
        ("https://dckurd.org", "diaspora", "diaspora", "DC Kurd"),
        ("https://kurdishpeace.org", "diaspora", "diaspora", "Kurdish Peace"),
    ]

    # =========================================================================
    # NGOs & HUMAN RIGHTS
    # =========================================================================
    ngo = [
        ("https://kurdsngo.org", "ngo", "ngo", "Kurds NGO"),
        ("https://kohrw.org", "ngo", "ngo", "Kurdistan Organization for Human Rights Watch"),
        ("https://borgenproject.org/womens-empowerment-in-kurdistan", "ngo", "ngo", "Borgen Project Kurdistan women"),
        ("https://harikar.org", "ngo", "ngo", "Harikar NGO"),
        ("https://warvin.org", "ngo", "ngo", "Warvin Organization"),
        ("https://womeninwar.org", "ngo", "ngo", "Women in War"),
    ]

    # =========================================================================
    # HEALTH
    # =========================================================================
    health = [
        ("https://hih-iq.com", "health", "health", "HIH Iraq hospital"),
        ("https://eiherbil.com", "health", "health", "EIH Erbil hospital"),
        ("https://maryamanahospital.com", "health", "health", "Maryamana Hospital"),
        ("https://farukmedicalcity.com", "health", "health", "Faruk Medical City"),
        ("https://kurdistanclinic.com", "health", "health", "Kurdistan Clinic"),
        ("https://parhospital.org", "health", "health", "Par Hospital"),
        ("https://cmcph.net", "health", "health", "CMC Private Hospital"),
    ]

    # =========================================================================
    # TECHNOLOGY
    # =========================================================================
    tech = [
        ("https://kurddev.com", "tech", "technology", "KurdDev developer community"),
        ("https://kurd.io", "tech", "technology", "Kurd.io tech platform"),
        ("https://kurdtechno.com", "tech", "technology", "Kurd Techno"),
        ("https://sinaahmadi.github.io/klpt", "tech", "technology", "Kurdish Language Processing Toolkit"),
        ("https://github.com/sinaahmadi/awesome-kurdish", "tech", "technology", "Awesome Kurdish GitHub repo"),
        ("https://github.com/sinaahmadi/KurdishLID", "tech", "technology", "Kurdish Language Identification"),
        ("https://github.com/topics/kurdish", "tech", "technology", "GitHub Kurdish topic"),
    ]

    # =========================================================================
    # SPORTS
    # =========================================================================
    sports = [
        ("https://kurdistansport.com", "sports", "sports", "Kurdistan Sport"),
        ("https://varzkurdistan.com", "sports", "sports", "Varz Kurdistan"),
        ("https://kurdsport.net", "sports", "sports", "Kurd Sport"),
        ("https://sportskurd.com", "sports", "sports", "Sports Kurd"),
        ("https://erbilmarathon.com", "sports", "sports", "Erbil Marathon"),
    ]

    # =========================================================================
    # BLOGS & FORUMS
    # =========================================================================
    blogs = [
        ("https://kurdblogger.com", "blogs", "blogs", "Kurd Blogger"),
        ("https://kurdforum.com", "blogs", "blogs", "Kurd Forum"),
        ("https://kurdnet.com", "blogs", "blogs", "KurdNet"),
        ("https://binav.com", "blogs", "blogs", "Binav"),
        ("https://zanist.org", "blogs", "blogs", "Zanist research"),
        ("https://newroz.com", "blogs", "blogs", "Newroz community"),
        ("https://azadikurd.com", "blogs", "blogs", "Azadi Kurd"),
        ("https://hevallo.com", "blogs", "blogs", "Hevallo blog"),
        ("https://kurdishdailynews.com", "blogs", "blogs", "Kurdish Daily News"),
        ("https://thekurdisharena.com", "blogs", "blogs", "The Kurdish Arena"),
        ("https://malpera.net", "blogs", "blogs", "Malpera Kurdish blogs"),
        ("https://peykurdistan.com", "blogs", "blogs", "Pey Kurdistan"),
        ("https://jinnews.com.tr", "blogs", "blogs", "JinNews Turkey"),
        ("https://jinhaagency.com", "blogs", "blogs", "Jinha Agency"),
        ("https://jinlife.com", "blogs", "blogs", "Jin Life"),
        ("https://zankoline.com", "blogs", "blogs", "Zanko Online"),
        ("https://nefel.com", "blogs", "blogs", "Nefel"),
        ("https://wlatey.me", "blogs", "blogs", "Wlatey blog"),
        ("https://ruwange.com", "blogs", "blogs", "Ruwange"),
        ("https://hawpshti.com", "blogs", "blogs", "Hawpshti"),
        ("https://kurdistanblogcount.wordpress.com", "blogs", "blogs", "Kurdistan Blog Count"),
    ]

    # =========================================================================
    # Combine all categories
    # =========================================================================
    all_entries = (
        news_media
        + tv_channels
        + radio_stations
        + government
        + political_parties
        + universities
        + cultural
        + religious
        + ecommerce
        + diaspora
        + ngo
        + health
        + tech
        + sports
        + blogs
    )

    # Deduplicate by URL (keep first occurrence)
    seen_urls = set()
    for url, source, category, notes in all_entries:
        normalized = url.rstrip("/").lower()
        if normalized not in seen_urls:
            seen_urls.add(normalized)
            urls.append({
                "url": url,
                "source": source,
                "category": category,
                "notes": notes,
            })

    return urls


def get_urls_by_category(category):
    """Return all URLs matching the given category string."""
    return [
        entry for entry in get_all_expanded_urls()
        if entry["category"] == category
    ]


def get_urls_by_source(source):
    """Return all URLs matching the given source string."""
    return [
        entry for entry in get_all_expanded_urls()
        if entry["source"] == source
    ]


def get_all_categories():
    """Return sorted list of unique category names."""
    return sorted({entry["category"] for entry in get_all_expanded_urls()})


def get_all_sources():
    """Return sorted list of unique source names."""
    return sorted({entry["source"] for entry in get_all_expanded_urls()})


if __name__ == "__main__":
    all_urls = get_all_expanded_urls()
    print(f"Total unique URLs: {len(all_urls)}")
    print()
    categories = get_all_categories()
    for cat in categories:
        count = len(get_urls_by_category(cat))
        print(f"  {cat}: {count}")
