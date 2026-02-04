from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import re
from urllib.parse import urlparse

# Baseline feature contract
NUMERIC_FEATURES_BASELINE = [
    "user_statuses_count",
    "user_favourites_count",
    "user_listed_count",
    "account_age_days",
    "user_description_len",
    "text_length_chars",
    "text_length_words",
    "n_hashtags",
    "n_mentions",
    "n_urls",
    "n_media",
    "n_photo",
    "n_video",
    "n_gif",
    "tweet_hour",
    "tweet_dow",
    "quoted_retweet_count",
    "quoted_favorite_count",
    "quoted_reply_count",
    "quoted_quote_count",
    "quoted_user_followers_count",
    "quoted_user_friends_count",
    "hashtags_per_word",
    "mentions_per_word",
    "urls_per_word",
    "media_per_word",
]

BOOL_FEATURES_BASELINE = [
    "is_reply",
    "is_quote",
    "is_standalone",
    "has_media",
    "has_hashtag",
    "has_mention",
    "has_url",
    "has_place",
    "user_has_location",
    "is_weekend",
    "quoted_user_verified",
    "source_is_twitter",
]

CATEGORICAL_FEATURES_BASELINE = [
    "source",
    "source_label",
    "source_domain",
    "source_family",
    "device_class",
    "device_os",
    "device_detail",
    "browser_family",
    "place_country",
    "place_country_code",
    "place_type",
    "quoted_lang",
]

REQUIRED_FEATURES = (
    ["challenge_id", "label", "text"]
    + NUMERIC_FEATURES_BASELINE
    + BOOL_FEATURES_BASELINE
    + CATEGORICAL_FEATURES_BASELINE
)

def parse_source_raw(raw_source: Optional[str]) -> Dict[str, Any]:
    """
    Parse a raw Twitter 'source' HTML string (e.g. <a href=...>Twitter for iPhone</a>)
    into several useful fields:
      - source_label: human-readable client name
      - source_href:  link URL
      - source_domain: domain (e.g. twitter.com, buffer.com)
      - source_is_twitter: bool – official Twitter client / property?
      - source_family: twitter_native / social_media_tool / other_social_network /
                       news_media / blog_cms / other_web_or_app / unknown
      - device_class: mobile / tablet / desktop / web / unknown
      - device_os: ios / android / macos / unknown
      - device_detail: finer-grain name (iphone, ipad, tweetdeck, ...)
      - browser_family: in_app / generic_web / server_or_web / in_app_or_web / unknown
    """
    # Handle missing/non-string gracefully
    if not isinstance(raw_source, str):
        return {
            'source_label': None,
            'source_href': None,
            'source_domain': None,
            'source_is_twitter': False,
            'source_family': 'unknown',
            'device_class': 'unknown',
            'device_os': 'unknown',
            'device_detail': None,
            'browser_family': 'unknown',
        }

    # --- Extract href + visible label from the <a> tag ---
    m_href = re.search(r'href="([^"]+)"', raw_source)
    href = m_href.group(1) if m_href else None

    m_text = re.search(r'>([^<]+)</a>', raw_source)
    label = m_text.group(1).strip() if m_text else raw_source

    # --- Domain parsing ---
    domain_root = None
    if href:
        parsed = urlparse(href)
        domain = parsed.netloc.lower()
        domain_root = domain.split(':')[0]
    else:
        domain_root = ''

    label_l = label.lower()
    dom = domain_root or ''

    # --- Is this a Twitter-owned client? ---
    is_twitter = ('twitter' in dom) or label_l.startswith('twitter ') or 'tweetdeck' in label_l

    # --- Device classification ---
    device_class = 'unknown'
    device_os = 'unknown'
    device_detail = None

    # Official Twitter clients first
    if 'twitter for iphone' in label_l or 'twitter pour iphone' in label_l or ('iphone' in label_l and 'twitter' in label_l):
        device_class = 'mobile'
        device_os = 'ios'
        device_detail = 'iphone'
    elif 'twitter for ipad' in label_l or 'twitter pour ipad' in label_l or ('ipad' in label_l and 'twitter' in label_l):
        device_class = 'tablet'
        device_os = 'ios'
        device_detail = 'ipad'
    elif 'twitter for mac' in label_l:
        device_class = 'desktop'
        device_os = 'macos'
        device_detail = 'mac'
    elif 'twitter for android' in label_l or ('android' in label_l and 'twitter' in label_l):
        device_class = 'mobile'
        device_os = 'android'
        device_detail = 'android_phone'
    elif 'twitter web app' in label_l or 'twitter web client' in label_l:
        device_class = 'web'
        device_os = 'unknown'
        device_detail = 'twitter_web'
    elif 'tweetdeck' in label_l:
        device_class = 'desktop'
        device_os = 'unknown'
        device_detail = 'tweetdeck'
    elif 'media studio' in label_l:
        device_class = 'web'
        device_detail = 'twitter_media_studio'

    # Non-Twitter but mobile hints
    if device_class == 'unknown':
        if 'ipad' in label_l or 'iphone' in label_l or 'ios' in label_l:
            device_class = 'mobile'
            device_os = 'ios'
            device_detail = device_detail or 'ios_device'
        elif 'android' in label_l:
            device_class = 'mobile'
            device_os = 'android'
            device_detail = device_detail or 'android_device'

    # --- Source family classification ---
    social_network_domains = {
        'instagram.com', 'www.instagram.com',
        'tumblr.com', 'www.tumblr.com',
        'linkedin.com', 'www.linkedin.com',
        'periscope.tv',
        'facebook.com', 'www.facebook.com',
        'stocktwits.com', 'www.stocktwits.com',
    }

    social_tools_domains = {
        'hootsuite.com', 'www.hootsuite.com',
        'buffer.com', 'www.buffer.com',
        'dlvrit.com', 'dlvr.it',
        'ifttt.com',
        'zapier.com',
        'sproutsocial.com', 'www.sproutsocial.com',
        'app.agorapulse.com', 'agorapulse.com',
        'sprinklr.com', 'www.sprinklr.com',
        'socialbakers.com', 'www.socialbakers.com',
        'socialflow.com', 'www.socialflow.com',
        'meltwater.com', 'www.meltwater.com',
        'publer.io',
        'app.socialpilot.co', 'socialpilot.co',
        'paper.li', 'www.paper.li',
        'fs-poster.com', 'www.fs-poster.com',
        'echobox.com', 'www.echobox.com',
        'twibble.io',
        'statusbrew.com',
    }

    blog_cms_domains = {
        'publicize.wp.com', 'wordpress.com',
        'over-blog-kiwi.com', 'www.over-blog-kiwi.com',
        'blogspot.com', 'blogger.com',
        'medium.com', 'www.medium.com',
        'scoop.it', 'www.scoop.it',
        'blog2social.com', 'www.blog2social.com',
    }

    # Very rough news heuristic (label or domain hints)
    news_keywords = ['news', 'gazette', 'press', 'presse', 'journal', 'radio', 'tv', 'télé', 'tele']

    if is_twitter:
        source_family = 'twitter_native'
    elif dom in social_network_domains:
        source_family = 'other_social_network'
    elif dom in social_tools_domains:
        source_family = 'social_media_tool'
    elif dom in blog_cms_domains:
        source_family = 'blog_cms'
    else:
        if any(k in label_l for k in news_keywords) or any(k in (dom or '') for k in news_keywords):
            source_family = 'news_media'
        else:
            source_family = 'other_web_or_app'

    # --- Browser family (coarse) ---
    # We CANNOT reliably know Chrome vs Safari vs Firefox.
    # We only infer broad categories.
    browser_family = 'unknown'
    if source_family == 'twitter_native':
        if device_class in {'mobile', 'tablet'}:
            browser_family = 'in_app'         # Twitter mobile apps
        elif device_class in {'desktop', 'web'}:
            browser_family = 'generic_web'    # Web App, TweetDeck, etc.
    elif source_family in {'social_media_tool', 'blog_cms', 'news_media', 'other_web_or_app'}:
        browser_family = 'server_or_web'      # likely scheduled/automated or website
    elif source_family == 'other_social_network':
        browser_family = 'in_app_or_web'      # could be in-app or web client

    return {
        'source_label': label,
        'source_href': href,
        'source_domain': domain_root,
        'source_is_twitter': bool(is_twitter),
        'source_family': source_family,
        'device_class': device_class,
        'device_os': device_os,
        'device_detail': device_detail,
        'browser_family': browser_family,
    }

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and feature-engineer a Twitter dataset *in place*.

    NOTE:
    - This version is a readability refactor of the original cleaning.py.
    - Algorithms, column names, and operations are kept identical.
    - The function mutates `dfc` and does NOT return anything.
    """
    dfc = df.copy()
    # =========================================================
    # 1. DROP USELESS / REDUNDANT COLUMNS
    # =========================================================

    # Columns with >98% missing values or completely empty
    useless_cols = [
        "contributors",          # always null
        "geo",                   # very sparse
        "coordinates",           # very sparse
        "withheld_in_countries", # very sparse, not even in test set
    ]

    # Redundant ID / timestamp columns
    redundant_cols = [
        "timestamp_ms",             # duplicate of created_at
        "in_reply_to_status_id",    # reply tweet ID (numeric)
        "in_reply_to_status_id_str",
        "in_reply_to_user_id",      # replied-to user ID (numeric)
        "in_reply_to_user_id_str",
        "quoted_status_id",         # quoted tweet IDs (numeric)
        "quoted_status_id_str",
    ]

    # Drop them if they exist
    cols_to_drop = [c for c in (useless_cols + redundant_cols) if c in dfc.columns]
    dfc.drop(columns=cols_to_drop, inplace=True)

    # =========================================================
    # 2. TYPE NORMALIZATION / BASIC CLEANING
    # =========================================================

    # ---------- 2.1 Datetime ----------
    if "created_at" in dfc.columns:
        dfc["created_at"] = pd.to_datetime(dfc["created_at"], utc=True, errors="coerce")

    # ---------- 2.2 Clean & standardize `source` ----------
    # Let's keep the source clean now to extract values later.
    # (Currently commented out in original code.)

    # if "source" in dfc.columns:
    #     # ensure string dtype
    #     dfc["source"] = dfc["source"].astype("string")
    #     # strip HTML tags like <a href="...">Twitter for iPhone</a> -> Twitter for iPhone
    #     dfc["source"] = dfc["source"].str.replace(r"<.*?>", "", regex=True).str.strip()

    # ---------- 2.3 Categorical-ish fields ----------
    for col in ["filter_level", "lang", "in_reply_to_screen_name"]:
        if col in dfc.columns:
            # keep NaNs but real values as categories
            dfc[col] = dfc[col].astype("category")

    # ---------- 2.4 Tweet ID as string (not a feature) ----------
    if "id_str" in dfc.columns:
        dfc = dfc.rename(columns={"id_str": "tweet_id"})
        dfc["tweet_id"] = dfc["tweet_id"].astype("string")

    # ---------- 2.5 Numeric counts ----------
    int_cols = [
        "retweet_count",
        "favorite_count",
        "quote_count",
        "reply_count",
        "challenge_id",
        "label",
    ]

    for col in int_cols:
        if col in dfc.columns:
            # Use nullable Int64 in case there are missing values
            dfc[col] = pd.to_numeric(dfc[col], errors="coerce").astype("Int64")

    # ---------- 2.6 Boolean flags ----------
    bool_cols = ["retweeted", "favorited", "truncated", "is_quote_status"]

    for col in bool_cols:
        if col in dfc.columns:
            # if they come as 0/1 or True/False, this will normalize to pandas boolean
            dfc[col] = dfc[col].astype("boolean")

    # ---------- 2.7 possibly_sensitive -> nullable boolean ----------
    if "possibly_sensitive" in dfc.columns:
        ps = dfc["possibly_sensitive"]

        # Some datasets store it as 0/1, some as True/False, some as NaN
        if str(ps.dtype) in ["bool", "boolean"]:
            dfc["possibly_sensitive"] = ps.astype("boolean")
        else:
            # coerce to Int64 (0/1/NaN), then map to boolean
            ps_int = pd.to_numeric(ps, errors="coerce").astype("Int64")
            dfc["possibly_sensitive"] = ps_int.map({1: True, 0: False})
            dfc["possibly_sensitive"] = dfc["possibly_sensitive"].astype("boolean")

    # ---------- 2.8 Text as proper string ----------
    if "text" in dfc.columns:
        dfc["text"] = dfc["text"].astype("string")

    # ---------- 2.9 Ensure complex JSON-ish columns are `object` ----------
    json_like_cols = [
        "quoted_status",
        "user",
        "entities",
        "extended_tweet",
        "extended_entities",
        "place",
        "quoted_status_permalink",
        "display_text_range",
    ]

    for col in json_like_cols:
        if col in dfc.columns:
            dfc[col] = dfc[col].astype("object")

    # Small helpers for nested structures
    def safe_get(d, key, default=np.nan):
        if isinstance(d, dict):
            return d.get(key, default)
        return default

    def safe_len(seq):
        if isinstance(seq, (list, tuple)):
            return len(seq)
        return 0

    # =========================================================
    # 3. USER FEATURES
    # =========================================================

    # Short alias to avoid writing .apply(lambda u: ...) everywhere
    u_series = dfc["user"]

    # ANONYMOUS DATASET
    # dfc["user_id"] = u_series.apply(lambda u: safe_get(u, "id_str"))
    # dfc["user_screen_name"] = u_series.apply(lambda u: safe_get(u, "screen_name"))

    dfc["user_verified"] = u_series.apply(lambda u: safe_get(u, "verified", False))
    dfc["user_verified"] = dfc["user_verified"].astype("boolean")

    # ANONYMOUS DATASET
    # dfc["user_followers_count"] = (
    #     u_series.apply(lambda u: safe_get(u, "followers_count"))
    #             .astype("Int64")
    # )
    # dfc["user_friends_count"] = (
    #     u_series.apply(lambda u: safe_get(u, "friends_count"))
    #             .astype("Int64")
    # )

    dfc["user_statuses_count"] = (
        u_series.apply(lambda u: safe_get(u, "statuses_count")).astype("Int64")
    )
    dfc["user_favourites_count"] = (
        u_series.apply(lambda u: safe_get(u, "favourites_count")).astype("Int64")
    )
    dfc["user_listed_count"] = (
        u_series.apply(lambda u: safe_get(u, "listed_count")).astype("Int64")
    )

    # Description + location
    dfc["user_description"] = u_series.apply(lambda u: safe_get(u, "description"))
    dfc["user_description"] = dfc["user_description"].astype("string")

    dfc['user_description_len'] = dfc['user_description'].apply(
        lambda s: len(s) if isinstance(s, str) else 0
    ).astype('Int64')

    dfc["user_location"] = u_series.apply(lambda u: safe_get(u, "location"))
    dfc["user_location"] = dfc["user_location"].astype("string")

    dfc["user_has_location"] = dfc["user_location"].notna() & (
        dfc["user_location"].str.strip() != ""
    )
    dfc["user_has_location"] = dfc["user_has_location"].astype("boolean")

    # Profile image / defaults
    dfc["user_default_profile_image"] = u_series.apply(
        lambda u: safe_get(u, "default_profile_image", False)
    ).astype("boolean")

    # Account creation time
    dfc["user_created_at"] = pd.to_datetime(
        u_series.apply(lambda u: safe_get(u, "created_at")),
        errors="coerce",
        utc=True,
    )

    # =========================================================
    # 4. QUOTED_STATUS FEATURES
    # =========================================================

    qs = dfc["quoted_status"]

    dfc["has_quoted_status"] = qs.notna()
    dfc["has_quoted_status"] = dfc["has_quoted_status"].astype("boolean")

    # Language of quoted tweet
    dfc["quoted_lang"] = qs.apply(lambda q: safe_get(q, "lang"))
    dfc["quoted_lang"] = dfc["quoted_lang"].astype("category")

    # Engagement on quoted tweet
    dfc["quoted_retweet_count"] = qs.apply(
        lambda q: safe_get(q, "retweet_count")
    ).astype("Int64")
    dfc["quoted_favorite_count"] = qs.apply(
        lambda q: safe_get(q, "favorite_count")
    ).astype("Int64")
    dfc["quoted_reply_count"] = qs.apply(
        lambda q: safe_get(q, "reply_count")
    ).astype("Int64")
    dfc["quoted_quote_count"] = qs.apply(
        lambda q: safe_get(q, "quote_count")
    ).astype("Int64")

    # Quoted user
    dfc["quoted_user_followers_count"] = qs.apply(
        lambda q: safe_get(safe_get(q, "user", {}), "followers_count")
    ).astype("Int64")

    dfc["quoted_user_friends_count"] = qs.apply(
        lambda q: safe_get(safe_get(q, "user", {}), "friends_count")
    ).astype("Int64")

    dfc["quoted_user_verified"] = qs.apply(
        lambda q: safe_get(safe_get(q, "user", {}), "verified", False)
    ).astype("boolean")

    if "quoted_user_verified" in dfc.columns and "has_quoted_status" in dfc.columns:
        dfc.loc[~dfc["has_quoted_status"], "quoted_user_verified"] = pd.NA
        dfc["quoted_user_verified"] = dfc["quoted_user_verified"].astype("boolean")

    # =========================================================
    # 5. FULL TEXT & ENTITIES
    # =========================================================

    # ---- 5.1 FULL TEXT FROM extended_tweet ----
    def get_full_text(row):
        ext = row.get("extended_tweet")
        if row.get("truncated") and isinstance(ext, dict) and "full_text" in ext:
            return ext["full_text"]
        return row["text"]

    dfc["text"] = dfc.apply(get_full_text, axis=1).astype("string")

    # ---- 5.2 UNIFIED ENTITIES ----
    def get_effective_entities(row):
        ext = row.get("extended_tweet")
        if row.get("truncated") and isinstance(ext, dict) and "entities" in ext:
            return ext["entities"]
        return row.get("entities")

    dfc["effective_entities"] = dfc.apply(get_effective_entities, axis=1)

    ent = dfc["effective_entities"]

    dfc["n_hashtags"] = ent.apply(
        lambda e: safe_len(safe_get(e, "hashtags", []))
    ).astype("Int64")
    dfc["n_mentions"] = ent.apply(
        lambda e: safe_len(safe_get(e, "user_mentions", []))
    ).astype("Int64")
    dfc["n_urls"] = ent.apply(
        lambda e: safe_len(safe_get(e, "urls", []))
    ).astype("Int64")

    dfc["has_hashtag"] = (dfc["n_hashtags"] > 0).astype("boolean")
    dfc["has_mention"] = (dfc["n_mentions"] > 0).astype("boolean")
    dfc["has_url"] = (dfc["n_urls"] > 0).astype("boolean")

    emoji_pattern = re.compile(
        "["                       # start character class
        "\U0001F300-\U0001F5FF"   # symbols & pictographs
        "\U0001F600-\U0001F64F"   # emoticons
        "\U0001F680-\U0001F6FF"   # transport & map
        "\U0001F700-\U0001F77F"   # alchemical symbols
        "\U0001F780-\U0001F7FF"   # geometric symbols
        "\U0001F800-\U0001F8FF"   # supplemental arrows
        "\U0001F900-\U0001F9FF"   # supplemental symbols & pictographs
        "\U0001FA00-\U0001FAFF"   # chess, symbols, etc.
        "\U00002600-\U000026FF"   # misc symbols
        "\U00002700-\U000027BF"   # dingbats
        "]+",
        flags=re.UNICODE
    )

    def count_emojis(text: str) -> int:
        # This counts each *single emoji codepoint* matched
        matches = emoji_pattern.findall(text)
        return sum(len(m) for m in matches)
    
    dfc["n_emojis"] = dfc["text"].apply(count_emojis).astype("Int64")

    # =========================================================
    # 6. MEDIA FEATURES (extended_entities)
    # =========================================================

    ee = dfc["extended_entities"]

    def media_stats(ee_row):
        if not isinstance(ee_row, dict):
            return 0, 0, 0, 0
        media = ee_row.get("media", [])
        if not isinstance(media, list):
            return 0, 0, 0, 0

        n = len(media)
        n_photo = sum(
            1 for m in media if isinstance(m, dict) and m.get("type") == "photo"
        )
        n_video = sum(
            1 for m in media if isinstance(m, dict) and m.get("type") == "video"
        )
        n_gif = sum(
            1 for m in media if isinstance(m, dict) and m.get("type") == "animated_gif"
        )
        return n, n_photo, n_video, n_gif

    media_stats_array = ee.apply(media_stats)

    dfc["n_media"] = media_stats_array.apply(lambda t: t[0]).astype("Int64")
    dfc["n_photo"] = media_stats_array.apply(lambda t: t[1]).astype("Int64")
    dfc["n_video"] = media_stats_array.apply(lambda t: t[2]).astype("Int64")
    dfc["n_gif"] = media_stats_array.apply(lambda t: t[3]).astype("Int64")

    dfc["has_media"] = (dfc["n_media"] > 0).astype("boolean")

    # =========================================================
    # 7. PLACE FEATURES
    # =========================================================

    pl = dfc["place"]

    dfc["has_place"] = pl.notna()
    dfc["has_place"] = dfc["has_place"].astype("boolean")

    dfc["place_country"] = pl.apply(lambda p: safe_get(p, "country"))
    dfc["place_country"] = dfc["place_country"].astype("category")

    dfc["place_country_code"] = pl.apply(lambda p: safe_get(p, "country_code"))
    dfc["place_country_code"] = dfc["place_country_code"].astype("category")

    dfc["place_type"] = pl.apply(lambda p: safe_get(p, "place_type"))
    dfc["place_type"] = dfc["place_type"].astype("category")

    dfc["place_full_name"] = pl.apply(lambda p: safe_get(p, "full_name"))
    dfc["place_full_name"] = dfc["place_full_name"].astype("string")

    # =========================================================
    # 8. DROP HEAVY JSON COLUMNS
    # =========================================================

    cols_to_drop = [
        "user",
        "quoted_status",
        "entities",
        "extended_tweet",
        "extended_entities",
        "place",
        "effective_entities",  # optional
    ]
    cols_to_drop = [c for c in cols_to_drop if c in dfc.columns]

    dfc = dfc.drop(columns=cols_to_drop)

    # =========================================================
    # 9. LOG TRANSFORMS & TIME FEATURES
    # =========================================================

    log_cols = [
        # "user_followers_count",
        # "user_friends_count",
        "user_statuses_count",
        "user_favourites_count",
        "user_listed_count",
    ]

    for col in log_cols:
        if col in dfc.columns:
            dfc[f"log_{col}"] = np.log1p(dfc[col].astype("float"))

    # Ensure created_at is datetime with tz
    dfc["created_at"] = pd.to_datetime(dfc["created_at"], utc=True, errors="coerce")

    dfc["tweet_hour"] = dfc["created_at"].dt.hour.astype("Int64")
    dfc["tweet_dow"] = dfc["created_at"].dt.dayofweek.astype("Int64")  # 0=Mon, 6=Sun

    dfc["is_weekend"] = dfc["tweet_dow"].isin([5, 6]).astype("boolean")

    # user_created_at should already be datetime (from step 3)
    dfc["account_age_days"] = (
        (dfc["created_at"] - dfc["user_created_at"]).dt.days.astype("Int64")
    )

    # Is it a reply?
    dfc["is_reply"] = dfc["in_reply_to_screen_name"].notna().astype("boolean")

    # Is it quoting another tweet?
    dfc["is_quote"] = dfc["has_quoted_status"].astype("boolean")

    # Standalone original tweet (neither reply nor quote)
    dfc["is_standalone"] = (~dfc["is_reply"] & ~dfc["is_quote"]).astype("boolean")

    # =========================================================
    # 10. ENGAGEMENT & TEXT STATISTICS
    # =========================================================

    eng_cols = ["retweet_count", "favorite_count", "reply_count", "quote_count"]

    dfc["tweet_total_engagement"] = (
        dfc["retweet_count"].fillna(0).astype("Int64")
        + dfc["favorite_count"].fillna(0).astype("Int64")
        + dfc["reply_count"].fillna(0).astype("Int64")
        + dfc["quote_count"].fillna(0).astype("Int64")
    ).astype("Int64")

    # Characters and words
    dfc["text_length_chars"] = dfc["text"].astype("string").str.len().astype("Int64")

    dfc["text_length_words"] = dfc["text"].apply(
        lambda s: len(str(s).split()) if pd.notnull(s) else 0
    ).astype("Int64")

    # Punctuation counts
    dfc["n_exclamation"] = dfc["text"].astype("string").str.count("!").astype("Int64")
    dfc["n_question"] = dfc["text"].astype("string").str.count(r"\?").astype("Int64")

    def caps_share(s):
        if not isinstance(s, str):
            return np.nan
        letters = [ch for ch in s if s and ch.isalpha()]
        if not letters:
            return 0.0
        caps = sum(ch.isupper() for ch in letters)
        return caps / len(letters)

    dfc["share_caps"] = dfc["text"].apply(caps_share)

    words = dfc["text_length_words"].replace(0, np.nan).astype("float")

    dfc["hashtags_per_word"] = (
        dfc["n_hashtags"].astype("float") / words
    ).fillna(0.0)
    dfc["mentions_per_word"] = (
        dfc["n_mentions"].astype("float") / words
    ).fillna(0.0)
    dfc["urls_per_word"] = (dfc["n_urls"].astype("float") / words).fillna(0.0)
    dfc["media_per_word"] = (dfc["n_media"].astype("float") / words).fillna(0.0)

    q_followers = dfc["quoted_user_followers_count"].astype("float")
    q_friends = dfc["quoted_user_friends_count"].astype("float")

    dfc["quoted_user_followers_ratio"] = q_followers / (q_friends.replace(0, np.nan))
    dfc["quoted_user_followers_ratio"] = dfc["quoted_user_followers_ratio"].fillna(0.0)

    # =========================================================
    # 11. SOURCE HANDLING
    # =========================================================
    source_features = dfc['source'].apply(parse_source_raw).apply(pd.Series)
    dfc = pd.concat([dfc, source_features], axis=1)

    # =========================================================
    # 12. DROP CONSTANT COLUMNS
    # =========================================================

    nunique = {}
    for col in dfc.columns:
        try:
            nunique[col] = dfc[col].nunique(dropna=False)
        except TypeError:  # e.g. dicts
            nunique[col] = np.nan

    nunique = pd.Series(nunique).sort_values()

    # Drop constant columns except those required by baseline
    constant_cols = [c for c in nunique[nunique == 1].index.tolist() if c not in REQUIRED_FEATURES]
    dfc.drop(columns=constant_cols, inplace=True)

    # =========================================================
    # 13. 
    # =========================================================

    # Guarantee presence and types for baseline features
    for col in NUMERIC_FEATURES_BASELINE:
        if col not in dfc.columns:
            dfc[col] = 0
        dfc[col] = pd.to_numeric(dfc[col], errors="coerce").astype("float")

    for col in BOOL_FEATURES_BASELINE:
        if col not in dfc.columns:
            dfc[col] = False
        dfc[col] = dfc[col].astype("boolean")

    for col in CATEGORICAL_FEATURES_BASELINE:
        if col not in dfc.columns:
            dfc[col] = pd.NA
        dfc[col] = dfc[col].astype("category")

    if "text" not in dfc.columns:
        dfc["text"] = ""
    dfc["text"] = dfc["text"].astype("string")

    if "label" in dfc.columns:
        dfc["label"] = pd.to_numeric(dfc["label"], errors="coerce").astype("Int64")
    if "challenge_id" in dfc.columns:
        dfc["challenge_id"] = pd.to_numeric(dfc["challenge_id"], errors="coerce").astype("Int64")

    # Reorder to match baseline expectation and drop unused extras
    ordered = (
        ["challenge_id", "label", "text"]
        + NUMERIC_FEATURES_BASELINE
        + BOOL_FEATURES_BASELINE
        + CATEGORICAL_FEATURES_BASELINE
    )
    existing_ordered = [c for c in ordered if c in dfc.columns]
    dfc = dfc[existing_ordered]

    return dfc