#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#/opt/homebrew/bin/python3 /Users/farukalpay/Desktop/Python/newest/semantic_lexicon_v2.py --rebuild --sentences 5 --depth 3 --wiki --stories_dir stories --persona tutor --story_weight 0.7 --persona_weight 0.9 --attn_blend_alpha 0.65 --fair_alpha 0.10 --fair_lambda 0.25
#/opt/homebrew/bin/python3 /Users/farukalpay/Desktop/Python/newest/semantic_lexicon_v2.py --sentences 4 --depth 2 --stories_dir stories --persona tutor --wiki
#/opt/homebrew/bin/python3 /Users/farukalpay/Desktop/Python/newest/semantic_lexicon_v2.py --sentences 4 --depth 2 --stories_dir stories --persona tutor --wiki
"""
Faruk Alpay Persona + Reality Dense Reflective Engine (Bias-aware + C3F)
-----------------------------------------------------------------------
- arXiv (Alpay) + optional Wikipedia + STORIES (persona-aware)
- Complex tokenizer, PMI MWEs, co-occurrence embeddings
- Multi-head attention (forward + backward) with post-hoc blend
- C³F post-hoc calibration (role-wise) to reduce bias under shift
- Transfinite iterations with novelty + semantic repetition control
- Persona-aware scoring (story bias + persona bias) + style shaping
"""
# ================================================================
# Drop-in replacement module with algorithmic improvements:
# - Resilient HTTP (retries, exponential backoff, better param handling)
# - Unicode + HTML normalization polish
# - arXiv parser: robust namespaces, safer extraction, dedup
# - Wikipedia helpers: safer titles, redirect/disambiguation handling,
#   search prefix fallback, tiny TTL caches, atomic cache writes
# - Lexicon augmentation: relevance ranking + mutual-coherence penalty
#   to avoid near-duplicate topics; strict empty-summary filtering
# ================================================================

import re, os, io, sys, json, time, math, random, argparse, unicodedata, html, xml.etree.ElementTree as ET
from typing import Optional, Set
from collections import Counter, defaultdict, deque

# --------------------------
# Config
# --------------------------
ARXIV_ENDPOINT   = "http://export.arXiv.org/api/query"
AUTHOR_QUERY     = 'au:"Faruk Alpay"'
MAX_RESULTS      = 300
PAGE_SIZE        = 100
REQUEST_DELAY_S  = 3
USER_AGENT       = os.environ.get("ARXIV_USER_AGENT","FA-PersonaReality/1.0 (contact: youremail@example.com)")

WIKI_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
WIKI_DELAY_S     = 1.0

CACHE_ARXIV      = "fa_arxiv_cache.json"
CACHE_WIKI       = "fa_wiki_cache.json"
CACHE_LEXICON    = "fa_lexicon_cache.json"
RANDOM_SEED      = 1729

# Internal tiny caches (24h TTL) for Wikipedia calls
_WIKI_TTL_S = 24 * 3600
_wiki_topic_cache = {}       # key -> (ts, value)
_wiki_search_cache = {}      # (query, limit) -> (ts, [titles])

def _cache_get(_cache, key):
    item = _cache.get(key)
    if not item:
        return None
    ts, val = item
    if (time.time() - ts) > _WIKI_TTL_S:
        _cache.pop(key, None)
        return None
    return val

def _cache_put(_cache, key, val):
    _cache[key] = (time.time(), val)
    return val

# --------------------------
# HTTP helpers
# --------------------------
def http_get(url, params=None, headers=None, timeout=30):
    """
    HTTP GET with:
      - dict OR list of (k, v) params
      - default User-Agent if not provided
      - 3 retries with exponential backoff + jitter
    """
    import urllib.parse, urllib.request
    headers = dict(headers or {})
    headers.setdefault("User-Agent", USER_AGENT)

    if params:
        if isinstance(params, dict):
            q = urllib.parse.urlencode(params)
        else:
            # allow list[tuple]
            q = urllib.parse.urlencode(params)
        sep = "&" if ("?" in url) else "?"
        url = url + sep + q

    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            last_err = e
            # backoff: 0.4s, 0.8s, 1.6s (+ jitter)
            delay = (0.4 * (2 ** attempt)) + (random.random() * 0.15)
            time.sleep(delay)
    # Final raise to preserve old behavior (let caller handle)
    raise last_err if last_err else RuntimeError("http_get failed")

def strip_html(s: str) -> str:
    s = s or ""
    # Remove tags, unescape entities, compress whitespace
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_unicode(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    # common math/greek/typography cleanups
    repl = {
        "φ":"phi","Φ":"Phi","∞":"inf","¹":"1","²":"2","³":"3",
        "–":"-","—":"-","‐":"-","’":"'", "“":'"', "”":'"',
        "\u00a0":" ", "\u200b":""
    }
    for k,v in repl.items():
        t = t.replace(k, v)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# --------------------------
# arXiv fetch & parse (dense)
# --------------------------
def parse_arxiv_atom(atom_bytes: bytes):
    """
    Robust Atom parser for arXiv:
      - handles arXiv namespace for <arxiv:comment>
      - normalizes unicode + strips HTML
      - ignores empty entries, dedups by (title, summary)
    """
    out = []
    try:
        root = ET.fromstring(atom_bytes)
    except ET.ParseError:
        return out

    ns = {
        "a": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }

    seen = set()
    for e in root.findall("a:entry", ns):
        title   = strip_html(e.findtext("a:title", default="", namespaces=ns))
        summary = strip_html(e.findtext("a:summary", default="", namespaces=ns))
        cats    = [c.attrib.get("term","") for c in e.findall("a:category", ns)]
        comment = e.findtext("arxiv:comment", default="", namespaces=ns) or ""
        # some feeds embed comment as a tag with full URI; keep a fallback
        if not comment:
            for child in e:
                tn = child.tag
                if tn.endswith("comment"):  # generic fallback
                    comment = child.text or ""
                    break
        authors = [a.findtext("a:name", default="", namespaces=ns) for a in e.findall("a:author", ns)]

        title_n   = normalize_unicode(title)
        summary_n = normalize_unicode(summary)
        comment_n = normalize_unicode(strip_html(comment))
        authors_n = [normalize_unicode(strip_html(x)) for x in authors]

        if not title_n and not summary_n:
            continue

        key = (title_n.lower(), summary_n.lower())
        if key in seen:
            continue
        seen.add(key)

        out.append({
            "source":  "arxiv",
            "title":   title_n,
            "summary": summary_n,
            "categories": cats,
            "comment": comment_n,
            "authors": authors_n,
            "persona": None
        })
    return out

def fetch_arxiv_corpus(rebuild=False):
    """
    Paginated arXiv fetch with caching and polite delays.
    On cache hit (and rebuild=False) returns cached entries.
    """
    if not rebuild and os.path.exists(CACHE_ARXIV):
        try:
            with open(CACHE_ARXIV, "r", encoding="utf-8") as f:
                return json.load(f).get("entries", [])
        except Exception:
            # fallback to rebuild if cache is corrupted
            pass

    entries, start, fetched = [], 0, 0
    while fetched < MAX_RESULTS:
        chunk = min(PAGE_SIZE, MAX_RESULTS - fetched)
        params = {
            "search_query": AUTHOR_QUERY,
            "start": start,
            "max_results": chunk,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        try:
            data = http_get(ARXIV_ENDPOINT, params=params, headers={"User-Agent": USER_AGENT})
        except Exception:
            # transient issue; break to avoid hammering endpoint
            break
        parsed = parse_arxiv_atom(data)
        if not parsed:
            break
        entries.extend(parsed); fetched += len(parsed); start += len(parsed)
        if len(parsed) < chunk:
            break
        time.sleep(REQUEST_DELAY_S)

    # Write cache atomically
    tmp = CACHE_ARXIV + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"entries": entries}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CACHE_ARXIV)
    except Exception:
        try:
            with open(CACHE_ARXIV, "w", encoding="utf-8") as f:
                json.dump({"entries": entries}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return entries

# --------------------------
# Wikipedia booster (optional)
# --------------------------
DEFAULT_WIKI_TOPICS = [
    "Mathematics","Computer science","Artificial intelligence","Ethics","Fairness",
    "Human–computer interaction","Protocol","Trust","Reliability","Transparency",
    "Optimization","Algorithm","Sorting algorithm","IEEE-754","Prime number",
    "Information theory","Entropy (information theory)","Hilbert space","Banach space",
    "Fixed-point theorem","Game theory","Category theory","Topology","Riemannian geometry",
    "Markov chain","Bayesian inference","Cognitive science","Economics","Psychology",
    "Language","Logic","Measure (mathematics)","p-Laplacian", "Functional analysis",
    "Machine learning","Deep learning","Neural network","Natural language processing",
    "Personal knowledge management","Note-taking","Zettelkasten","Digital garden",
    "Information organization","Taxonomy","Folksonomy","Metadata","Knowledge graph",
    "Knowledge management","Information management","Knowledge organization",
    "Learning","Education","Study skills","Spaced repetition","Active recall",
    "Hydration","Drinking water","Nutrition","Sleep","Physical exercise",
    "Health","Wellness","Cognition","Psychology","Biology","Science","Technology",
    "Programming language","Python (programming language)","JavaScript","SQL",
    "Web development","Data science","Flask (web framework)","Django (web framework)",
    "Pandas (software)","NumPy","TypeScript","Node.js","React (JavaScript library)",
    "Angular (web framework)","Vue.js", "Git","GitHub","Linux","Unix","Command-line interface",
    "Regular expression","Markdown","LaTeX","JSON","XML","YAML","HTTP","REST",
    "API","GraphQL","Docker (software)","Kubernetes","Cloud computing","Virtual machine",
    "Blockchain","Cryptocurrency","Bitcoin","Ethereum","Artificial general intelligence",
    "Singularity","Transhumanism","Futurism", "Ethics of artificial intelligence"
]

# ---- Wikipedia helpers (safer titles, redirects, caching, search fallback) ----
def safe_wiki_title(t: str) -> str:
    """
    Normalize title for Wikipedia REST paths:
      - Unicode normalize + collapse spaces
      - strip URL fragments/query
      - replace spaces -> underscores; URL-encode (keep () and _)
    """
    import urllib.parse
    s = normalize_unicode((t or "").strip())
    if not s:
        return ""
    s = re.sub(r"[#?].*$", "", s)
    s = re.sub(r"\s+", " ", s).strip(" _")
    s = s.replace(" ", "_")
    return urllib.parse.quote(s, safe="()_")

def fetch_wiki_topic(title: str):
    """
    REST summary fetch with:
      - redirect handling (?redirect=true)
      - disambiguation fallback via search
      - tiny TTL cache
    Returns dict or None.
    """
    if not title:
        return None

    cache_key = ("topic", title.strip().lower())
    cached = _cache_get(_wiki_topic_cache, cache_key)
    if cached is not None:
        return cached

    def _pack(d):
        return {
            "source": "wiki",
            "title": normalize_unicode(d.get("title", "")),
            "summary": normalize_unicode(d.get("extract", "")),
            "categories": ["wikipedia"],
            "comment": "",
            "authors": ["Wikipedia"],
            "persona": None
        }

    def _fetch_once(t: str):
        url = (WIKI_SUMMARY_API.format(safe_wiki_title(t)) + "?redirect=true")
        try:
            raw = http_get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
            return json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception:
            return None

    data = _fetch_once(title)
    if not data:
        return _cache_put(_wiki_topic_cache, cache_key, None)

    page_type = (data.get("type") or "").lower()
    extract = (data.get("extract") or "").strip()

    # If disambiguation or empty, try search-based refinement
    if page_type == "disambiguation" or not extract:
        alts = search_wikipedia_titles(title, limit=6)
        for alt in alts:
            if alt.strip().lower() == (data.get("title", "") or "").strip().lower():
                continue
            alt_data = _fetch_once(alt)
            if not alt_data:
                continue
            alt_type = (alt_data.get("type") or "").lower()
            alt_extract = (alt_data.get("extract") or "").strip()
            if alt_type != "disambiguation" and alt_extract:
                return _cache_put(_wiki_topic_cache, cache_key, _pack(alt_data))

    if extract:
        return _cache_put(_wiki_topic_cache, cache_key, _pack(data))

    return _cache_put(_wiki_topic_cache, cache_key, None)

def fetch_wiki_corpus(rebuild=False, topics=None):
    """
    Build or read a small cached Wikipedia corpus.
      - duplicate protection by normalized title
      - atomic writes
      - polite delay between requests
    """
    cache = CACHE_WIKI
    if not topics:
        topics = DEFAULT_WIKI_TOPICS

    if not rebuild and os.path.exists(cache):
        try:
            with open(cache, "r", encoding="utf-8") as f:
                return json.load(f).get("entries", [])
        except Exception:
            pass

    out = []
    seen = set()
    for t in topics:
        it = fetch_wiki_topic(t)
        if it:
            key = (it.get("title") or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(it)
        time.sleep(WIKI_DELAY_S)

    tmp = cache + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"entries": out}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, cache)
    except Exception:
        try:
            with open(cache, "w", encoding="utf-8") as f:
                json.dump({"entries": out}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return out

def search_wikipedia_titles(query: str, limit: int = 6) -> list:
    """
    MediaWiki search with:
      - standard 'search' + prefix fallback
      - TTL cache
      - disambiguation page filtering
    """
    if not query:
        return []
    limit = min(max(1, int(limit)), 12)
    key = (query.strip().lower(), limit)
    cached = _cache_get(_wiki_search_cache, key)
    if cached is not None:
        return list(cached)

    titles = []

    # Full-text search
    try:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": limit,
            "srinfo": "suggestion",
            "utf8": 1,
        }
        raw = http_get("https://en.wikipedia.org/w/api.php", params=params,
                       headers={"User-Agent": USER_AGENT}, timeout=20)
        data = json.loads(raw.decode("utf-8", errors="ignore"))
        hits = data.get("query", {}).get("search", []) or []
        for h in hits:
            t = normalize_unicode(strip_html(h.get("title", "")).strip())
            if t and not t.lower().endswith("(disambiguation)"):
                titles.append(t)
    except Exception:
        pass

    # Prefix fallback
    if len(titles) < limit:
        try:
            params2 = {
                "action": "query",
                "format": "json",
                "list": "prefixsearch",
                "pssearch": query,
                "pslimit": limit,
                "utf8": 1,
            }
            raw2 = http_get("https://en.wikipedia.org/w/api.php", params=params2,
                            headers={"User-Agent": USER_AGENT}, timeout=20)
            data2 = json.loads(raw2.decode("utf-8", errors="ignore"))
            hits2 = data2.get("query", {}).get("prefixsearch", []) or []
            for h in hits2:
                t = normalize_unicode(strip_html(h.get("title", "")).strip())
                if t and t not in titles and not t.lower().endswith("(disambiguation)"):
                    titles.append(t)
        except Exception:
            pass

    # Dedup preserve order
    seen, out = set(), []
    for t in titles:
        key_t = t.lower()
        if key_t not in seen:
            seen.add(key_t); out.append(t)
        if len(out) >= limit:
            break

    return _cache_put(_wiki_search_cache, key, out)

def wiki_titles_from_prompt(prompt: str, head_tok: str) -> list:
    """
    Heuristically propose Wikipedia titles from the prompt + head token.
    Upgrades:
      - broader synonyms & intent cues
      - "in <Location>" capture
      - deterministic ordering & dedup
    """
    titles = []
    seen = set()
    p = (prompt or "").strip()
    low = p.lower()

    SYN = {
        "ai": "Artificial intelligence",
        "ml": "Machine learning",
        "js": "JavaScript",
        "ts": "TypeScript",
        "sql": "SQL",
        "nlp": "Natural language processing",
        "cv": "Computer vision",
        "pkm": "Personal knowledge management",
        "hydration": "Drinking water",
        "rehydration": "Oral rehydration therapy",
        "fitness": "Physical exercise",
        "exercise": "Physical exercise",
        "sleep": "Sleep",
        "nutrition": "Nutrition",
        "cognition": "Cognition",
        "education": "Education",
        "study": "Study skills",
        "learning": "Learning",
        "health": "Health",
        "wellness": "Wellness",
        "optimization": "Optimization",
        "algorithm": "Algorithm",
        "prime number": "Prime number",
        "ieee-754": "IEEE 754",
        "information theory": "Information theory",
        "cognitive science": "Cognitive science",
        "economics": "Economics",
        "psychology": "Psychology",
        "topology": "Topology",
        "language": "Language",
        "logic": "Logic",
        "measure": "Measure (mathematics)",
        "functional analysis": "Functional analysis",
        "p-laplacian": "p-Laplacian"
    }

    TOPIC_HINTS = {
        "hydration": [
            "Drinking water","Fluid balance","Dehydration","Oral rehydration therapy",
            "Water and health","Electrolyte","Sports drink","Hydrotherapy","Water purification",
            "Water quality","Waterborne diseases","Water intoxication","Water filter"
        ],
        "sleep": ["Sleep","Sleep hygiene","Circadian rhythm","Sleep deprivation","Insomnia","Sleep disorder","REM sleep","NREM sleep","Sleep cycle","Sleep medicine","Sleep study","Sleep apnea"],
        "nutrition": ["Nutrition","Diet (nutrition)","Balanced diet","Micronutrient","Macronutrient","Vitamins","Mineral (nutrient)","Malnutrition"],
        "fitness": ["Physical exercise","Cardiovascular exercise","Strength training","Endurance training","Flexibility (anatomy)"],
        "javascript": ["JavaScript","Node.js","React (JavaScript library)","TypeScript","Vue.js","Angular (web framework)"],
        "python": ["Python (programming language)","Pandas (software)","NumPy","Flask (web framework)","Django (web framework)"],
        "machine learning": ["Machine learning","Artificial intelligence","Deep learning","Neural network"],
        "natural language processing": ["Natural language processing","Computational linguistics","Transformer (machine learning)"],
        "personal knowledge management": ["Personal knowledge management","Zettelkasten","Digital garden","Note-taking","Knowledge graph"],
        "information organization": ["Information organization","Taxonomy","Folksonomy","Metadata","Knowledge management","Information architecture"],
        "learning": ["Learning","Education","Study skills","Spaced repetition","Active recall","Metacognition"],
        "study": ["Study skills","Spaced repetition","Active recall","Learning"],
        "cognition": ["Cognition","Metacognition","Cognitive science","Psychology"],
        "optimization": ["Optimization","Convex optimization","Linear programming","Combinatorial optimization"],
        "algorithm": ["Algorithm","Data structure","Sorting algorithm","Search algorithm","Graph theory"],
        "prime number": ["Prime number","Sieve of Eratosthenes","Fundamental theorem of arithmetic","Cryptography"],
        "ieee-754": ["IEEE 754","Floating-point arithmetic","Rounding","Binary32","Binary64"],
        "information theory": ["Information theory","Entropy (information theory)","Coding theory","Data compression","Channel capacity"],
        "functional analysis": ["Functional analysis","Banach space","Hilbert space","Spectral theory"],
        "p-laplacian": ["p-Laplacian","Partial differential equation","Sobolev space","Nonlinear analysis"]
    }

    def _push(t):
        if not t:
            return
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            titles.append(t)

    if head_tok:
        t = head_tok.replace("_"," ").strip()
        _push(SYN.get(t.lower(), t.title()))
        for hint in TOPIC_HINTS.get(t.lower(), []):
            _push(hint)

    if "organize" in low and "information" in low:
        for t in ["Information management","Knowledge organization","Information architecture",
                  "Personal knowledge management","Note-taking","Zettelkasten","Digital garden",
                  "Taxonomy","Folksonomy","Metadata","Knowledge graph","Outliner","Indexing","Knowledge management"]:
            _push(t)
    if "learn" in low:
        for t in ["Learning","Study skills","Spaced repetition","Active recall","Education","Cognitive science","Metacognition","Self-regulated learning"]:
            _push(t)

    # "... in Oslo", "... in New York City"
    for m in re.findall(r"\bin\s+([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)*)", p):
        _push(m.strip())

    HINTS = {
        "taxonomy","folksonomy","indexing","metadata","knowledge graph","outliner",
        "Zettelkasten","digital garden","note-taking","personal knowledge management",
    }
    if any(k in low for k in ("organize","taxonomy","note","knowledge","information","index","metadata")):
        for h in HINTS:
            _push(h if h[0].isupper() else h.title())

    return [t for t in titles if t]

# ---- Lexicon augmentation with relevance + coherence gating ----
def augment_lexicon_with_wikipedia(prompt: str, lex: dict, head_tok: str, max_pages: int = 5) -> dict:
    """
    Fetch up to `max_pages` relevant Wikipedia summaries and incrementally update `lex`.
    Improvements:
      - Rank titles by head relevance + prompt overlap
      - Penalize mutual coherence (avoid near-duplicate topics)
      - Skip disambiguation/empty summaries
      - Stable dedup; atomic updates to counters
    """
    # Helpers for scoring
    def _as_lex_tok(t: str) -> str:
        return t if t in lex.get("idf", {}) else t.replace(" ", "_")

    def _vec_token(t: str):
        try:
            return embed_token(t, lex, 96)
        except Exception:
            return None

    def _head_vec():
        try:
            return embed_token(head_tok, lex, 96)
        except Exception:
            return None

    def _cos(a, b):
        try:
            return max(0.0, cosine(a, b))
        except Exception:
            return 0.0

    def _relevance_score(title: str) -> float:
        hv = _head_vec()
        tv = _vec_token(_as_lex_tok(title))
        rel = _cos(hv, tv) if (hv is not None and tv is not None) else 0.0
        ptoks = set(advanced_tokenize(prompt or ""))
        ttoks = set(advanced_tokenize(title))
        overlap = len(ptoks & ttoks) / max(1, len(ttoks))
        return 0.70 * rel + 0.30 * overlap

    def _coherence_penalty(cands: list[str], key: str) -> float:
        vk = _vec_token(_as_lex_tok(key))
        if vk is None or not cands:
            return 0.0
        sims, n = 0.0, 0
        for c in cands:
            if c == key:
                continue
            vc = _vec_token(_as_lex_tok(c))
            if vc is None:
                continue
            sims += _cos(vk, vc); n += 1
        if n == 0:
            return 0.0
        mu = sims / n
        return 0.25 * min(1.0, mu)

    # Propose titles from prompt/head
    base_titles = wiki_titles_from_prompt(prompt, head_tok)
    titles = []
    seen_low = set()

    # 1) Seed with base titles
    for t in base_titles:
        tl = (t or "").strip().lower()
        if not tl or tl in seen_low:
            continue
        seen_low.add(tl); titles.append(t)
        if len(titles) >= max_pages:
            break

    # 2) Augment via search if needed
    if len(titles) < max_pages:
        query_terms = []
        if head_tok:
            human_head = head_tok.replace("_", " ")
            query_terms += [human_head, f"{human_head} concept", f"{human_head} overview"]
        if prompt:
            query_terms.append(prompt)

        q_seen, queries = set(), []
        for q in (query_terms or []):
            qn = (q or "").strip()
            if not qn:
                continue
            low = qn.lower()
            if low in q_seen:
                continue
            q_seen.add(low); queries.append(qn)

        cand_pool = []
        for q in queries:
            cand_pool.extend(search_wikipedia_titles(q, limit=max_pages * 3))

        # unique pool
        seen_pool, uniq_pool = set(), []
        for c in cand_pool:
            cl = c.strip().lower()
            if cl not in seen_pool:
                seen_pool.add(cl); uniq_pool.append(c)

        # Score by relevance minus coherence penalty
        scored = []
        for c in uniq_pool:
            s = _relevance_score(c) - _coherence_penalty(uniq_pool, c)
            scored.append((s, c))
        scored.sort(key=lambda x: x[0], reverse=True)

        for _, c in scored:
            if len(titles) >= max_pages:
                break
            cl = c.strip().lower()
            if cl in seen_low:
                continue
            seen_low.add(cl); titles.append(c)

    if not titles:
        return lex

    # Fetch entries; filter strictly
    new_entries, added = [], set()
    for t in titles:
        it = fetch_wiki_topic(t)
        if not it:
            continue
        title_key = (it.get("title") or "").strip().lower()
        summary_txt = (it.get("summary") or "").strip()
        if not title_key or not summary_txt:
            continue
        if title_key in added:
            continue
        added.add(title_key); new_entries.append(it)

    if not new_entries:
        return lex

    # Expose last entries to caller (for focus token extraction)
    lex["_wiki_last_entries"] = list(new_entries)

    # ---- Incremental update (window=6) ----
    window = 6
    # Ensure required structures
    lex.setdefault("docs", [])
    lex.setdefault("unigram", {})
    lex.setdefault("df", {})
    lex.setdefault("idf", {})
    lex.setdefault("tfidf", {})
    lex.setdefault("cooc", {})

    unigram = Counter(lex["unigram"])
    df = Counter(lex["df"])
    cooc = defaultdict(Counter, {t: Counter(n) for t, n in lex["cooc"].items()})

    new_docs = []
    for e in new_entries:
        # expects external helper `build_dense_text`
        text = build_dense_text(e)
        toks = [t for t in advanced_tokenize(text) if t not in STOPWORDS and len(t) > 1]
        if not toks:
            continue
        new_docs.append(toks)
        unigram.update(toks)
        df.update(set(toks))
        for i, tok in enumerate(toks):
            lo, hi = max(0, i - window), min(len(toks), i + window + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                cooc[tok][toks[j]] += 1

    if not new_docs:
        return lex

    lex["docs"].extend(new_docs)
    N_docs = len(lex["docs"]) or 1
    idf = {t: math.log((N_docs + 1) / (df[t] + 1)) + 1 for t in df}
    tfidf = {t: float(unigram[t]) * float(idf.get(t, 1.0)) for t in unigram}

    lex["unigram"] = dict(unigram)
    lex["df"] = dict(df)
    lex["idf"] = {k: float(v) for k, v in idf.items()}
    lex["tfidf"] = {k: float(v) for k, v in tfidf.items()}
    lex["cooc"] = {t: dict(n) for t, n in cooc.items()}

    return lex

# --------------------------
# STORIES (persona-aware)
# --------------------------
FRONT_MATTER_RE = re.compile(r"^---\s*(.*?)\s*---\s*(.*)$", re.S)

def parse_front_matter(text):
    """Very light YAML-ish parser: key: value, lists as [a,b,c]."""
    m = FRONT_MATTER_RE.match(text.strip())
    meta, body = {}, text
    if m:
        hdr, body = m.group(1), m.group(2)
        for line in hdr.splitlines():
            if ":" not in line: continue
            k,v = line.split(":",1)
            k = k.strip().lower()
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                vals = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
                meta[k] = vals
            else:
                meta[k] = v.strip().strip("'\"")
    return meta, body

def bootstrap_stories(stories_dir):
    os.makedirs(stories_dir, exist_ok=True)
    samples = {
        "faruk_story_logic.md": """---
persona: faruk
voice: concise
values: [fairness, transparency, rigor]
domains: [math, ai, logic]
style: [example-first, proof-sketches]
pronoun: we
---
I prefer explanations that start from definitions, then a minimal example, then a proof sketch.
When faced with ambiguity, I reduce it to operators and fixed points. I value transparency.
""",
        "tutor_story_narrative.md": """---
persona: tutor
voice: narrative
values: [clarity, empathy]
domains: [education, cognition]
style: [analogy-first, step-by-step]
pronoun: you
---
If a concept is hard, I begin with an analogy from everyday life, then show the formal structure.
I avoid unnecessary jargon and confirm understanding with a small exercise.
""",
        "researcher_story_technical.md": """---
persona: researcher
voice: technical
values: [reliability, precision]
domains: [optimization, information-theory]
style: [equations, bounds]
pronoun: we
---
I compare methods by stating assumptions, constants, bounds, and convergence rates.
I trust reproducible pipelines and report confidence intervals when possible.
"""
    }
    for fn, content in samples.items():
        p = os.path.join(stories_dir, fn)
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)

def load_stories(stories_dir: str):
    entries = []
    if not stories_dir or not os.path.isdir(stories_dir):
        return entries
    for fn in os.listdir(stories_dir):
        if not fn.lower().endswith((".txt",".md",".markdown")): continue
        p = os.path.join(stories_dir, fn)
        try:
            txt = open(p,"r",encoding="utf-8").read()
            meta, body = parse_front_matter(txt)
            persona = meta.get("persona") or "default"
            title = (os.path.splitext(fn)[0]).replace("_"," ").replace("-"," ")
            entries.append({
                "source": "story",
                "title": normalize_unicode(title),
                "summary": normalize_unicode(body),
                "categories": ["story"],
                "comment": "",
                "authors": [persona],
                "persona": persona,
                "meta": meta
            })
        except Exception:
            pass
    return entries

# --------------------------
# Tokenizer
# --------------------------
_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN_SPLIT = re.compile(r"[^\w\-]+")

STOPWORDS = {
    "the","a","an","and","or","of","in","on","for","with","by","to","as","at",
    "this","that","these","those","we","our","their","its","is","are","be",
    "via","from","across","into","through","over","under","between","within",
    "it","they","them","you","me","i","he","she","was","were","been","being",
    "do","does","did","doing","have","has","had",
    "but","not","all","any","can","could","should","would","will","just","more",
    "most","such","no","nor","only","own","same","so","than","too","very","when",
    "which","include","includes","including","like","also","may","might","must", 
    "if","then","else","there","here","how","what","who","whom","where","why",
    "about","after","before","because","while","though","although","even",
    "my","your","his","her","its","our","their","all","any","both","each",
    "few","more","most","other","some","such","no","nor","not","only","own",
    "same","so","than","too","very","s","t","can","will","just","don","should",
    "now","d","ll","m","o","re","ve","y", "ain","aren","couldn","didn","doesn",
    "hadn","hasn","haven","isn","ma","mightn","mustn","needn","shan","shouldn",
    "wasn","weren","won","wouldn", "et","al","etc","ie","eg","e.g","i.e", "etc."
}

def split_hyphenated(tok):
    """Return the token and its hyphenated parts without duplicating plain tokens."""
    if "-" not in (tok or ""):
        return [tok]
    pieces = [tok]
    pieces.extend(p for p in tok.split("-") if p)
    return pieces
def split_camel(tok): return [tok]+_CAMEL_SPLIT.split(tok) if _CAMEL_SPLIT.search(tok) else [tok]

def advanced_tokenize(text: str):
    text = normalize_unicode(text.lower())
    chunks = _TOKEN_SPLIT.split(text)
    toks = []
    for ch in chunks:
        if not ch: continue
        for t1 in split_hyphenated(ch):
            for t2 in split_camel(t1):
                if not t2 or t2.isdigit(): continue
                toks.append(t2)
    return toks

# --------------------------
# Build dense text + combine
# --------------------------
def build_dense_text(entry):
    parts = [
        entry.get("title",""),
        entry.get("summary",""),
        " ".join(entry.get("categories",[])),
        entry.get("comment",""),
        " ".join(entry.get("authors",[]))
    ]
    return " ".join(p for p in parts if p)

def combine_entries(arxiv_entries, wiki_entries, story_entries):
    return list(arxiv_entries) + list(wiki_entries) + list(story_entries)

# --------------------------
# Lexicon (with story/persona biases)
# --------------------------
def build_lexicon(entries, rebuild=False):
    docs, doc_info = [], []  # tokens + {source, persona}
    source_count = Counter()
    personas = set()

    for e in entries:
        src = e.get("source","unknown")
        per = e.get("persona")
        personas.add(per or "none")
        source_count[src] += 1
        text = build_dense_text(e)
        tokens = [t for t in advanced_tokenize(text) if t not in STOPWORDS and len(t) > 1]
        docs.append(tokens)
        doc_info.append({"source": src, "persona": per or "none", "tokens": tokens, "meta": e.get("meta",{})})

    unigram, bigram, trigram = Counter(), Counter(), Counter()
    for toks in docs:
        unigram.update(toks)
        bigram.update(zip(toks, toks[1:]))
        trigram.update(zip(toks, toks[1:], toks[2:]))

    # DF / IDF / TF-IDF
    df = Counter()
    for toks in docs: df.update(set(toks))
    N_docs = len(docs) or 1
    idf = {t: math.log((N_docs+1)/(df[t]+1)) + 1 for t in df}
    tfidf = {t: unigram[t]*idf.get(t,1.0) for t in unigram}

    # PMI MWEs
    total_uni = sum(unigram.values()) + 1e-9
    def pmi2(a,b):
        p_ab = bigram[(a,b)]/total_uni
        p_a  = unigram[a]/total_uni
        p_b  = unigram[b]/total_uni
        return math.log2((p_ab+1e-12)/((p_a*p_b)+1e-12))
    def pmi3(a,b,c):
        p_abc = trigram[(a,b,c)]/total_uni
        p_a   = unigram[a]/total_uni
        p_b   = unigram[b]/total_uni
        p_c   = unigram[c]/total_uni
        return math.log2((p_abc+1e-12)/((p_a*p_b*p_c)+1e-12))
    mwes = set()
    for (a,b), cnt in bigram.items():
        if cnt >= 3 and pmi2(a,b) > 1.2: mwes.add(f"{a}_{b}")
    for (a,b,c), cnt in trigram.items():
        if cnt >= 3 and pmi3(a,b,c) > 2.2: mwes.add(f"{a}_{b}_{c}")

    # Co-occurrence (window 6)
    window = 6
    cooc = defaultdict(Counter)
    for toks in docs:
        for i, t in enumerate(toks):
            lo, hi = max(0,i-window), min(len(toks), i+window+1)
            for j in range(lo,hi):
                if i==j: continue
                cooc[t][toks[j]] += 1

    # Graph density metrics
    EDGE_THR = 3
    nodes = set(unigram.keys())
    E = 0
    for t, neigh in cooc.items():
        for u, w in neigh.items():
            if t < u and w >= EDGE_THR: E += 1
    V = len(nodes)
    graph_density = (2*E)/(V*(V-1)) if V>1 else 0.0
    avg_degree    = (2*E)/V if V>0 else 0.0
    mwe_rate      = len(mwes)/max(1,V)
    tfidf_mass    = sum(tfidf.values())

    # Token source counts & persona counts
    token_story_count  = Counter()
    token_arxiv_count  = Counter()
    token_wiki_count   = Counter()
    token_persona      = defaultdict(Counter)
    for info in doc_info:
        src, per = info["source"], info["persona"]
        seen = set(info["tokens"])
        for t in seen:
            if src == "story": token_story_count[t] += 1
            elif src == "arxiv": token_arxiv_count[t] += 1
            elif src == "wiki": token_wiki_count[t] += 1
            token_persona[per][t] += 1

    # Role map (curated + heuristics)
    role_map = {}
    def mark(role, terms):
        for trm in terms: role_map[trm] = role

    mark("agent", ["human","observer","agency","scientist","student","engineer", "researcher", "practitioner", "tutor", "teacher", "educator", "learner", "you", "we", "i", "me", "us", "they", "them", "author", "authors", "reader", "readers", "user", "users"])
    mark("system",["ai","models","system","systems","architecture","algorithm, algorithms","method","methods","technique","techniques","tool","tools","platform","platforms","software","hardware, devices","network","networks","database","databases"])
    mark("relation",["interaction","protocol","protocols","allocation","dialogue","communication","workflow, workflows","processes","process","mechanism","mechanisms","dynamics","relation","relations","correspondence","mapping","mappings, association","associations","link","links","network","networks"])
    mark("process",["iteration","anchoring","diffusion","optimization","simulation","projection",
                    "transform","transforms","calibration","inference","learning","training","operation, operations", "computation","computations","integration","differentiation","aggregation","aggregation","clustering","classification","regression","evaluation","evaluations","validation","testing","analysis","analyses"])
    mark("stability",["equilibrium","convergence","stabilization","guarantees","rates","robust","consistency, reliable","reliability","stability","stable","robustness, robustness","safety","secure","security, privacy, confidentiality, anonymity, fairness, ethics, ethical, bias, trustworthy, trustworthiness, transparency, explainability, interpretability"])
    mark("structure",["framework","algebra","foundation","schema","categorical","structural","architecture","standard"])
    mark("value",["ethics","transparency","fairness","reliability","trust","safety", "privacy","confidentiality","anonymity","accuracy","precision","clarity","empathy","rigor","creativity","curiosity","efficiency","scalability","sustainability","inclusivity","diversity", "equity","justice","accountability","responsibility","open-source","collaboration","community","impact","benefit","well-being","health","education","learning"])
    mark("geometry",["space","spaces","hilbert","banach","manifold","continuum","metric","geometry", "topology","graph","graphs","lattice","lattices","tensor","tensors", "matrix","matrices","vector","vectors","dimension","dimensions","dimensionality","euclidean","non-euclidean", "discrete","continuous", "measure","measures","norm","norms","distance","distances", "angle","angles","curvature","curvatures", "surface","surfaces","volume","volumes", "area","areas"])
    mark("data",["coreset","entropy","index","scores","literature","kernel","sorting","dataset","corpus", "data","information","knowledge","signal","signals","image","images","video","videos","audio","text","texts","graph","graphs","tree","trees","table","tables","time-series","timeseries","point-cloud","point-clouds", "distribution","distributions","sample","samples","sampling","statistics","statistical","probability","probabilities","random","stochastic","bayesian","frequentist","inference","hypothesis","hypotheses","regression","classification","clustering"])
    mark("math",["theorem","theorems","lemma","lemmas","corollary","corollaries","proposition","propositions","proof","proofs","axiom","axioms","conjecture","conjectures","equation","equations","inequality","inequalities","function","functions","functional","operator","operators","calculus","algebra","geometry","topology","analysis","number-theory","combinatorics","graph-theory","optimization","optimizations"])
    mark("health",["hydration","nutrition","sleep","exercise","fitness","cognition","mental-health","wellness","health","disease","illness","treatment","therapy","prevention","diagnosis","symptom","symptoms","medication","medications","vaccine","vaccines","virus","bacteria","infection","infections","immune-system","immunity","public-health"])
    mark("time",["time","timescale","temporal","chronological","synchronous","asynchronous","real-time","latency","delay","frequency","periodicity","cycle","cycles","trend","trends","seasonality"])
    mark("space",["spatial","location","locations","geographical","geography","map","maps","mapping","coordinate","coordinates","dimension","dimensions","dimensionality","scale","scales","resolution","distance","distances","proximity","adjacency","neighborhood","region","regions","cluster","clusters"])
    mark("emotion",["emotion","emotions","feeling","feelings","mood","moods","sentiment","sentiments","affect","affects","empathy","empathize","compassion","compassionate","anxiety","stress","happiness","joy","sadness","anger","fear","love"])
    mark("cognition",["attention","attentional","perception","perceive","memory","memories","learning","learn","reasoning","reason","decision-making","decision","problem-solving","problem","creativity","creative","imagination","imagine","language","languages","linguistic","communication","communicate","understanding","understand","knowledge","knowledgeable"])
    mark("social",["social","societal","society","community","communities","culture","cultures","cultural","interaction","interactions","relationship","relationships","collaboration","collaborate","group","groups","network","networks","influence","influences","norm","norms","value","values"])
    mark("economics",["economics","economic","market","markets","trade","trading","finance","financial","investment","investments","cost","costs","benefit","benefits","price","prices","demand","supply","competition","competitions","growth","growths","productivity","productivities","efficiency","efficiencies"])
    for t in unigram:
        if t.endswith(("ity","ness")): role_map.setdefault(t,"value")
        elif t.endswith(("tion","sion","ment","ance","ence")): role_map.setdefault(t,"process")
        elif t in {"operator","operators","transform","transforms","embedding","embeddings"}: role_map.setdefault(t,"process")
        elif t in {"framework","algebra","schema","foundation","architecture","standard"}: role_map.setdefault(t,"structure")
        elif t in {"dataset","datasets","corpus","corpora","data","information","knowledge","signal","signals","image","images","video","videos","audio","text","texts","graph","graphs","tree","trees","table","tables","time-series","timeseries","point-cloud","point-clouds"}: role_map.setdefault(t,"data")
        elif t in {"human","observer","agency","scientist","student","engineer", "researcher", "practitioner", "tutor", "teacher", "educator", "learner", "you", "we", "i", "me", "us", "they", "them", "author", "authors", "reader", "readers", "user", "users"}: role_map.setdefault(t,"agent")
        elif t in {"ai","model","models","system","systems","architecture","algorithm", "algorithms","method","methods","technique","techniques","tool","tools","platform","platforms","software","hardware", "devices","network","networks","database","databases"}: role_map.setdefault(t,"system")
        elif t in {"equilibrium","convergence","stabilization","guarantees","rates","robust","consistency","reliable","reliability","stability","stable","robustness","safety","secure","security","privacy","confidentiality","anonymity","fairness","ethics","ethical","bias","trustworthy","trustworthiness","transparency","explainability","interpretability"}: role_map.setdefault(t,"stability")
        elif t in {"space","spaces","hilbert","banach","manifold","continuum","metric","geometry", "topology","graph","graphs","lattice","lattices","tensor","tensors", "matrix","matrices","vector","vectors","dimension","dimensions","dimensionality","euclidean","non-euclidean", "discrete","continuous", "measure","measures","norm","norms","distance","distances", "angle","angles","curvature","curvatures", "surface","surfaces","volume","volumes", "area","areas"}: role_map.setdefault(t,"geometry")
        elif t in {"theorem","theorems","lemma","lemmas","corollary","corollaries","proposition","propositions","proof","proofs","axiom","axioms","conjecture","conjectures","equation","equations","inequality","inequalities","function","functions","functional","operator","operators","calculus","algebra","geometry","topology","analysis","number-theory","combinatorics","graph-theory","optimization","optimizations"}: role_map.setdefault(t,"math")
        elif t in {"hydration","nutrition","sleep","exercise","fitness","cognition","mental-health","wellness","health","disease","illness","treatment","therapy","prevention","diagnosis","symptom","symptoms","medication","medications","vaccine","vaccines","virus","bacteria","infection","infections","immune-system","immunity","public-health"}: role_map.setdefault(t,"health")
        elif t in {"time","timescale","temporal","chronological","synchronous","asynchronous","real-time","latency","delay","frequency","periodicity","cycle","cycles","trend","trends","seasonality"}: role_map.setdefault(t,"time")
        elif t in {"spatial","location","locations","geographical","geography","map","maps","mapping","coordinate","coordinates","dimension","dimensions","dimensionality","scale","scales","resolution","distance","distances","proximity","adjacency","neighborhood","region","regions","cluster","clusters"}: role_map.setdefault(t,"space")
        elif t in {"emotion","emotions","feeling","feelings","mood","moods","sentiment","sentiments","affect","affects","empathy","empathize","compassion","compassionate","anxiety","stress","happiness","joy","sadness","anger","fear","love"}: role_map.setdefault(t,"emotion")
        elif t in {"attention","attentional","perception","perceive","memory","memories","learning","learn","reasoning","reason","decision-making","decision","problem-solving","problem","creativity","creative","imagination","imagine","language","languages","linguistic","communication","communicate","understanding","understand","knowledge","knowledgeable"}: role_map.setdefault(t,"cognition")
        elif t in {"social","societal","society","community","communities","culture","cultures","cultural","interaction","interactions","relationship","relationships","collaboration","collaborate","group","groups","network","networks","influence","influences","norm","norms","value","values"}: role_map.setdefault(t,"social")
        elif t in {"economics","economic","market","markets","trade","trading","finance","financial","investment","investments","cost","costs","benefit","benefits","price","prices","demand","supply","competition","competitions","growth","growths","productivity","productivities","efficiency","efficiencies"}: role_map.setdefault(t,"economics")

    density = {
        "nodes": V, "edges>=3": E, "graph_density": graph_density, "avg_degree": avg_degree,
        "mwe_count": len(mwes), "mwe_rate": mwe_rate, "tfidf_mass": tfidf_mass,
        "sources": dict(source_count),
        "wiki_ratio": source_count["wiki"]/max(1,sum(source_count.values())),
        "story_ratio": source_count["story"]/max(1,sum(source_count.values()))
    }

    # Persona style registry (from front-matter)
    persona_style = {}
    for info in doc_info:
        per = info["persona"]
        if not per or per == "none": continue
        meta = info.get("meta",{})
        style = {
            "voice": meta.get("voice","neutral"),
            "values": meta.get("values",[]),
            "domains": meta.get("domains",[]),
            "style": meta.get("style",[]),
            "pronoun": meta.get("pronoun","we")
        }
        base = persona_style.get(per, {"voice":"neutral","values":[], "domains":[], "style":[], "pronoun":"we"})
        base["voice"]   = style["voice"] or base["voice"]
        base["pronoun"] = style["pronoun"] or base["pronoun"]
        base["values"]  = sorted(set(base["values"] + style["values"]))
        base["domains"] = sorted(set(base["domains"] + style["domains"]))
        base["style"]   = sorted(set(base["style"] + style["style"]))
        persona_style[per] = base

    return {
        "docs": docs,
        "unigram": dict(unigram),
        "bigram": {f"{a} {b}":c for (a,b),c in bigram.items()},
        "trigram": {f"{a} {b} {c}":c for (a,b,c),c in trigram.items()},
        "mwes": sorted(mwes),
        "df": dict(df),
        "idf": {k: float(v) for k,v in idf.items()},
        "tfidf": {k: float(v) for k,v in tfidf.items()},
        "cooc": {t: dict(n) for t,n in cooc.items()},
        "role_map": role_map,
        "density": density,
        "edge_threshold": EDGE_THR,
        "token_story_count": dict(token_story_count),
        "token_arxiv_count": dict(token_arxiv_count),
        "token_wiki_count": dict(token_wiki_count),
        "token_persona": {p: dict(c) for p,c in token_persona.items()},
        "personas": sorted([p for p in personas if p and p!="none"]),
        "persona_style": persona_style,
        "total_entries": len(entries),
        "total_docs": len(docs),
        "last_built": int(time.time())
    }

# --------------------------
# Embedding + Attention
# --------------------------
def top_k_context(cooc, token, k=128):
    neigh = cooc.get(token, {})
    if not neigh: return []
    return [w for w,_ in sorted(neigh.items(), key=lambda x: x[1], reverse=True)[:k]]

def embed_token(token, lex, dim=128):
    random.seed(hash(token) & 0xffffffff)
    vec = [0.0]*dim
    ctx = top_k_context(lex["cooc"], token, k=dim)
    for i, w in enumerate(ctx):
        val = lex["idf"].get(w, 1.0)
        vec[i % dim] += val
    vec[(hash(token)+13)%dim] += lex["idf"].get(token, 1.0)
    n = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x/n for x in vec]

def dot(u,v): return sum(a*b for a,b in zip(u,v))
def softmax(xs):
    m = max(xs) if xs else 0.0
    ex = [math.exp(x-m) for x in xs]
    s = sum(ex) or 1.0
    return [x/s for x in ex]

def multihead_attention(query_tokens, candidate_tokens, lex, dim=96, heads=4, seed=RANDOM_SEED):
    rng = random.Random(seed)
    head_offsets = [rng.randint(0,10_000_000) for _ in range(heads)]
    cand_vecs = {t: embed_token(t, lex, dim) for t in candidate_tokens}
    scores = Counter()
    for off in head_offsets:
        qv = [0.0]*dim
        for qt in query_tokens:
            ev = embed_token(qt + f"_h{off}", lex, dim)
            qv = [a+b for a,b in zip(qv, ev)]
        scale = math.sqrt(dim)
        raw, order = [], []
        for ct, v in cand_vecs.items():
            s = dot(qv, v) / scale
            raw.append(s); order.append(ct)
        attn = softmax(raw)
        for ct, w in zip(order, attn):
            scores[ct] += w
    return scores

# =====================[ NEW: Fwd/Bwd/Post-hoc + C3F ]==========================
def mh_forward_scores(query_tokens, candidate_tokens, embed_fn, dim=96, heads=4, seed=1729):
    rng = random.Random(seed)
    head_offsets = [rng.randint(0,10_000_000) for _ in range(heads)]
    cand_vecs = {t: embed_fn(t, dim) for t in candidate_tokens}
    scores = Counter()
    for off in head_offsets:
        qv = [0.0]*dim
        for qt in query_tokens:
            ev = embed_fn(qt + f"_h{off}", dim)
            qv = [a+b for a,b in zip(qv, ev)]
        scale = math.sqrt(dim)
        for ct, v in cand_vecs.items():
            scores[ct] += dot(qv, v) / scale
    return scores

def mh_backward_scores(generated_tokens, candidate_tokens, embed_fn, dim=96, heads=4, seed=7331):
    rng = random.Random(seed)
    head_offsets = [rng.randint(0,10_000_000) for _ in range(heads)]
    cand_vecs = {t: embed_fn(t, dim) for t in candidate_tokens}
    scores = Counter()
    for off in head_offsets:
        gv = [0.0]*dim
        for gt in generated_tokens:
            ev = embed_fn(gt + f"_b{off}", dim)
            gv = [a+b for a,b in zip(gv, ev)]
        scale = math.sqrt(dim)
        for ct, v in cand_vecs.items():
            scores[ct] += dot(gv, v) / scale
    return scores

def mh_posthoc_blend(forward_scores, backward_scores, alpha=0.65):
    out = Counter()
    keys = set(forward_scores) | set(backward_scores)
    for k in keys:
        f = forward_scores.get(k,0.0)
        b = backward_scores.get(k,0.0)
        out[k] = alpha*f + (1.0-alpha)*b
    return out

def normalize_scores(counter):
    if not counter: return {}
    items = list(counter.items())
    tokens, vals = zip(*items)
    probs = softmax(list(vals))
    return dict(zip(tokens, probs))

def weighted_quantile(values, weights, q):
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    tot = sum(w for _,w in pairs) or 1.0
    c = 0.0
    for v, w in pairs:
        c += w
        if c / tot >= q:
            return v
    return pairs[-1][0]

class C3F:
    def __init__(self, groups, alpha=0.1, pi=None, lambda_reg=0.0):
        self.groups = list(groups)
        self.alpha = float(alpha)
        self.pi = {a: 1.0/len(self.groups) for a in self.groups} if pi is None else dict(pi)
        self.alpha_a = {a: self.alpha * self.pi[a] for a in self.groups}
        self.lambda_reg = float(lambda_reg)
        self.q_hat = {a: None for a in self.groups}

    def calibrate(self, Dcal, weight_model=None, pse_module=None):
        weight_model = weight_model or (lambda x,a: 1.0)
        by_vals, by_wts = defaultdict(list), defaultdict(list)
        for row in Dcal:
            a = row["a"]; w = float(weight_model(row["x"], a))
            by_vals[a].append(float(row["eta"])); by_wts[a].append(max(0.0, w))
        q = {}
        for a in self.groups:
            vals, wts = by_vals[a], by_wts[a]
            q[a] = float("inf") if not vals else weighted_quantile(vals, wts, 1.0 - self.alpha_a[a])
        if pse_module and self.lambda_reg > 0.0:
            delta_cf, grad = pse_module(q)
            q = {a: q[a] * (1.0 + self.lambda_reg * float(grad.get(a, 0.0))) for a in self.groups}
        self.q_hat = q
        return q

def pse_surrogate_factory(groups):
    def _pse(q_dict):
        qs = [q_dict[a] for a in groups if q_dict[a] is not None and math.isfinite(q_dict[a])]
        if not qs: return 0.0, {a:0.0 for a in groups}
        gap = max(qs) - min(qs)
        mean_q = sum(qs)/len(qs)
        g = {a: ((q_dict[a]-mean_q)/(abs(gap)+1e-9)) if (q_dict[a] is not None and math.isfinite(q_dict[a])) else 0.0
             for a in groups}
        return gap, g
    return _pse

def c3f_on_role_scores(role_to_scores, alpha=0.1, lambda_reg=0.0):
    groups = list(role_to_scores.keys())
    Dcal = []
    for a, pairs in role_to_scores.items():
        if not pairs: continue
        smax = max(s for _,s in pairs)
        for tok, s in pairs:
            eta = smax - s  # nonconformity
            Dcal.append({"x": tok, "y": None, "a": a, "eta": float(eta)})
    c3f = C3F(groups, alpha=alpha, pi=None, lambda_reg=lambda_reg)
    q = c3f.calibrate(Dcal, weight_model=None, pse_module=pse_surrogate_factory(groups))
    return q  # role -> threshold on nonconformity

# --------------------------
# Roles & Grammar + Persona weighting
# --------------------------
ROLE_WEIGHT = {
    "agent": 0.95, "system": 0.90, "relation": 0.88, "process": 0.86,
    "stability": 0.84, "structure": 0.82, "geometry": 0.80, "value": 0.78,
    "data": 0.76, "misc": 0.40
}

def token_role(tok, lex): return lex["role_map"].get(tok, "misc")

# ----------[ Minimal verb planner ]----------
ALLOWED_PREPS = ["for", "to", "in", "with", "by"]

ROLE_TO_VERBS = {
    "process":   ["explains", "organizes", "updates", "links", "stabilizes", "optimizes", "models", "predicts", "clarifies", "analyzes", "transforms", "learns", "trains", "infers", "operates", "simulates", "iterates", "projects", "calibrates", "validates", "evaluates"],
    "relation":  ["connects", "coordinates", "mediates", "guides", "structures", "relates", "communicates", "allocates", "schedules", "balances", "interfaces", "negotiates", "regulates", "facilitates", "orchestrates", "integrates", "synthesizes", "harmonizes"],
    "stability": ["stabilizes", "balances", "maintains", "regulates", "ensures", "guarantees", "converges", "secures", "sustains", "controls", "preserves", "fortifies", "validates", "verifies", "confirms", "supports", "strengthens", "protects", "resists"],
    "structure": ["defines", "frames", "specifies", "describes", "structures", "organizes", "classifies", "categorizes", "models", "represents", "encodes", "schemas", "architects", "designs", "builds", "constructs", "formulates", "establishes", "standardizes"],
    "system":    ["processes", "represents", "supports", "enables", "powers", "controls", "manages", "executes", "runs", "operates", "facilitates", "orchestrates", "integrates", "synthesizes", "harmonizes", "architects", "designs", "builds"],
    "agent":     ["learns", "uses", "builds", "reasons about", "adapts", "decides", "plans", "observes", "measures", "tests", "explores", "discovers", "creates", "innovates", "solves", "analyzes", "synthesizes", "collaborates", "communicates", "teaches", "educates", "instructs"],
    "data":      ["stores", "summarizes", "indexes", "describes", "analyzes", "processes", "collects", "curates", "manages", "retrieves", "sorts", "filters", "samples", "clusters", "embeds", "represents", "visualizes", "interprets", "measures", "quantifies", "evaluates"],
    "geometry":  ["shapes", "positions", "locates", "embeds", "maps", "transforms", "measures", "quantifies", "analyzes", "models", "represents", "visualizes", "navigates", "orients", "aligns", "rotates", "scales", "projects", "optimizes", "interpolates", "extrapolates"],
    "value":     ["improves", "respects", "supports", "promotes", "ensures", "enhances", "fosters", "builds", "cultivates", "advances", "protects", "safeguards", "upholds", "champions", "prioritizes", "values", "trusts", "relies on", "guarantees", "validates", "verifies"],
    "misc":      ["relates to", "involves", "pertains to", "concerns", "addresses", "discusses", "examines", "explores", "analyzes", "investigates", "considers", "reviews", "summarizes", "highlights", "emphasizes", "focuses on", "centers on", "revolves around", "touches on", "connects with", "links to"]
}

OVERUSED_JARGON = {"categorical", "operators", "operator", "protocols", "allocation",
                   "standard", "schema", "framework"}
def penalize_jargon(tok: str) -> float:
    # Small, surgical penalty: discourage tokens that fuel word-salad, without banning them.
    return 0.10 if tok in OVERUSED_JARGON else 0.0

def pick_verb(subj_role: str, proc_role: str, obj_role: str, rng: random.Random) -> str:
    # Prefer the process role; back off to subject/object role; then misc.
    for r in (proc_role, subj_role, obj_role, "misc"):
        cand = ROLE_TO_VERBS.get(r, [])
        if cand: return rng.choice(cand)
    return "relates to"

def article_for(word: str) -> str:
    if not word: return ""
    return "an " if word[0].lower() in "aeiou" else "a "

def cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s

def score_token(tok, lex, attn_scores, used, persona=None, story_weight=0.4, persona_weight=0.6, source_pref=None):
    role = token_role(tok, lex)
    base = ROLE_WEIGHT.get(role, 0.4)
    tfidf = lex["tfidf"].get(tok, 0.0)
    attn  = attn_scores.get(tok, 0.0)
    ts = lex["token_story_count"].get(tok,0)
    ta = lex["token_arxiv_count"].get(tok,0)
    tw = lex["token_wiki_count"].get(tok,0)
    denom = ts+ta+tw or 1
    story_bias = ts/denom
    arxiv_bias = ta/denom
    wiki_bias = tw/denom
    persona_bias = 0.0
    if persona and persona in lex["token_persona"]:
        persona_bias = lex["token_persona"][persona].get(tok,0) / max(1,sum(lex["token_persona"][persona].values()))
        persona_bias *= 3.0
    use_pen = min(0.6, 0.12*used.get(tok,0))
    jargon_pen = penalize_jargon(tok)  # NEW: gentle downweight for overused abstractions
    score = (0.55*base + 0.20*math.tanh(0.1*tfidf) + 0.15*attn +
             0.07*story_weight*story_bias + 0.08*persona_weight*persona_bias +
             0.09*arxiv_bias - 0.03*wiki_bias)
    if source_pref:
        score += 0.12 * source_pref.get("arxiv", 0.0) * arxiv_bias
        score += 0.05 * source_pref.get("story", 0.0) * story_bias
        score += 0.03 * source_pref.get("wiki", 0.0) * wiki_bias
    score = score * (1.0 - use_pen) * (1.0 - jargon_pen)
    return score, role

def pick_by_role(candidates, roles, lex, attn_scores, used, persona, story_weight, persona_weight, avoid=set(), source_pref=None):
    best, best_s = None, -1
    for t in candidates:
        if t in avoid: continue
        s, r = score_token(t, lex, attn_scores, used, persona, story_weight, persona_weight, source_pref=source_pref)
        if r in roles and s > best_s:
            best, best_s = t, s
    return best

def style_params_for_persona(lex, persona):
    style = lex["persona_style"].get(persona or "", {"voice":"neutral","values":[],"domains":[],"style":[],"pronoun":"we"})
    pronoun = style.get("pronoun","we")
    voice   = style.get("voice","neutral")
    # Avoid "through" completely; keep prepositions short and concrete.
    if voice == "concise":
        connectors = ["with", "by"]
    elif voice == "narrative":
        connectors = ["with", "by"]
    else:
        connectors = ["with", "in"]
    return {"pronoun": pronoun, "voice": voice, "connectors": connectors}

def compose_sentence_weighted(cand, lex, attn_scores, used_tokens, rng, persona=None, story_weight=0.4, persona_weight=0.6, source_pref=None):
    """
    Drop-in replacement: produces cleaner, grammatical sentences while preserving
    your role-aware and persona-weighted selection logic.
    """

    # ---------- small helpers (local; no global changes required) ----------
    def tok_to_phrase(t: str) -> str:
        # Turn MWEs into readable phrases and strip artifacts
        return (t or "").replace("_", " ").strip()
    
    def is_acronym(w: str) -> bool:
        return w.isupper() and 1 < len(w) <= 5  # e.g., HCI, NLP, SVM

    def is_plural(word: str) -> bool:
        w = tok_to_phrase(word).split()[-1] if word else ""
        # Heuristic plural: ends with 's' but not 'ss'/'is'; skip very short tokens.
        return len(w) > 2 and w.endswith("s") and not (w.endswith("ss") or w.endswith("is"))

    def needs_an(w: str) -> bool:
        lead = tok_to_phrase(w).strip()
        if not lead: return False
        first = lead.split()[0]
        # If it's an acronym, some letters sound like vowels (F, H, L, M, N, R, S, X)
        if is_acronym(first):
            return first[0] in set("FHLMNRSX")
        # Otherwise, normal vowel rule
        return first[0].lower() in set("aeiou")

    def article_for_np(head: str) -> str:
        if not head: return ""
        # If the head starts with a determiner already, don't add another
        if re.match(r"^(a|an|the)\s+", head, flags=re.I): return ""
        # Plurals get no article
        if is_plural(head): return ""
        return "an " if needs_an(head) else "a "

    def cap(s: str) -> str:
        return s[:1].upper() + s[1:] if s else s

    def base_form_verb(v: str) -> str:
        # Convert a 3sg verb ("explains"/"stabilizes"/"balances"/"guides") to base form
        if v.endswith("ies"): return v[:-3] + "y"
        if v.endswith("ches") or v.endswith("shes"): return v[:-2]
        if v.endswith("ses") or v.endswith("xes") or v.endswith("zes") or v.endswith("oes"): return v[:-2]
        if v.endswith("es") and len(v) > 3: return v[:-2]
        if v.endswith("s") and not v.endswith("ss"): return v[:-1]
        return v

    def agree_verb(v: str, subj_plural: bool) -> str:
        # ROLE_TO_VERBS stores 3sg ("explains"). Use base form for plural subjects.
        return base_form_verb(v) if subj_plural else v

    def role_of(t: str) -> str:
        return token_role(t, lex)

    def best_fallback(role_set):
        pool = [t for t in filtered if role_of(t) in role_set]
        pool.sort(key=lambda t: (attn_scores.get(t, 0.0), lex["tfidf"].get(t, 0.0)), reverse=True)
        return pool[0] if pool else None

    def build_np(head: str, extra_roles):
        if not head: return None
        head_p = tok_to_phrase(head)
        BAD_MODIFIERS = {"schema", "framework", "standard", "operator", "protocols", "allocation", "categorical", "operators", "algebra", "foundation", "architecture", "structure", "structures", "system", "systems", "data", "value", "values", "geometry", "geometries", "relation", "relations", "process", "processes", "stability", "stabilities"}
        extras = [t for t in filtered if t != head and role_of(t) in extra_roles]
        rng.shuffle(extras)
        for mod_tok in extras:
            mod = tok_to_phrase(mod_tok)
            if not mod or mod in BAD_MODIFIERS or mod in head_p:
                continue
            # prefer short, concrete modifiers
            if 3 <= len(mod) <= 14:
                return f"{head_p} {mod}"
        return head_p

    # ---------- candidate filtering ----------
    filtered = [t for t in cand if t and t in lex["idf"] and t not in STOPWORDS and len(t) > 2]
    if not filtered:
        return "Insight organizes understanding."

    # ---------- pick roles (with robust fallbacks) ----------
    subj = pick_by_role(filtered, {"agent", "system", "structure"}, lex, attn_scores, used_tokens,
                        persona, story_weight, persona_weight, source_pref=source_pref)
    if not subj:
        subj = best_fallback({"agent", "system", "structure"}) or rng.choice(filtered)

    proc = pick_by_role(filtered, {"process", "relation", "stability"}, lex, attn_scores, used_tokens,
                        persona, story_weight, persona_weight, avoid={subj}, source_pref=source_pref)
    if not proc:
        proc = best_fallback({"process", "relation", "stability"}) or rng.choice(filtered)

    obj  = pick_by_role(filtered, {"system", "structure", "geometry", "data", "value"}, lex, attn_scores, used_tokens,
                        persona, story_weight, persona_weight, avoid={subj, proc}, source_pref=source_pref)
    if not obj:
        obj = best_fallback({"system", "structure", "geometry", "data", "value"})

    mod1 = pick_by_role(filtered, {"value", "stability", "structure"}, lex, attn_scores, used_tokens,
                        persona, story_weight, persona_weight, avoid={subj, proc, obj}, source_pref=source_pref)

    # ---------- NPs & verb ----------
    subj_np = build_np(subj, {"relation", "structure"})
    obj_np  = build_np(obj,  {"process", "stability", "value"})
    subj_plural = is_plural(subj_np or "")

    # Verb choice & agreement
    v = pick_verb(role_of(subj), role_of(proc), role_of(obj), rng)
    v = agree_verb(v, subj_plural)

    # Preposition choice guided by modifier role
    role_prep = {
        "value": "for", "stability": "with", "structure": "in",
        "relation": "with", "process": "by"
    }
    prep = role_prep.get(role_of(mod1 or ""), rng.choice(["with", "by", "in", "for", "to", "using", "via", "through", "about", "regarding", "concerning", "on", "of", "as", "at", "over", "under", "along", "around", "near", "inside", "outside", "beyond", "across", "between", "among", "towards", "upon", "within", "without", "alongside", "amidst", "despite", "except", "including", "regardless of", "pertaining to", "relating to", "involving", "concerning", "focusing on", "centered on", "geared towards", "aimed at", "dedicated to", "devoted to", "specializing in", "emphasizing", "highlighting", "showcasing", "featuring", "spotlighting", "underscoring", "accentuating", "illuminating", "clarifying", "elucidating", "explicating", "demonstrating", "exemplifying", "illustrating", "depicting", "portraying", "representing", "symbolizing", "signifying", "denoting", "indicating", "suggesting", "implying", "alluding to", "hinting at", "referencing", "citing", "quoting", "mentioning", "noting", "observing", "commenting on", "remarking on", "reflecting on", "pondering", "contemplating", "considering", "examining", "analyzing", "evaluating", "assessing", "appraising", "judging", "weighing", "balancing", "comparing", "contrasting", "differentiating", "distinguishing", "separating", "dividing", "partitioning", "segmenting", "classifying", "categorizing", "grouping", "clustering", "organizing", "structuring", "arranging", "aligning", "coordinating", "synchronizing", "integrating", "combining", "merging", "fusing", "blending", "mixing", "amalgamating", "uniting", "joining", "linking", "connecting", "associating", "relating", "correlating", "mapping", "charting", "graphing", "plotting", "visualizing", "illustrating", "depicting", "representing", "modeling", "simulating", "emulating", "replicating", "mirroring", "reflecting", "echoing", "resonating with", "harmonizing with", "synchronizing with", "aligning with", "coordinating with", "integrating with", "combining with", "merging with", "fusing with", "blending with", "mixing with", "amalgamating with", "uniting with", "joining with", "linking with", "connecting with", "associating with", "relating to", "correlating with", "mapping to", "charting to", "graphing to", "plotting to", "visualizing to", "illustrating to", "depicting to", "representing to", "modeling to", "simulating to", "emulating to", "replicating to", "mirroring to", "reflecting to", "echoing to", "resonating with", "harmonizing with", "synchronizing with", "aligning with"]))

    # ---------- compose clean sentence ----------
    parts = []

    if subj_np:
        parts.append(cap(subj_np))
    else:
        parts.append("System")  # hard fallback

    parts.append(v)

    if obj_np:
        art = article_for_np(obj_np)
        # Avoid "a systems" & duplicate articles
        obj_phrase = f"{art}{obj_np}".strip()
        parts.append(obj_phrase)

    if mod1:
        parts.append(f"{prep} {tok_to_phrase(mod1)}")

    sent = " ".join([p for p in parts if p]).strip()

    # Final cleanups: collapse repeats, normalize spaces, ensure period.
    sent = re.sub(r"\b(\w+)\s+\1\b", r"\1", sent)
    sent = re.sub(r"\s{2,}", " ", sent).rstrip(".") + "."

    # Update usage penalties
    for t in filter(None, [subj, proc, obj, mod1]):
        used_tokens[t] = used_tokens.get(t, 0) + 1

    return sent

# --------------------------
# Transfinite iteration + semantic repetition learning
# --------------------------
def sentence_vector(sent, lex, dim=96):
    toks = [t for t in advanced_tokenize(sent) if t in lex["idf"]]
    if not toks: return [0.0]*dim
    vecs = [embed_token(t, lex, dim) for t in toks]
    agg = [0.0]*dim
    for v in vecs: agg = [a+b for a,b in zip(agg,v)]
    n = math.sqrt(sum(x*x for x in agg)) or 1.0
    return [x/n for x in agg]

def cosine(u,v):
    return dot(u,v) / (math.sqrt(dot(u,u))+1e-9) / (math.sqrt(dot(v,v))+1e-9)

def candidate_pool(lex, top_k=4000):
    cand = [t for t,_ in sorted(lex["unigram"].items(), key=lambda x: x[1], reverse=True)[:top_k]]
    cand = [t for t in cand if t not in STOPWORDS]
    cand += [m for m in lex["mwes"][:600]]
    seen, out = set(), []
    for t in cand:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def fairness_filter_with_c3f(cand, attn_scores, lex, alpha=0.10, lambda_reg=0.25):
    role_buckets = defaultdict(list)
    for t in cand:
        role_buckets[token_role(t, lex)].append((t, attn_scores.get(t, 0.0)))
    q_nonconf = c3f_on_role_scores(role_buckets, alpha=alpha, lambda_reg=lambda_reg)
    filtered = []
    for role, pairs in role_buckets.items():
        if not pairs: continue
        smax = max(s for _,s in pairs)
        thr = q_nonconf.get(role, float("inf"))
        for tok, s in pairs:
            eta = smax - s
            if eta <= thr:
                filtered.append(tok)
    return (filtered if filtered else cand), q_nonconf

# --------[ Definition-first booster helpers ]--------
DEF_HINTS = {
    "schema":       "a mental framework used to organize and predict experiences",
    "existence":    "the condition of there being anything at all rather than nothing",
    "algorithm":    "a step-by-step procedure for solving a class of problems",
    "optimization": "the process of improving a system to meet a goal under constraints",
    "model":        "a simplified representation of a system used for explanation or prediction",
    "data":         "factual information used as a basis for reasoning, discussion, or calculation",
    "system":       "a set of interacting or interdependent components forming an integrated whole",
    "structure":    "the arrangement of and relations between the parts or elements of something complex",
    "process":      "a series of actions or steps taken to achieve a particular end",
    "relation":     "the way in which two or more concepts, objects, or people are connected",
    "stability":    "the quality of being steady and unchanging over time",
    "geometry":     "the branch of mathematics concerned with the properties and relations of points, lines, surfaces, and solids",
    "value":        "the importance, worth, or usefulness of something",
    "agent":        "an entity that perceives its environment and takes actions to achieve goals",
    "fairness":     "the quality of making judgments that are free from discrimination or bias",
    "optimization": "the process of making something as effective or functional as possible",
    "convergence":  "the process of approaching a limit or an optimal solution",
    "algebra":      "a branch of mathematics dealing with symbols and the rules for manipulating those symbols",
    "theorem":      "a statement that has been proven on the basis of previously established statements",
    "metric":       "a standard of measurement used to assess performance or progress",
    "mapping":      "a function that associates elements of one set with elements of another set",
    "fixed point":  "a point that remains unchanged under a given function or operation",
    "cannabis":     "a plant used for medicinal and recreational purposes, known for its psychoactive properties",
    "cannabinoid":  "a class of chemical compounds that act on cannabinoid receptors in cells",
    "arxiv":        "an open-access repository of electronic preprints in various fields of science",
    "wikipedia":    "a free online encyclopedia that allows collaborative editing of its content",
    "reflective output": "a summary layer that stitches evidence into a grounded takeaway",
    "medical cannabis": "the regulated medical use of cannabis to manage chronic pain and other conditions",
    "recep tayyip erdogan": "the president of Turkey known for his influential role in Turkish politics",
    "german citizenship": "the status of being a legal member of Germany, granting rights and responsibilities",
    "faruk alpay":  "a researcher known for work on fairness and optimization in algorithms",
    "alpay algebra": "an algebraic framework for reasoning about fairness and adaptive protocols",
    "banach fixed-point theorem": "a theorem guaranteeing a unique fixed point for contraction mappings on complete metric spaces",
    "hilbert space": "a complete inner product space used in functional analysis and quantum mechanics"
}

TOPIC_ALIAS_DEFS = [
    {
        "keys": {"weed", "cannabis", "medical_cannabis"},
        "display": "medical cannabis",
        "lookup": "medical cannabis",
        "wiki_topics": ["Medical cannabis", "Cannabinoid"],
        "dictionary_override": "{Head} is the regulated medical use of cannabis to manage chronic pain, calm seizures, and ease treatment side effects.",
        "booster_sentences": [
            "Clinicians rely on {head_lower} to reduce chemotherapy nausea, improve appetite, and give patients steadier sleep.",
            "By tuning THC and CBD ratios, {head_lower} can ease neuropathic pain while helping people cut back on opioids."
        ],
        "force_wiki": True
    },
    {
        "keys": {"faruk", "faruk_alpay", "alpay"},
        "display": "Faruk Alpay",
        "lookup": "Faruk Alpay",
        "preferred_sources": {"arxiv": 0.7},
        "dictionary_override": "{Head} is a fairness and optimization researcher whose arXiv preprints link algorithmic guarantees with human-centered deployment.",
        "booster_sentences": [
            "Recent arXiv work by {head_lower} studies stability bounds for resource allocation and conversational agents.",
            "Across the corpus, {head_lower} blends fixed-point constructions with bias-aware evaluation pipelines."
        ],
        "arxiv_terms": ["Faruk Alpay", "Alpay"],
        "force_wiki": False
    },
    {
        "keys": {"alpay_algebra"},
        "display": "Alpay algebra",
        "lookup": "Alpay algebra",
        "preferred_sources": {"arxiv": 0.6},
        "dictionary_override": "{Head} is Faruk Alpay's algebraic framework for reasoning about fairness, convergence, and adaptive protocols.",
        "booster_sentences": [
            "In arXiv manuscripts it packages fixed points, coinduction, and allocation rules into a single operator algebra.",
            "Use {head_lower} to track how updates propagate through socio-technical loops while keeping bias constraints explicit."
        ],
        "arxiv_terms": ["Alpay algebra"],
        "force_wiki": False
    },
    {
        "keys": {"reflective_output"},
        "display": "reflective output",
        "lookup": "reflective output",
        "dictionary_override": "{Head} is the engine's persona-aware summary layer that stitches evidence into a grounded takeaway.",
        "booster_sentences": [
            "Each reflective output blends arXiv, Wikipedia, and story traces, then calibrates tone with the chosen persona.",
            "It finishes with a quick check so you restate the idea and link it to your own practice.",
            "If you need a different angle, ask for another reflective output and I'll regenerate with the same grounded evidence."
        ],
        "force_wiki": False
    },
    {
        "keys": {"recep_tayyip_erdogan"},
        "display": "Recep Tayyip Erdogan",
        "lookup": "Recep Tayyip Erdogan",
        "wiki_topics": ["Recep Tayyip Erdogan"],
        "dictionary_override": "{Head} has steered modern Turkish politics, serving as president since 2014 after a decade as prime minister for the Justice and Development Party he co-founded.",
        "booster_sentences": [
            "He rose from Istanbul's mayoralty in the 1990s to national leadership, pairing rapid infrastructure growth with tighter executive control.",
            "Critics cite democratic backsliding and curbs on the press during {head_lower}'s tenure, especially after the 2016 coup attempt."
        ],
        "force_wiki": True
    },
    {
        "keys": {"german_citizen", "german_citizenship"},
        "display": "German citizenship",
        "lookup": "German citizenship",
        "wiki_topics": ["German nationality law", "Citizenship of the European Union"],
        "dictionary_override": "{Head} unlocks EU mobility, deep social insurance, and the right to participate in Germany's democratic institutions.",
        "booster_sentences": [
            "As an EU citizen you can live, work, and study across the Schengen area without extra visas.",
            "Germany's public healthcare and pension systems extend to {head_lower}, providing stability for families and founders.",
            "Dual citizenship reforms now make it easier to keep another passport while enjoying Germany's protections."
        ],
        "force_wiki": True
    },
    {
        "keys": {"fixed_point_theorem", "banach_fixed_point_theorem", "banach_fixed-point_theorem"},
        "display": "Banach fixed-point theorem",
        "lookup": "Banach fixed point theorem",
        "wiki_topics": ["Banach fixed-point theorem"],
        "preferred_sources": {"arxiv": 0.5},
        "dictionary_override": "{Head} guarantees a unique fixed point for any contraction mapping on a complete metric space.",
        "booster_sentences": [
            "It underpins iterative solvers: start anywhere, apply the contraction repeatedly, and the sequence converges to the fixed point.",
            "Analysts use {head_lower} to prove existence of solutions to differential and integral equations via successive approximations."
        ],
        "arxiv_terms": ["Banach", "fixed point"],
        "force_wiki": True
    },
    {
        "keys": {"hilbert_space", "hilbert_spaces"},
        "display": "Hilbert space",
        "lookup": "Hilbert space",
        "wiki_topics": ["Hilbert space"],
        "preferred_sources": {"arxiv": 0.4},
        "dictionary_override": "{Head} is a complete inner-product space where geometry and analysis meet for infinite-dimensional problems.",
        "booster_sentences": [
            "Orthonormal bases in {head_lower} let you expand signals, quantum states, or PDE solutions like Fourier series.",
            "Completeness means every Cauchy sequence of vectors converges inside the space, so limit arguments stay valid."
        ],
        "arxiv_terms": ["Hilbert space"],
        "force_wiki": True
    },
    {
        "keys": {"water_quality", "water_testing"},
        "display": "water quality",
        "lookup": "water quality",
        "wiki_topics": ["Water quality"],
        "dictionary_override": "To check {Head_lower}, start by matching your tests to the standard you care about—drinking water, irrigation, or aquatic habitat.",
        "booster_sentences": [
            "Field kits cover quick indicators such as pH, dissolved oxygen, and turbidity so you catch shifts before they escalate.",
            "For compliance sampling, draw water in clean bottles, label the site and time, and send split samples to a certified lab for confirmation."
        ],
        "howto_steps": [
            "Define the regulatory or health limits you need to satisfy, such as EPA drinking-water tables or local watershed criteria.",
            "Collect a representative sample: rinse the container with site water, grab mid-stream flow, and keep it chilled while you travel.",
            "Run quick field checks—pH, conductivity, dissolved oxygen, and turbidity—then note weather, odors, or visible pollution.",
            "Use test strips or portable meters for nitrate, hardness, and chlorine when you need instant guidance.",
            "Ship a sealed split sample to a certified laboratory for metals, pathogens, and organics that require precise instruments.",
            "Compare the results against your target standard and log them, flagging any parameter that exceeds the threshold for follow-up treatment."
        ],
        "force_wiki": True
    },
    {
        "keys": {"machine_learning", "ml"},
        "display": "machine learning",
        "lookup": "machine learning",
        "wiki_topics": ["Machine learning"],
        "dictionary_override": "{Head} is a branch of artificial intelligence where algorithms learn patterns from data to make predictions or decisions.",
        "booster_sentences": [
            "Common techniques include supervised learning for labeled data, unsupervised learning for clustering, and reinforcement learning for sequential decision-making.",
            "{head_lower} powers applications like recommendation systems, image recognition, natural language processing, and autonomous vehicles."
        ],
        "force_wiki": True
    },
    {
        "keys": {"neural_network", "neural_networks"},
        "display": "neural network",
        "lookup": "neural network",
        "wiki_topics": ["Artificial neural network"],
        "dictionary_override": "{Head} is a computational model inspired by the brain's interconnected neurons, used to recognize patterns and make decisions.",
        "booster_sentences": [
            "They consist of layers of nodes (neurons) that process inputs through weighted connections, applying activation functions to capture non-linear relationships.",
            "{head_lower} are widely used in deep learning for tasks like image and speech recognition, natural language processing, and game playing."
        ],
        "force_wiki": True
    },
    {
        "keys": {"blockchain"},
        "display": "blockchain",
        "lookup": "blockchain",
        "wiki_topics": ["Blockchain"],
        "dictionary_override": "{Head} is a decentralized ledger technology that records transactions across a network of computers in a secure, transparent, and tamper-evident way.",
        "booster_sentences": [
            "It enables trustless interactions by using cryptographic techniques and consensus mechanisms to validate and append new blocks of data.",
            "{head_lower} underpins cryptocurrencies like Bitcoin and Ethereum, as well as applications in supply chain management, voting systems, and digital identity."
        ],
        "force_wiki": True
    },
    {
        "keys": {"quantum_computing", "quantum_computer"},
        "display": "quantum computing",
        "lookup": "quantum computing",
        "wiki_topics": ["Quantum computing"],
        "dictionary_override": "{Head} leverages quantum mechanics principles like superposition and entanglement to perform computations that can be exponentially faster for certain problems.",
        "booster_sentences": [
            "Quantum computers use qubits that can represent both 0 and 1 simultaneously, enabling parallelism in processing.",
            "{head_lower} holds promise for breakthroughs in cryptography, optimization, drug discovery, and complex system simulations."
        ],
        "force_wiki": True
    },
    {
        "keys": {"immune"},
        "display": "immune system",
        "lookup": "immune",
        "wiki_topics": ["Immune system", "Immunity (medical)"],
        "dictionary_override": "{Head} is the body's defense network that detects pathogens and protects you from disease.",
        "booster_sentences": [
            "At the cellular level it relies on white blood cells, antibodies, and signaling molecules to neutralize threats.",
            "Vaccination builds immunity by training memory cells to respond faster to specific antigens."
        ],
        "force_wiki": True
    },
    {
        "keys": {"climate_change", "global_warming"},
        "display": "climate change",
        "lookup": "climate change",
        "wiki_topics": ["Climate change", "Global warming"],
        "dictionary_override": "{Head} refers to long-term shifts in temperature and weather patterns, primarily driven by human activities like burning fossil fuels and deforestation.",
        "booster_sentences": [
            "It leads to rising sea levels, more extreme weather events, and disruptions to ecosystems and agriculture.",
            "Mitigation strategies include transitioning to renewable energy, enhancing energy efficiency, and protecting natural carbon sinks."
        ],
        "force_wiki": True
    },
    {
        "keys": {"cryptocurrency", "crypto"},
        "display": "cryptocurrency",
        "lookup": "cryptocurrency",
        "wiki_topics": ["Cryptocurrency"],
        "dictionary_override": "{Head} is a digital or virtual currency that uses cryptography for security and operates independently of a central authority.",
        "booster_sentences": [
            "Popular cryptocurrencies like Bitcoin and Ethereum rely on blockchain technology to enable peer-to-peer transactions.",
            "{head_lower} can be used for online purchases, investment, and as a means of transferring value across borders."
        ],
        "force_wiki": True
    },
    {
        "keys": {"sustainable_energy", "renewable_energy"},
        "display": "sustainable energy",
        "lookup": "sustainable energy",
        "wiki_topics": ["Sustainable energy", "Renewable energy"],
        "dictionary_override": "{Head} comes from sources that are naturally replenished, such as solar, wind, hydro, and geothermal power.",
        "booster_sentences": [
            "It reduces greenhouse gas emissions and dependence on finite fossil fuels.",
            "Adopting {head_lower} technologies can lead to long-term economic and environmental benefits."
        ],
        "force_wiki": True
    },
    {
        "keys": {"artificial_intelligence", "ai"},
        "display": "artificial intelligence",
        "lookup": "artificial intelligence",
        "wiki_topics": ["Artificial intelligence"],
        "dictionary_override": "{Head} is the simulation of human intelligence processes by machines, especially computer systems.",
        "booster_sentences": [
            "It encompasses machine learning, natural language processing, robotics, and computer vision.",
            "{head_lower} applications range from virtual assistants and recommendation systems to autonomous vehicles and advanced data analytics."
        ],
        "force_wiki": True
    },
    {
        "keys": {"data_science", "data_analysis"},
        "display": "data science",
        "lookup": "data science",
        "wiki_topics": ["Data science"],
        "dictionary_override": "{Head} combines statistics, computer science, and domain expertise to extract insights and knowledge from structured and unstructured data.",
        "booster_sentences": [
            "It involves data collection, cleaning, exploration, modeling, and visualization to inform decision-making.",
            "{head_lower} is widely used in business intelligence, healthcare, finance, and social sciences."
        ],
        "force_wiki": True
    },
    {
        "keys": {"internet_of_things", "iot"},
        "display": "Internet of Things",
        "lookup": "Internet of Things",
        "wiki_topics": ["Internet of Things"],
        "dictionary_override": "{Head} refers to the network of interconnected physical devices that collect and exchange data via the internet.",
        "booster_sentences": [
            "These devices range from everyday objects like thermostats and wearables to industrial sensors and smart city infrastructure.",
            "{head_lower} enables automation, remote monitoring, and improved efficiency across various sectors."
        ],
        "force_wiki": True
    },
    {
        "keys": {"cybersecurity", "information_security"},
        "display": "cybersecurity",
        "lookup": "cybersecurity",
        "wiki_topics": ["Cybersecurity"],
        "dictionary_override": "{Head} involves protecting computer systems, networks, and data from digital attacks, theft, and damage.",
        "booster_sentences": [
            "It encompasses practices like encryption, firewalls, intrusion detection, and user education to mitigate risks.",
            "{head_lower} is critical for safeguarding personal information, financial assets, and national security."
        ],
        "force_wiki": True
    },
    {
        "keys": {"virtual_reality", "vr"},
        "display": "virtual reality",
        "lookup": "virtual reality",
        "wiki_topics": ["Virtual reality"],
        "dictionary_override": "{Head} is a simulated experience that immerses users in a computer-generated environment, often using headsets and motion tracking.",
        "booster_sentences": [
            "It is used in gaming, training simulations, education, and therapeutic applications.",
            "{head_lower} can create realistic scenarios that enhance learning and engagement."
        ],
        "force_wiki": True
    },
    {
        "keys": {"augmented_reality", "ar"},
        "display": "augmented reality",
        "lookup": "augmented reality",
        "wiki_topics": ["Augmented reality"],
        "dictionary_override": "{Head} overlays digital information onto the real world, enhancing perception and interaction through devices like smartphones and AR glasses.",
        "booster_sentences": [
            "It is used in applications ranging from gaming and retail to industrial maintenance and medical visualization.",
            "{head_lower} enables users to access contextual information and interactive experiences in their physical environment."
        ],
        "force_wiki": True
    },
    {
        "keys": {"edge_computing"},
        "display": "edge computing",
        "lookup": "edge computing",
        "wiki_topics": ["Edge computing"],
        "dictionary_override": "{Head} processes data closer to its source rather than relying on centralized cloud servers, reducing latency and bandwidth use.",
        "booster_sentences": [
            "It is essential for applications requiring real-time processing, such as IoT devices, autonomous vehicles, and augmented reality.",
            "{head_lower} enhances performance and reliability by distributing computation across the network."
        ],
        "force_wiki": True
    },
    {
        "keys": {"5g", "5g_network"},
        "display": "5G network",
        "lookup": "5G network",
        "wiki_topics": ["5G"],
        "dictionary_override": "{Head} is the fifth generation of mobile network technology, offering faster speeds, lower latency, and greater connectivity than previous generations.",
        "booster_sentences": [
            "It supports advanced applications like IoT, augmented reality, and autonomous vehicles by enabling high data rates and massive device connections.",
            "{head_lower} utilizes technologies such as millimeter waves, small cells, and beamforming to enhance network performance."
        ],
        "force_wiki": True
    }
]

TOPIC_ALIASES = {}
for spec in TOPIC_ALIAS_DEFS:
    data = {k: v for k, v in spec.items() if k != "keys"}
    for key in spec["keys"]:
        TOPIC_ALIASES[key] = data

_ARXIV_ENTRY_CACHE = None

def cached_arxiv_entries():
    global _ARXIV_ENTRY_CACHE
    if _ARXIV_ENTRY_CACHE is None:
        _ARXIV_ENTRY_CACHE = fetch_arxiv_corpus(False)
    return _ARXIV_ENTRY_CACHE

def arxiv_highlight_lines(terms, limit=2):
    if not terms:
        return []
    terms_low = [t.lower() for t in terms if t]
    if not terms_low:
        return []
    lines, seen = [], set()
    for entry in cached_arxiv_entries():
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        blob = f"{title} {summary}".lower()
        if not any(term in blob for term in terms_low):
            continue
        if title in seen:
            continue
        seen.add(title)
        snippet = summary.split('. ')[0].strip() if summary else ""
        snippet = " ".join(snippet.split())
        snippet = snippet[:180] + ("…" if len(snippet) > 180 else "")
        line = f"arXiv highlight: {title.strip()}"
        if snippet:
            line += f" — {snippet}"
        lines.append(line)
        if len(lines) >= max(1, limit):
            break
    return lines

def token_source_counts(token: str, lex: dict) -> dict:
    return {
        "arxiv": lex.get("token_arxiv_count", {}).get(token, 0),
        "wiki":  lex.get("token_wiki_count", {}).get(token, 0),
        "story": lex.get("token_story_count", {}).get(token, 0)
    }

def dominant_source_for_token(token: str, lex: dict):
    counts = token_source_counts(token, lex)
    total = sum(counts.values())
    if total <= 0:
        return None
    source = max(counts, key=counts.get)
    ratio = counts[source] / total if total else 0.0
    return source, ratio

def is_definitional_prompt(prompt: str) -> bool:
    p = prompt.lower()
    triggers = [
        "what is", "what are", "what's", "define", "definition", "explain",
        "tell me about", "describe", "who is", "who are", "who was", "who were", "who's"
    ]
    return any(k in p for k in triggers)

def head_concept(tokens, lex, raw_tokens=None):
    """
    Prefer the user's true target concept.
    - If prompt is a 'how to learn X' style, pick X (not 'learn').
    - Otherwise, rank by TF–IDF with a role prior, but allow OOV tokens.
    """
    QUESTION_WORDS = {
        "what","who","whom","whose","which","why","how","when","where",
        "define","definition","explain","about","tell"
    }
    raw = raw_tokens or tokens
    raw_lower = [t.lower() for t in raw]

    # Alias-aware direct lookup: prefer longest matching span
    for span in range(len(raw_lower), 0, -1):
        for start in range(0, len(raw_lower) - span + 1):
            parts = raw_lower[start:start+span]
            if all(p in STOPWORDS for p in parts):
                continue
            key = "_".join(parts)
            if key in TOPIC_ALIASES:
                return key

    # Reconstruct a light string to detect 'how to' patterns
    text = " ".join(raw)
    low = text.lower()

    # --- 'who is/was' pattern: treat remainder as the head entity ---
    WHO_VERBS = {"is", "was", "are", "were"}
    if "who" in raw_lower:
        try:
            who_idx = raw_lower.index("who")
        except ValueError:
            who_idx = -1
        if who_idx != -1:
            j = who_idx + 1
            while j < len(raw_lower) and raw_lower[j] not in WHO_VERBS:
                j += 1
            if j < len(raw_lower) and raw_lower[j] in WHO_VERBS:
                tail = []
                for k in range(j + 1, len(raw)):
                    tok = re.sub(r"[^\w\-]", "", raw[k])
                    low_tok = raw_lower[k]
                    if not tok or low_tok in QUESTION_WORDS or low_tok in STOPWORDS:
                        continue
                    tail.append(tok)
                if tail:
                    combos = []
                    for i, tok in enumerate(tail):
                        combos.append(tok)
                        if i + 1 < len(tail):
                            combos.append(f"{tok}_{tail[i+1]}")
                        if i + 2 < len(tail):
                            combos.append(f"{tok}_{tail[i+1]}_{tail[i+2]}")

                    def who_combo_score(token: str) -> float:
                        parts = token.split("_")
                        tfidf_sum = sum(lex["tfidf"].get(p, 0.0) for p in parts)
                        idf_sum = sum(lex["idf"].get(p, 0.0) for p in parts)
                        length_bonus = 0.12 * len(parts)
                        multi_bonus = 0.35 * max(0, len(parts) - 1)
                        novelty = 0.25 if token not in lex.get("idf", {}) else 0.0
                        return tfidf_sum + 0.4 * idf_sum + length_bonus + multi_bonus + novelty

                    combos = [c for c in combos if c]
                    if combos:
                        combos.sort(key=who_combo_score, reverse=True)
                        return combos[0]

    # --- 'how to learn X' / 'how do i learn X' ---
    if "how" in low and "learn" in low:
        # take tokens after 'learn' as the head, ignoring stopwords and 'learn'
        try:
            li = [i for i,t in enumerate(raw) if t.lower()=="learn"][-1]
            tail = [t for t in raw[li+1:] if t.lower() not in QUESTION_WORDS and t.lower() not in STOPWORDS]
            if tail:
                # prefer the longest token (often the concrete noun) if TF–IDF ties
                tail.sort(key=lambda t: (lex["tfidf"].get(t, 0.0), len(t)), reverse=True)
                return tail[0]
        except Exception:
            pass

    if low.startswith("how to"):
        try:
            to_idx = [i for i, tok in enumerate(raw) if tok.lower() == "to"][-1]
        except IndexError:
            to_idx = None
        if to_idx is not None and to_idx + 1 < len(raw):
            tail = [raw[j] for j in range(to_idx + 1, len(raw))
                    if raw[j].lower() not in QUESTION_WORDS and raw[j].lower() not in STOPWORDS]
            ACTION_WORDS = {
                "check", "measure", "assess", "improve", "optimize", "build", "make", "create",
                "design", "plan", "develop", "use", "apply", "study", "analyze", "understand",
                "compare", "evaluate", "keep", "stay", "get", "become", "manage", "organize"
            }
            pruned = []
            for idx, tok in enumerate(tail):
                low_tok = tok.lower()
                if idx == 0 and (low_tok in ACTION_WORDS or low_tok.endswith("ing") or low_tok.endswith("ed")):
                    continue
                pruned.append(tok)
            if pruned:
                combos = []
                for i, tok in enumerate(pruned):
                    combos.append(tok)
                    if i + 1 < len(pruned):
                        combos.append(f"{tok}_{pruned[i+1]}")
                def combo_score(token: str) -> float:
                    parts = token.split("_")
                    tfidf_sum = sum(lex["tfidf"].get(p, 0.0) for p in parts)
                    idf_sum = sum(lex["idf"].get(p, 0.0) for p in parts)
                    novelty = 0.35 if token not in lex.get("idf", {}) else 0.0
                    length_bonus = 0.05 * sum(min(len(p), 10) for p in parts)
                    multi_bonus = 0.30 if "_" in token else 0.0
                    return tfidf_sum + 0.4*idf_sum + novelty + length_bonus + multi_bonus
                combos = [c for c in combos if c]
                if combos:
                    combos.sort(key=combo_score, reverse=True)
                    return combos[0]

    DEF_LEADS = {"what", "define", "definition", "explain", "describe", "tell", "tutor", "teach", "guide", "show"}
    for idx, tok in enumerate(raw_lower):
        if tok not in DEF_LEADS:
            continue
        tail = raw[idx+1:]
        if not tail:
            break
        filtered = [t for t in tail if t.lower() not in QUESTION_WORDS and t.lower() not in STOPWORDS]
        noise = {"difference", "between", "versus", "vs", "benefits", "advantages"}
        filtered = [t for t in filtered if t.lower() not in noise]
        if not filtered:
            continue
        combos = []
        L = len(filtered)
        for i, tok in enumerate(filtered):
            combos.append(tok)
            if i + 1 < L:
                combos.append(f"{tok}_{filtered[i+1]}")
            if i + 2 < L:
                combos.append(f"{tok}_{filtered[i+1]}_{filtered[i+2]}")
        def def_combo_score(token: str) -> float:
            parts = token.split("_")
            tfidf_sum = sum(lex["tfidf"].get(p, 0.0) for p in parts)
            idf_sum = sum(lex["idf"].get(p, 0.0) for p in parts)
            novelty = 0.25 if token not in lex.get("idf", {}) else 0.0
            length_bonus = 0.08 * len(parts)
            multi_bonus = 0.30 * max(0, len(parts) - 1)
            return tfidf_sum + 0.35*idf_sum + novelty + length_bonus + multi_bonus
        combos = [c for c in combos if c]
        if combos:
            combos.sort(key=def_combo_score, reverse=True)
            return combos[0]
        break

    # --- theorem / space combos for math prompts ---
    MATH_TAILS = {"theorem", "theorems", "space", "spaces"}
    best_combo = None
    best_combo_score = float("-inf")
    for idx, token in enumerate(raw_lower):
        if token not in MATH_TAILS:
            continue
        for span in range(4, 0, -1):
            start = idx - span + 1
            if start < 0:
                continue
            parts = raw[start:idx+1]
            parts_lower = [p.lower() for p in parts]
            if parts_lower and any(p in QUESTION_WORDS or p in STOPWORDS for p in parts_lower[:-1]):
                continue
            combo = "_".join(p.lower() for p in parts)
            tfidf_sum = sum(lex["tfidf"].get(p.lower(), 0.0) for p in parts)
            idf_sum = sum(lex["idf"].get(p.lower(), 0.0) for p in parts)
            length_bonus = 0.08 * len(parts)
            novelty_bonus = 0.25 if combo not in lex.get("idf", {}) else 0.0
            structure_bonus = 0.3 if token in {"theorem", "theorems"} else 0.2
            score = tfidf_sum + 0.35 * idf_sum + length_bonus + novelty_bonus + structure_bonus
            if score > best_combo_score:
                best_combo_score = score
                best_combo = combo
        # once we've considered combos ending here, continue
    if best_combo:
        return best_combo

    # --- "benefits/advantages/importance of" style prompts ---
    BENEFIT_KEYS = {"benefit", "benefits", "advantage", "advantages", "importance", "value", "values"}
    CONNECTORS = {"of", "to", "for", "from"}
    FILLER = {"staying", "being", "getting", "keeping", "stay", "get", "keep", "make"}
    PRONOUNS = {"my", "your", "our", "their", "his", "her", "its", "me", "you", "us", "them"}
    for i, tok in enumerate(raw_lower):
        if tok not in BENEFIT_KEYS:
            continue
        j = i + 1
        while j < len(raw_lower) and raw_lower[j] in CONNECTORS:
            j += 1
        tail = [raw[k] for k in range(j, len(raw)) if raw_lower[k] not in QUESTION_WORDS and raw_lower[k] not in STOPWORDS]
        if not tail:
            continue
        GENERIC_DOWNSCALE = {"body", "health", "thing", "stuff", "way", "people", "life"}
        candidates = []
        single_scores = []
        for idx, cand in enumerate(tail):
            c_low = cand.lower()
            if c_low in FILLER or c_low in PRONOUNS:
                continue
            tfidf = lex["tfidf"].get(cand, 0.0)
            idf = lex["idf"].get(cand, 0.0)
            role_bonus = 0.2 if token_role(cand, lex) in {"structure", "system", "process", "value", "agent"} else 0.0
            position_bonus = 1.2 / (1 + idx)
            morphology_bonus = 0.12 if c_low.endswith(("tion", "sion", "ing", "ism", "ity", "ous", "al", "ive", "ate", "ed")) else 0.0
            novelty_bonus = 0.18 if cand not in lex.get("idf", {}) else 0.0
            length_bonus = 0.03 * min(len(cand), 12)
            general_penalty = 0.55 if c_low in GENERIC_DOWNSCALE else 0.0
            citizen_bonus = 0.60 if ("citizen" in c_low or "citizenship" in c_low) else 0.0
            multi_bonus = 0.45 if "_" in cand else 0.0
            score = (0.55 * tfidf + 0.25 * idf + role_bonus + position_bonus +
                     morphology_bonus + novelty_bonus + length_bonus + citizen_bonus + multi_bonus - general_penalty)
            single_scores.append((score, cand, idx))

        candidates.extend((score, cand) for score, cand, _ in single_scores)

        # pairwise and triple combos
        def combo_stats(parts):
            tfidf_sum = sum(lex["tfidf"].get(part, 0.0) for part in parts)
            idf_sum = sum(lex["idf"].get(part, 0.0) for part in parts)
            role_bonus = sum(0.2 if token_role(part, lex) in {"structure", "system", "process", "value", "agent"} else 0.0 for part in parts)
            length_bonus = 0.05 * sum(min(len(part), 12) for part in parts)
            novelty_bonus = 0.18 * sum(1 for part in parts if part not in lex.get("idf", {}))
            citizen_bonus = 0.60 if any("citizen" in part.lower() or "citizenship" in part.lower() for part in parts) else 0.0
            multi_bonus = 0.55 * max(0, len(parts) - 1)
            return tfidf_sum, idf_sum, role_bonus, length_bonus, novelty_bonus, citizen_bonus, multi_bonus

        n_tail = len(tail)
        for i in range(n_tail):
            for span in (2, 3):
                if i + span > n_tail:
                    continue
                parts = tail[i:i+span]
                combo = "_".join(parts)
                tfidf_sum, idf_sum, role_bonus, length_bonus, novelty_bonus, citizen_bonus, multi_bonus = combo_stats(parts)
                position_bonus = 1.0 / (1 + i)
                score = (0.55 * tfidf_sum + 0.25 * idf_sum + role_bonus + position_bonus +
                         length_bonus + novelty_bonus + citizen_bonus + multi_bonus)
                candidates.append((score, combo))
        if not candidates:
            candidates = [(0.0, tail[-1])]
        candidates.sort(key=lambda x: x[0], reverse=True)
        primary = [c for c in candidates if c[1].lower() not in GENERIC_DOWNSCALE]
        pool = primary if primary else candidates
        chosen = pool[0][1]
        NORMALIZE = {
            "hydrated": "hydration",
            "hydrating": "hydration",
            "dehydration": "hydration",
            "sleep": "sleep",
        }
        return NORMALIZE.get(chosen.lower(), chosen)

    # --- generic case: filter question words; prefer conceptual roles ---
    filt = [t for t in tokens if t not in QUESTION_WORDS and t not in STOPWORDS]
    if not filt:
        return "concept"

    preferred_roles = {"structure","system","data","value","agent","process"}
    cands = [t for t in filt if lex["idf"].get(t) is not None and token_role(t, lex) in preferred_roles]
    if not cands:
        cands = filt

    def score(t):
        # allow OOV gracefully
        tfidf = lex["tfidf"].get(t, 0.0)
        idf   = lex["idf"].get(t, 0.0)
        roleb = 0.15 if token_role(t, lex) in preferred_roles else 0.0
        length = min(len(t), 12) * 0.01
        return tfidf + 0.25*idf + roleb + length

    return max(cands, key=score)

def make_definition_sentence(head: str) -> str:
    """
    Produce 'Existence is the condition ...' or 'X is a concept ...' without double articles.
    If the gloss already starts with a determiner (a/an/the), don't add another.
    """
    h = (head or "concept").replace("_", " ").strip()
    gloss = DEF_HINTS.get(h, None)
    if not gloss:
        gloss = "a concept that helps organize information and guide action"

    txt = gloss.strip()
    # If gloss begins with a/an/the, use as-is; otherwise prepend an appropriate indefinite article
    if re.match(r"^(a|an|the)\s+", txt, flags=re.I):
        rhs = txt
    else:
        first = txt.split()[0].lower()
        # simple vowel/initialism heuristic
        rhs = ("an " if first and first[0] in set("aeioufhlmnrsx") else "a ") + txt

    return f"{h[:1].upper() + h[1:]} is {rhs}."

def estimate_confidence(prompt: str, lex: dict, head_tok: str) -> float:
    """
    Returns a confidence score in [0,1] using only local signals
    (no network calls): coverage, anchor mass, wiki evidence, frequency, tf-idf.
    """
    q_all = advanced_tokenize(prompt)
    q_in  = [t for t in q_all if t in lex.get("idf", {})]
    coverage = len(q_in) / max(1, len(q_all))

    anchors = top_k_context(lex.get("cooc", {}), head_tok, k=24)
    anchor_mass = len([w for w in anchors if w in lex.get("idf", {})]) / 24.0

    wiki_ev = 1.0 if lex.get("token_wiki_count", {}).get(head_tok, 0) > 0 else 0.0
    freq    = min(1.0, (lex.get("unigram", {}).get(head_tok, 0) / 5.0))
    tfidf   = math.tanh(0.05 * float(lex.get("tfidf", {}).get(head_tok, 0.0)))

    # Smooth convex combination; tuned so mid–high kicks in sensibly.
    score = 0.25*coverage + 0.25*anchor_mass + 0.20*wiki_ev + 0.15*freq + 0.15*tfidf
    return max(0.0, min(1.0, score))

def creative_spin(sentence: str, head: str, anchors: list, persona: str, rng: random.Random, strength: float=0.40) -> str:
    """
    Light, grammar-safe paraphrasing:
      - punctuation (em-dash, colon), brief appositions
      - soft intensifiers ("at its core", "in plain terms")
      - optional short lead-ins ("Short answer:", "Think ...")
    Uses only common function words; content stays anchored.
    """
    s = sentence.strip()
    if strength <= 0.0 or len(s) < 6:
        return s

    # Build a compact anchor phrase if we have anchors
    A = [a.replace("_", " ").strip() for a in anchors if a and isinstance(a, str)]
    A = [a for a in A if a.lower() != head.lower()]
    tri = ""
    if A:
        uniq = []
        seen = set()
        for a in A:
            if a.lower() not in seen:
                seen.add(a.lower()); uniq.append(a)
            if len(uniq) >= 3: break
        if len(uniq) == 1:
            tri = uniq[0]
        elif len(uniq) == 2:
            tri = f"{uniq[0]} and {uniq[1]}"
        elif len(uniq) == 3:
            tri = f"{uniq[0]}, {uniq[1]}, and {uniq[2]}"

    # Small, safe transforms
    def dash_hint(x):    return f"{x.rstrip('.')} — think {tri}." if tri else x
    def colon_hint(x):   return f"{x.rstrip('.')} : {tri}." if tri else x
    def core_hint(x):    return re.sub(r"\bis\b", "is, at its core,", x, count=1) if " is " in f" {x} " else x
    def plain_hint(x):   return f"In plain terms, {x[0].lower() + x[1:]}" if x and x[0].isupper() else f"In plain terms, {x}"
    def short_ans(x):    return f"Short answer: {x}"
    def head_lead(x):    return f"{head}: {x}" if head and head.lower() not in x.lower() else x

    styles = [dash_hint, colon_hint, core_hint, plain_hint, short_ans, head_lead]

    # Persona-aware nudge: tutors prefer “In plain terms…”
    if (persona or "").lower() == "tutor":
        styles = [plain_hint, core_hint, dash_hint, short_ans, head_lead, colon_hint]

    # Pick 1–2 transforms; keep it brief and grammatical
    k1 = rng.choice(styles); out = k1(s)
    if rng.random() < min(0.5, strength) and len(out.split()) < 26:
        k2 = rng.choice([f for f in styles if f is not k1])
        out = k2(out)

    # Ensure head is present at least once
    if head and head.lower() not in out.lower():
        out = f"{head} — {out}"

    # Clean spacing & trailing punctuation
    out = re.sub(r"\s{2,}", " ", out).strip()
    if not out.endswith("."):
        out += "."
    return out

def paraphrase_with_roles(text: str,
                          head: str,
                          lex: dict,
                          persona: Optional[str],
                          rng: random.Random,
                          used_tokens: dict,
                          story_weight: float = 0.4,
                          persona_weight: float = 0.6,
                          source_pref: Optional[dict] = None,
                          spin: float = 0.40) -> str:
    """
    Turn a Wikipedia-like sentence into a fresh, grammatical line by:
      • using the sentence tokens as a query,
      • scoring a candidate set with your multi-head attention,
      • composing a clean SVO sentence via compose_sentence_weighted,
      • then lightly 'spinning' it for style (no word salad).
    """
    q = [t for t in advanced_tokenize(text) if t in lex.get("idf", {})][:32]
    if not q:
        return text.strip().rstrip(".") + "."

    # Candidate set: prioritize the query + a small slice from the global pool
    base_pool = candidate_pool(lex, top_k=1200)[:300]
    cand = list(dict.fromkeys(q + base_pool))  # dedupe, preserve order

    # Attention from query -> candidates
    attn = multihead_attention(q, cand, lex, dim=96, heads=4, seed=RANDOM_SEED)

    # Compose a clean sentence using your role map + persona weighting
    sent = compose_sentence_weighted(
        cand, lex, attn, used_tokens, rng,
        persona=persona,
        story_weight=story_weight,
        persona_weight=persona_weight,
        source_pref=source_pref
    )

    # Light stylistic spin anchored on the query tokens
    anchors = [a for a in q[:3] if a != head.lower()]
    return creative_spin(sent, head, anchors, persona or "", rng, strength=spin)

# =====[ SEMANTIC DICTIONARY GROUNDER ]========================================
DICT_CACHE = "fa_dict_cache.json"

def _load_dict_cache():
    try:
        if os.path.exists(DICT_CACHE):
            return json.load(open(DICT_CACHE, "r", encoding="utf-8"))
    except Exception:
        pass
    return {}

def _save_dict_cache(cache):
    try:
        with open(DICT_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def clean_dictionary_text(text: str) -> str:
    """Strip markup/footnotes and unescape entities for external glosses."""
    t = strip_html(text or "")
    t = html.unescape(t)
    t = re.sub(r"\s*\[[0-9]+\]\s*", " ", t)
    t = re.sub(r"\s*\{\{[^}]*\}\}\s*", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def normalize_gloss(text: str) -> str:
    return clean_dictionary_text(normalize_unicode((text or "").strip()))

def fetch_semantic_entry(term: str, lang: str = "en", use_cache: bool = True):
    """
    Get dictionary senses for `term` from trusted public sources:
      1) Wiktionary REST: https://{lang}.wiktionary.org/api/rest_v1/page/definition/<term>
      2) Free Dictionary API: https://api.dictionaryapi.dev/api/v2/entries/<lang>/<term>
    Returns a normalized dict or None.
    """
    term = (term or "").strip()
    if not term: return None

    cache = _load_dict_cache() if use_cache else {}
    k = f"{lang}:{term.lower()}"
    if use_cache and k in cache:
        return cache[k]

    entry = None
    # ---- 1) Wiktionary REST ----
    try:
        import urllib.parse
        url = f"https://{lang}.wiktionary.org/api/rest_v1/page/definition/{urllib.parse.quote(term)}"
        raw = http_get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        data = json.loads(raw.decode("utf-8", errors="ignore"))
        # Typical shape: {"en":[{"partOfSpeech":"noun","definitions":[{"definition":"...", "examples":[...], "synonyms":[...]}], ...}]}
        blocks = data.get(lang) or data.get("en") or []
        senses, pos_all, syn_all, ex_all = [], [], set(), []
        for b in blocks or []:
            pos = b.get("partOfSpeech","").strip()
            if pos: pos_all.append(pos)
            for d in b.get("definitions", []):
                defi = normalize_gloss(d.get("definition"))
                if defi: senses.append(defi)
                for ex in d.get("examples", []) or []:
                    exs = normalize_gloss(ex)
                    if exs: ex_all.append(exs)
                for sy in d.get("synonyms", []) or []:
                    s = normalize_unicode((sy or "").strip())
                    if s: syn_all.add(s)
        if senses:
            entry = {
                "source": "wiktionary",
                "url": f"https://{lang}.wiktionary.org/wiki/{term.replace(' ','_')}",
                "pos": sorted(set(pos_all)),
                "definitions": senses,
                "synonyms": sorted(syn_all),
                "examples": ex_all
            }
    except Exception:
        entry = None

    # ---- 2) Free Dictionary API (fallback) ----
    if entry is None:
        try:
            import urllib.parse
            url = f"https://api.dictionaryapi.dev/api/v2/entries/{lang}/{urllib.parse.quote(term)}"
            raw = http_get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
            data = json.loads(raw.decode("utf-8", errors="ignore"))
            senses, pos_all, syn_all, ex_all = [], [], set(), []
            if isinstance(data, list):
                for item in data:
                    for m in item.get("meanings", []) or []:
                        pos = m.get("partOfSpeech","").strip()
                        if pos: pos_all.append(pos)
                        for d in m.get("definitions", []) or []:
                            defi = normalize_gloss(d.get("definition"))
                            if defi: senses.append(defi)
                            for ex in d.get("example", [] if isinstance(d.get("example"), list) else [d.get("example")]):
                                if ex:
                                    ex_all.append(normalize_gloss(ex))
                            for sy in d.get("synonyms", []) or []:
                                s = normalize_unicode((sy or "").strip())
                                if s: syn_all.add(s)
            if senses:
                # FD API often cites Wiktionary as source; keep “dictionaryapi” to mark the route
                entry = {
                    "source": "dictionaryapi",
                    "url": f"https://{lang}.wiktionary.org/wiki/{term.replace(' ','_')}",
                    "pos": sorted(set(pos_all)),
                    "definitions": senses,
                    "synonyms": sorted(syn_all),
                    "examples": ex_all
                }
        except Exception:
            entry = None

    if entry and use_cache:
        cache[k] = entry
        _save_dict_cache(cache)
    return entry

def best_gloss_for_context(entry: dict,
                           prompt_tokens: list,
                           lex: dict,
                           context_words: Optional[Set[str]] = None,
                           head: Optional[str] = None) -> str:
    """
    Pick the best single-line definition for the user's prompt by attention-similarity,
    with POS-aware scoring so we avoid 'to …' verb senses for adjective/noun heads.
    """
    defs = (entry or {}).get("definitions") or []
    if not defs: return ""
    if not prompt_tokens:
        # prefer longer, contentful glosses
        defs = sorted(defs, key=lambda g: (-len(g.split()), g))
        return defs[0]

    def gloss_pos(g: str) -> str:
        s = (g or "").strip().lower()
        if s.startswith("to "): return "verb"
        if any(k in s for k in ("resistant", "not susceptible", "protected against", "immune to")):
            return "adjective"
        if s.startswith("the ") or " state of " in s or " system" in s:
            return "noun"
        return "unknown"

    # very light preference from the head word (e.g., 'immune' -> adjective/noun)
    head_low = (head or "").lower()
    if head_low in {"immune"}:
        pos_pref = ["adjective", "noun", "verb"]
    else:
        # default: definition-style prompts prefer noun then adjective
        pos_pref = ["noun", "adjective", "verb"]

    pos_bonus = {pos_pref[0]: 0.45, pos_pref[1]: 0.20}

    dim, heads = 96, 3
    scored = []
    ctx = {t.lower() for t in (context_words or prompt_tokens) if t}

    for g in defs:
        g_clean = clean_dictionary_text(g)
        g_toks = [t for t in advanced_tokenize(g_clean) if t in lex.get("idf", {})][:48]

        # attention score (same as before)
        if g_toks:
            fwd = mh_forward_scores(prompt_tokens[:16], g_toks, lambda t,d: embed_token(t, lex, d), dim=dim, heads=heads, seed=1729)
            bwd = mh_backward_scores(prompt_tokens[:16], g_toks, lambda t,d: embed_token(t, lex, d), dim=dim, heads=heads, seed=7331)
            blend = mh_posthoc_blend(fwd, bwd, alpha=0.65)
            score = sum(blend.values())/max(1,len(blend))
        else:
            score = 0.0

        # POS preference
        gp = gloss_pos(g_clean)
        score += pos_bonus.get(gp, 0.0)

        # penalize ultra-short or meta-ish lines
        words = len(g_clean.split())
        if words < 5:
            score -= 0.45

        scored.append((score, g_clean))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Prefer concise but meaningful: <= 40 words
    for _, g in scored:
        if len(g.split()) <= 40:
            return g
    return scored[0][1]

def render_faq_lines(faq, limit):
    raise NotImplementedError

def render_bibliography_lines(entries, max_lines):
    raise NotImplementedError

def transfinite_generate(prompt, lex, sentences=5, depth=3, persona=None,
                         story_weight=0.4, persona_weight=0.6, seed=RANDOM_SEED,
                         attn_blend_alpha=0.65, fair_alpha=0.10, fair_lambda=0.25):
    """
    Definition-accurate generator with dictionary/Wikipedia grounding,
    semantic-density gating, and an improved HEAD mechanism.

    What's new (concise):
      • HEAD selection = fixed-point style refinement with mutual-coherence penalty:
          - Start from a seed head; iteratively re-score candidates against a context centroid
            plus dictionary/Wikipedia gloss features; damp large jumps (nonexpansive-like).
          - Penalize high mutual coherence (too-synonymous clusters) and genericity; reward IDF,
            gloss clarity, and overlap with query/topic focus.
      • Stronger anti–word-salad & grammar polish:
          - Quality gates use content ratio, verb presence, head relevance, salad score, and length.
          - Auto-split long clauses; normalize punctuation; de-filler; enforce sentence case.
      • Safe fallbacks & deterministic behavior (even if seed=None).
      • Compatible with existing helpers; drop-in replacement.

    Inspired by fixed-point/low-coherence ideas (Alpay & Alakkad, 2025) and semantic-density
    scoring in recent LLM work.  (See chat context for citations.)
    External helpers expected (unchanged):
      advanced_tokenize, STOPWORDS, head_concept, TOPIC_ALIASES,
      sentence_vector, cosine, top_k_context, fetch_semantic_entry, best_gloss_for_context,
      clean_dictionary_text, search_wikipedia_titles, fetch_wiki_topic, paraphrase_with_roles,
      candidate_pool, mh_forward_scores, mh_backward_scores, mh_posthoc_blend, embed_token,
      arxiv_highlight_lines, render_bibliography_lines, render_faq_lines, is_definitional_prompt,
      make_definition_sentence, augment_lexicon_with_wikipedia, dominant_source_for_token.
    """
    # ---------- setup ----------
    rng = random.Random(RANDOM_SEED if seed is None else seed)
    p = (prompt or "").strip()
    p_low = p.lower()
    sentences = max(1, int(sentences or 1))

    outputs, vecs, used = [], [], set()
    used_tokens_local = defaultdict(int)
    used_words = set()

    # ---------- small utilities ----------
    def clean_period(s: str) -> str:
        s = re.sub(r"\s+", " ", (s or "")).strip()
        if not s:
            return ""
        s = re.sub(r"\s+([,;:.!?])", r"\1", s)                # trim spaces before punct
        s = re.sub(r"([.!?]){2,}$", r"\1", s)                 # compress terminal repeats
        return s if re.search(r"[.!?]$", s) else s + "."

    def semvec(s: str): return sentence_vector(s, lex, dim=96)
    def has_word(w: str) -> bool: return re.search(rf"\b{re.escape(w)}\b", p_low) is not None
    def safe_fetch_semantic_entry(term):
        try: return fetch_semantic_entry(term)
        except Exception: return None
    def safe_fetch_wiki_topic(title):
        try: return fetch_wiki_topic(title)
        except Exception: return None

    # ---------- semantic-density / anti-salad gates ----------
    VERB_HINTS = {
        "is","are","was","were","be","being","been","has","have","had","do","does","did",
        "use","build","create","provide","enable","allow","measure","compare","analyze",
        "support","improve","reduce","increase","optimize","model","learn","predict","estimate",
        "fetch","choose","explain","define","show","run","train","evaluate","deploy","calibrate",
        "hydrate","protect","sustain","anchor","stabilize","record","note","schedule","review"
    }
    FILLER_MAP = {
        "in order to": "to", "due to the fact that": "because", "utilize": "use",
        "leverage": "use", "and also": "and", "so therefore": "therefore", "advance forward": "advance",
        "basically": "", "actually": "", "quite": "", "rather": "", "really": "", "very": "",
        "kinda": "", "sorta": "", "paradigm": "", "state of the art": "", "cutting edge": "",
        "synergy": "", "holistic": "", "robustly": ""
    }
    FILLER_PAT = re.compile("|".join(rf"\b{re.escape(k)}\b" for k in FILLER_MAP), re.I)

    def _contains_verb(tokens: list[str]) -> bool:
        low = set(t.lower() for t in tokens)
        return any(v in low for v in VERB_HINTS)

    def _type_token_ratio(tokens: list[str]) -> float:
        toks = [t.lower() for t in tokens if t]
        return (len(set(toks)) / max(1, len(toks))) if toks else 0.0

    def _content_ratio(tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        content = [t for t in tokens if t.lower() not in STOPWORDS and re.search(r"[A-Za-z0-9]", t)]
        return len(content) / max(1, len(tokens))

    def _avg_word_len(tokens: list[str]) -> float:
        words = [t for t in tokens if re.search(r"[A-Za-z]", t)]
        return (sum(len(w) for w in words) / max(1, len(words))) if words else 0.0

    def word_salad_score(text: str) -> float:
        """Lower is better. 0.0–0.3 good, 0.3–0.6 risky, >0.6 reject."""
        tokens = advanced_tokenize(text)
        if not tokens: return 1.0
        ttr = _type_token_ratio(tokens)
        content = _content_ratio(tokens)
        verbless = 1.0 if not _contains_verb(tokens) else 0.0
        punct_density = len(re.findall(r"[,:;)(\-/]", text)) / max(1, len(tokens))
        adverb_run = 1.0 if re.search(r"\b\w+ly\b(?:\s+\w+ly\b){2,}", text, flags=re.I) else 0.0
        noun_pile = 1.0 if (content > 0.8 and not _contains_verb(tokens) and len(tokens) > 8) else 0.0
        too_long = 1.0 if len(tokens) > 38 else 0.0
        score = (0.30*ttr + 0.25*verbless + 0.15*punct_density + 0.10*adverb_run +
                 0.10*noun_pile + 0.10*too_long)
        if _avg_word_len(tokens) > 7.2: score += 0.05
        return min(1.0, max(0.0, score))

    def grammar_polish(text: str) -> str:
        s = " " + (text or "") + " "
        s = re.sub(r"\s+", " ", s)
        s = FILLER_PAT.sub(lambda m: FILLER_MAP.get(m.group(0).lower(), ""), s)
        s = re.sub(r"\bthat that\b", "that", s, flags=re.I)
        s = re.sub(r"\s+([,;:.!?])", r"\1", s)
        s = re.sub(r"([(\[]) ", r"\1", s); s = re.sub(r" ([)\]])", r"\1", s)
        s = re.sub(r"([,;:.!?])\1+", r"\1", s)
        s = s.strip()
        if s: s = s[0].upper() + s[1:]
        return clean_period(s)

    def split_overlong(text: str) -> list[str]:
        toks = advanced_tokenize(text)
        if len(toks) <= 34: return [text]
        parts = re.split(r"\s*[;—–-]\s*|\s*,\s*", text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) <= 1: return [text]
        chunks, cur = [], []
        for part in parts:
            ptoks = advanced_tokenize(part)
            if len(cur) + len(ptoks) <= 24: cur += ptoks
            else:
                if cur: chunks.append(" ".join(cur))
                cur = ptoks
        if cur: chunks.append(" ".join(cur))
        return [grammar_polish(c) for c in chunks if c]

    def head_relevance(text: str, head_token: str) -> float:
        try:
            sv = semvec(text); hv = embed_token(head_token, lex, 96)
            return max(0.0, min(1.0, (1.0 + cosine(sv, hv)) / 2.0))
        except Exception:
            return 0.5

    def quality_score(text: str, head_token: str) -> float:
        tokens = advanced_tokenize(text)
        if not tokens: return 0.0
        cr = _content_ratio(tokens); salad = word_salad_score(text)
        rel = head_relevance(text, head_token); verb = 1.0 if _contains_verb(tokens) else 0.0
        cr_pen = 0.15 if cr < 0.35 else (0.10 if cr > 0.80 else 0.0)
        base = (0.35*rel) + (0.25*cr) + (0.20*verb) + (0.20*(1.0 - salad)) - cr_pen
        if len(tokens) > 34: base -= 0.05
        return max(0.0, min(1.0, base))

    def polish_and_validate(text: str, head_token: str) -> list[str]:
        if not text: return []
        s = grammar_polish(text)
        variants = split_overlong(s)
        clean_variants = []
        for v in variants:
            if quality_score(v, head_token) >= 0.55 and word_salad_score(v) <= 0.45:
                clean_variants.append(v)
        return clean_variants

    # ---------- improved HEAD mechanism (fixed-point refinement + mutual coherence) ----------
    q_raw = advanced_tokenize(prompt)
    q_toks = [t for t in q_raw if t not in STOPWORDS and len(t) > 1]

    def _idf(t: str) -> float: return float(lex.get("idf", {}).get(t, 0.0))
    def _tfidf(t: str) -> float: return float(lex.get("tfidf", {}).get(t, 0.0))
    def _vec(t: str):
        try: return embed_token(t, lex, 96)
        except Exception: return None

    def _ngrams(tokens: list[str], n: int) -> list[str]:
        outs = []
        for i in range(len(tokens)-n+1):
            gram = " ".join(tokens[i:i+n]).strip()
            if gram and all(len(tok) > 1 for tok in tokens[i:i+n]):
                outs.append(gram)
        return outs

    def _lex_has(t: str) -> bool:
        return t in lex.get("idf", {}) or t.replace(" ", "_") in lex.get("idf", {})

    def _as_lex_tok(t: str) -> str:
        return t if t in lex.get("idf", {}) else t.replace(" ", "_")

    def _coherence_penalty(cands: list[str], key: str) -> float:
        """Mutual coherence: average similarity of key to others; higher -> more penalty."""
        vk = _vec(_as_lex_tok(key))
        if vk is None or not cands: return 0.0
        sims, count = 0.0, 0
        for c in cands:
            if c == key: continue
            vc = _vec(_as_lex_tok(c))
            if vc is None: continue
            sims += max(0.0, cosine(vk, vc)); count += 1
        if count == 0: return 0.0
        mu = sims / count
        # map [0,1] -> [0,0.35] approximately
        return 0.35 * min(1.0, mu)

    def _context_centroid(tokens: list[str]) -> Optional[list[float]]:
        vecs = []
        for t in tokens:
            tt = _as_lex_tok(t)
            v = _vec(tt)
            if v is not None: vecs.append(v)
        if not vecs: return None
        # simple average
        n = len(vecs)
        return [sum(col)/n for col in zip(*vecs)]

    def _sim_to_centroid(t: str, centroid) -> float:
        try:
            vt = _vec(_as_lex_tok(t))
            if vt is None or centroid is None: return 0.0
            # cosine(u, v) expects same shape vectors in your helper
            return max(0.0, cosine(vt, centroid))
        except Exception:
            return 0.0

    def _gloss_features(term: str, q_tokens: list[str]) -> tuple[float, float]:
        """(clarity_score, overlap_score) from dictionary gloss."""
        entry = safe_fetch_semantic_entry(term) or safe_fetch_semantic_entry(term.title())
        if not entry: return (0.0, 0.0)
        gloss = best_gloss_for_context(entry, q_tokens, lex) or ""
        gloss = clean_dictionary_text(gloss or "").strip()
        if not gloss: return (0.0, 0.0)
        wc = len(gloss.split())
        clarity = 1.0 if 7 <= wc <= 28 else (0.7 if 5 <= wc <= 36 else 0.3)
        toks = set(advanced_tokenize(gloss))
        overlap = len(toks.intersection(set(q_tokens))) / max(1, len(set(q_tokens)))
        return (clarity, overlap)

    def _generic_penalty(t: str) -> float:
        freq = float(lex.get("unigram", {}).get(t, 0.0))
        idf = _idf(t)
        penalty = 0.0
        if idf < 0.8: penalty += 0.25
        if freq > 20: penalty += 0.15
        if len(t) <= 2: penalty += 0.25
        return min(0.6, penalty)

    def _candidate_pool(seed: str) -> list[str]:
        base = []
        # include content tokens, and common n-grams
        base += [t for t in q_toks if _lex_has(t)]
        base += [ng for n in (2,3) for ng in _ngrams([t for t in q_toks if t not in STOPWORDS], n) if _lex_has(ng)]
        # plus neighbors by co-occurrence
        base += [w for w in top_k_context(lex.get("cooc", {}), seed, k=24) if _lex_has(w)]
        # plus the seed itself
        if _lex_has(seed): base.append(seed)
        # dedup preserve order
        seen, out = set(), []
        for b in base:
            key = b.lower()
            if key not in seen:
                seen.add(key); out.append(b)
        # protect against tiny pools
        if not out and _lex_has("concept"):
            out = ["concept"]
        return out[:64] if out else []

    def _score_candidate(term: str, centroid, pool: list[str], prev_head: Optional[str]) -> float:
        """Blend relevance, IDF/TF–IDF, gloss signals, and mutual-coherence penalty."""
        rel = _sim_to_centroid(term, centroid)
        idf = _idf(_as_lex_tok(term))
        tfidf = _tfidf(_as_lex_tok(term))
        cl, ov = _gloss_features(term, q_toks)
        coh_pen = _coherence_penalty(pool, term)
        gen_pen = _generic_penalty(term)
        # discourage big jumps relative to previous head (nonexpansive-like damping)
        jump_pen = 0.0
        if prev_head and term != prev_head:
            try:
                v1 = _vec(_as_lex_tok(term)); v0 = _vec(_as_lex_tok(prev_head))
                if v1 is not None and v0 is not None:
                    sim = max(0.0, cosine(v1, v0))
                    if sim < 0.25: jump_pen = 0.12
                    elif sim < 0.45: jump_pen = 0.06
            except Exception:
                pass
        # scale IDF/TF–IDF smoothly
        idf_s = math.tanh(0.15*idf)
        tfidf_s = math.tanh(0.08*tfidf)
        # final score
        score = (0.46*rel + 0.16*idf_s + 0.10*tfidf_s + 0.10*cl + 0.06*ov
                 + 0.18*(1.0 - coh_pen) - 0.14*gen_pen - jump_pen)
        return float(score)

    def refine_head(seed: str) -> str:
        """
        Fixed-point style refinement:
          - iterate: centroid -> re-score -> pick argmax with damping
          - stop on stability (no change) or small improvement window
        """
        head = seed
        prev_best, prev_score = None, -1e9
        stable_steps = 0
        for it in range(1, 6):  # small, safe number of iterations
            pool = _candidate_pool(head)
            # centroid mixes context + top-3 gloss tokens of current head
            extra = []
            cl, ov = _gloss_features(head, q_toks)
            entry = safe_fetch_semantic_entry(head) or safe_fetch_semantic_entry(head.title())
            if entry:
                g = best_gloss_for_context(entry, q_toks, lex) or ""
                g_toks = advanced_tokenize(clean_dictionary_text(g))[:16]
                extra = [t for t in g_toks if t not in STOPWORDS][:8]
            centroid = _context_centroid([*q_toks[:12], *extra]) or _context_centroid(q_toks[:12])
            # score
            scores = [(t, _score_candidate(t, centroid, pool, head)) for t in pool]
            scores.sort(key=lambda x: x[1], reverse=True)
            best, best_score = (scores[0] if scores else (head, prev_score))
            # mild inertia: if best is too generic or a near-synonym cluster peak, consider runner-up
            if len(scores) > 1:
                _, runner_score = scores[1]
                if best_score - runner_score < 0.03 and _coherence_penalty(pool, best) > 0.25:
                    best, best_score = scores[1]
            # check stability
            if best == prev_best or abs(best_score - prev_score) < 1e-3:
                stable_steps += 1
            else:
                stable_steps = 0
            prev_best, prev_score = best, best_score
            head = best
            if stable_steps >= 1:  # two consecutive near-equal choices -> stop
                break
        # fall back if head got weird
        if not head or len(head) < 2: head = seed or "concept"
        return head

    # seed from old heuristic then refine
    seed_head = head_concept(q_toks, lex, raw_tokens=q_raw) or "concept"
    head_tok = refine_head(seed_head)
    Head_orig = head_tok.replace("_", " ").strip()

    # ---------- alias & topic scaffolding (same as before) ----------
    alias = TOPIC_ALIASES.get(head_tok.lower(), {})
    Head = alias.get("display", Head_orig)
    Head_cap = Head[:1].upper() + Head[1:]
    Head_lower = Head.lower()
    lookup_term = alias.get("lookup", Head)
    alias_boosters = list(alias.get("booster_sentences", []) or [])
    alias_wiki_topics = list(alias.get("wiki_topics", []) or [])
    alias_dict_override = alias.get("dictionary_override")
    alias_force_wiki = bool(alias.get("force_wiki", False))
    alias_howto_steps = list(alias.get("howto_steps", []) or [])
    alias_source_pref = dict(alias.get("preferred_sources", {}) or {})
    alias_arxiv_terms = list(alias.get("arxiv_terms", []) or [])
    alias_geo_terms = list(alias.get("geo_terms", []) or [])
    alias_bib_entries = list(alias.get("bib_entries", []) or [])
    alias_faq = list(alias.get("faq", []) or [])
    alias_definition = alias.get("definition")
    pending_alias_intro = None

    # ---------- coverage & prompt intent ----------
    def coverage_low(h: str, L: dict) -> bool:
        oov = h not in L.get("idf", {})
        freq = L.get("unigram", {}).get(h, 0)
        anchors = [w for w in top_k_context(L.get("cooc", {}), h, k=24) if w in L.get("idf", {})]
        wiki_ev = L.get("token_wiki_count", {}).get(h, 0)
        story_ev = L.get("token_story_count", {}).get(h, 0)
        base = oov or (freq < 2 and not anchors) or (len(anchors) < 4)
        return base or (freq <= 3 and (wiki_ev + story_ev) == 0)

    benefit_cues = {"benefit","benefits","advantage","advantages","importance","value"}
    is_benefit_prompt = any(cue in p_low for cue in benefit_cues)
    person_query = bool(re.search(r"\bwho\s+(?:is|are|was|were)\b", p_low))
    how_to_prompt = bool(re.search(r"\bhow\s+(?:to|do|can|should)\b", p_low)) or p_low.startswith("how to")

    # ---- robust comparison extraction (vs/versus/between/compare) ----
    def extract_comparison_targets(text: str) -> list[str]:
        if not text: return []
        seg = ""
        m = re.search(r"\bbetween\b(.+)$", text, flags=re.I)
        if m: seg = m.group(1)
        else:
            m2 = re.search(r"\bcompare\b(?:.+?)\b(?:to|and|vs\.?|versus)\b(.+)$", text, flags=re.I)
            if m2: seg = m2.group(1)
            else:
                m3 = re.search(r"\b(.+?)\s+(?:vs\.?|versus)\s+(.+)$", text, flags=re.I)
                if m3: seg = f"{m3.group(1)} and {m3.group(2)}"
        if not seg: return []
        seg = re.split(r"\?$", seg.strip())[0]
        parts = re.split(r"\band\b|,|/|;|\|", seg, flags=re.I)
        cleaned = [re.sub(r"[^\w\s-]", "", p).strip() for p in parts]
        cleaned = [c for c in cleaned if c]
        if not cleaned: return []
        tail = cleaned[-1]; tail_low = tail.lower(); shared = ""
        if " " in tail and any(w in tail_low for w in ("intelligence","learning","energy","power")):
            shared = " " + tail.split()[-1].lower()
        targets, seen = [], set()
        for item in cleaned:
            cand = (item + shared).strip() if shared and shared.strip() not in item.lower() else item.strip()
            key = cand.lower()
            if key not in seen:
                seen.add(key); targets.append(cand)
        return targets if len(targets) >= 2 else []

    def comparison_response(targets: list[str], max_lines: int) -> str:
        lines = []
        def short_summary(term: str) -> str:
            summary = ""
            entry = safe_fetch_semantic_entry(term.title())
            if entry:
                gloss = best_gloss_for_context(entry, q_toks, lex)
                if gloss: summary = clean_dictionary_text(gloss)
            if summary: return summary
            for title in [term, term.title()] + search_wikipedia_titles(term, limit=4):
                info = safe_fetch_wiki_topic(title)
                if not info: continue
                txt = clean_dictionary_text(info.get("summary", ""))
                if txt:
                    for part in re.split(r"(?<=[.!?])\s+", txt):
                        if len(part.split()) >= 5: return part.strip()
            return "a topic where I lack a focused summary"
        for term in targets[:max(1, max_lines-1)]:
            label = term.strip().title()
            candidate = f"{label}: {short_summary(term).strip().rstrip('.')}."
            polished = polish_and_validate(candidate, head_tok)
            if polished: lines.extend(polished[:1])
        if len(lines) < max_lines:
            lows = [t.lower() for t in targets]; bits = []
            if any("animal" in t for t in lows): bits.append("animal intelligence leans on embodied perception and evolved instincts")
            if any("human" in t for t in lows): bits.append("human intelligence layers symbolic language, culture, and self-reflection")
            if any("artificial" in t for t in lows): bits.append("artificial intelligence depends on data-driven algorithms running on hardware")
            if bits:
                clause = bits[0] if len(bits)==1 else (f"{bits[0]} while {bits[1]}" if len(bits)==2 else ", ".join(bits[:2]) + f", and {bits[2]}")
                tail = polish_and_validate(f"Key difference: {clause}.", head_tok)
                if tail: lines.extend(tail[:1])
        return " ".join(lines[:max_lines]) if lines else " ".join(targets[:2]) + " differ in purpose and mechanism."

    comparison_targets = extract_comparison_targets(prompt)
    if comparison_targets:
        return comparison_response(comparison_targets, max(2, sentences))

    # ---------- wiki augmentation if needed ----------
    wiki_entries = []
    initial_cov_low = coverage_low(head_tok, lex)
    cov_low = initial_cov_low
    if initial_cov_low or is_benefit_prompt or person_query or alias_force_wiki:
        lex = augment_lexicon_with_wikipedia(prompt, lex, head_tok, max_pages=5)
        wiki_entries = list(lex.pop("_wiki_last_entries", []))
        cov_low = coverage_low(head_tok, lex)

    keep_short_terms = {"water","fluid","cells","blood","brain","heart","skin","ions"}
    wiki_focus_counter = Counter()
    alias_wiki_summaries = []
    fetch_alias_wiki = bool(alias_wiki_topics) and not (alias_dict_override and alias_boosters)
    if fetch_alias_wiki:
        for title in alias_wiki_topics:
            info = safe_fetch_wiki_topic(title)
            if info and info.get("summary"):
                alias_wiki_summaries.append(clean_dictionary_text(info.get("summary", "")))

    def consider_focus_token(tok: str, weight: float = 1.0):
        if not tok or len(tok) <= 2 or tok in STOPWORDS: return
        if lex.get("idf", {}).get(tok, 0.0) < 1.0: return
        if len(tok) < 5 and tok not in keep_short_terms: return
        bias_weight = 1.0
        if alias_source_pref:
            ts = lex.get("token_story_count", {}).get(tok, 0)
            ta = lex.get("token_arxiv_count", {}).get(tok, 0)
            tw = lex.get("token_wiki_count", {}).get(tok, 0)
            total = ts + ta + tw
            if total:
                if alias_source_pref.get("arxiv"): bias_weight += alias_source_pref["arxiv"] * (ta/total)
                if alias_source_pref.get("story"): bias_weight += 0.5 * alias_source_pref["story"] * (ts/total)
                if alias_source_pref.get("wiki"):  bias_weight += 0.2 * alias_source_pref["wiki"]  * (tw/total)
        wiki_focus_counter[tok] += float(weight * bias_weight)

    for we in wiki_entries:
        for t in advanced_tokenize(we.get("summary", "")):
            consider_focus_token(t)
    for summary in alias_wiki_summaries:
        for tok in advanced_tokenize(summary):
            consider_focus_token(tok, weight=1.2)

    # ---------- identity (unchanged behavior, but polished) ----------
    is_who = has_word("who")
    has_you = has_word("you"); has_i = has_word("i")
    has_we = has_word("we") or has_word("us") or has_word("our")
    is_what_identity = bool(re.search(r"\bwhat\s+(?:are|is)\s+(?:you|we)\b", p_low))
    identity_prompt = (is_who and (has_you or has_i or has_we)) or (is_what_identity and (has_you or has_we))
    if identity_prompt:
        ENGINE = "Faruk Alpay Persona + Reality Dense Reflective Engine (Bias-aware + C3F)"
        lines = ([
            f"I'm the {ENGINE}, a retrieval-grounded writing engine.",
            "I grow a lexicon from arXiv, Wikipedia, and persona stories, then draft short reflective answers.",
            "Each output ends with a quick check so you can restate the idea and anchor it to your field."
        ] if (is_what_identity and has_you and not has_we) else
        [
            f"I'm the {ENGINE}, a retrieval-grounded writing engine.",
            "I build a lexicon from arXiv, Wikipedia, and persona stories, then compose short, on-topic answers.",
            "I use forward/backward attention with C³F calibration; when unsure, I fetch Wikipedia first."
        ] if has_you and not (has_i or has_we) else
        [
            "I don’t have personal data about you unless you share it here.",
            "Tell me your goals and constraints; I’ll adapt the plan."
        ] if has_i and not (has_you or has_we) else
        [
            "We’re a team: you steer; I retrieve facts, plan steps, and draft concise outputs.",
            "Together we iterate quickly—grounded details first, short checks to stay on track."
        ])
        outs = []
        for s in lines[:max(1, sentences)]:
            outs.extend(polish_and_validate(s, head_tok))
        return " ".join(outs[:sentences]) if outs else " ".join(lines[:sentences])

    # ---------- dictionary-led path if definitional or low coverage ----------
    should_use_dict = (not how_to_prompt) and (is_definitional_prompt(prompt) or initial_cov_low)
    outputs, vecs, used = [], [], set()
    used_words = set()
    dictionary_used = False
    dictionary_entry = None

    if should_use_dict:
        entry = safe_fetch_semantic_entry(lookup_term)
        alias_line = alias_dict_override.format(Head=Head_cap, head_lower=Head_lower,
                                               Head_lower=Head_lower, original=Head_orig) if alias_dict_override else None
        if entry:
            dictionary_entry = entry
            context_words = set(t.lower() for t in q_raw if len(t) > 1)
            context_words.update({Head_lower, Head_orig.lower()})
            for e in wiki_entries:
                context_words.update(advanced_tokenize(e.get("summary", ""))[:32])
            gloss = best_gloss_for_context(entry, q_toks, lex, context_words=context_words, head=Head)
        else:
            gloss = ""

        lead_candidate, focus_source = None, None
        if alias_line:
            lead_candidate = clean_period(clean_dictionary_text(alias_line)); focus_source = alias_line
        elif gloss:
            if re.match(rf"(?i)\b{re.escape(Head)}\b\s+is\b", gloss):
                lead_candidate = clean_period(clean_dictionary_text(gloss)); focus_source = gloss
            else:
                composed = f"{Head_cap} is {gloss}"
                lead_candidate = clean_period(clean_dictionary_text(composed)); focus_source = gloss

        if lead_candidate:
            for s0 in polish_and_validate(lead_candidate, head_tok):
                if s0.lower() in used: continue
                v = semvec(s0); outputs.append(s0); vecs.append(v); used.add(s0.lower())
                if focus_source:
                    for tok in advanced_tokenize(focus_source): consider_focus_token(tok, weight=0.6)
                used_words.update(advanced_tokenize(s0)); dictionary_used = True
                break

        if entry and not alias_dict_override and entry.get("examples") and len(outputs) < sentences:
            ex_raw = clean_dictionary_text(entry["examples"][0])
            polished = polish_and_validate(f"Example: {ex_raw}", head_tok)
            if polished:
                ex = polished[0]
                if ex.lower() not in used:
                    v = semvec(ex)
                    # check similarity threshold
                    if not any(cosine(v, u) > 0.88 for u in vecs):
                        outputs.append(ex); vecs.append(v); used.add(ex.lower())
                        used_words.update(advanced_tokenize(ex))

    if alias_dict_override and not dictionary_used:
        alias_line_raw = alias_dict_override.format(Head=Head_cap, head_lower=Head_lower,
                                                    Head_lower=Head_lower, original=Head_orig)
        lead_polished = polish_and_validate(alias_line_raw, head_tok)
        if how_to_prompt:
            pending_alias_intro = lead_polished[0] if lead_polished else None
            for tok in advanced_tokenize(alias_line_raw): consider_focus_token(tok, weight=0.6)
        elif lead_polished:
            s0 = lead_polished[0]
            if s0.lower() not in used:
                v = semvec(s0); outputs.append(s0); vecs.append(v); used.add(s0.lower())
                for tok in advanced_tokenize(alias_line_raw): consider_focus_token(tok, weight=0.6)
                used_words.update(advanced_tokenize(s0)); dictionary_used = True

    wiki_summary_pending = []

    def harvest_summary(text: str):
        if not text: return
        for part in re.split(r"(?<=[.!?])\s+", text):
            chunk = part.strip()
            if len(chunk.split()) < 6: continue
            cands = []
            for spin_val in (0.35, 0.25):
                try:
                    para = paraphrase_with_roles(
                        chunk, Head_cap, lex, persona, rng,
                        used_tokens=used_tokens_local, story_weight=story_weight,
                        persona_weight=persona_weight, source_pref=alias_source_pref, spin=spin_val
                    )
                    cands.append(para)
                except Exception:
                    pass
            cands.append(chunk)
            scored = []
            for c in cands:
                for pl in polish_and_validate(c, head_tok):
                    scored.append((quality_score(pl, head_tok), pl))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored: wiki_summary_pending.append(scored[0][1])
            if len(wiki_summary_pending) >= max(2, sentences - len(outputs)): break

    if len(outputs) < sentences:
        for summary in alias_wiki_summaries:
            if len(wiki_summary_pending) >= max(2, sentences - len(outputs)): break
            harvest_summary(summary)
        skip_generic_wiki = alias_dict_override and (alias_wiki_summaries or alias_boosters)
        if not skip_generic_wiki and len(wiki_summary_pending) < max(2, sentences - len(outputs)):
            for e in wiki_entries:
                if len(wiki_summary_pending) >= max(2, sentences - len(outputs)): break
                harvest_summary(clean_dictionary_text(e.get("summary", "")))

    wiki_focus_tokens = set(tok for tok in wiki_focus_counter if tok in lex.get("idf", {}))
    wiki_focus_order = [tok for tok, _ in wiki_focus_counter.most_common() if tok in wiki_focus_tokens]

    # ---------- fallback frames & assembly ----------
    def render_benefit_sentences(head_cap: str, anchors_order: list[str]) -> list[str]:
        benefit_map = {
            "fluid": "keeps fluid balance steady",
            "balance": "keeps fluid balance steady",
            "electrolyte": "keeps electrolytes available so nerves and muscles fire correctly",
            "metabol": "supports metabolic reactions that turn food into energy",
            "homeostasis": "anchors overall homeostasis",
            "blood": "stabilizes blood volume and circulation",
            "volume": "stabilizes blood volume",
            "cells": "keeps your cells hydrated so nutrients diffuse",
            "brain": "sustains cognitive focus and mood",
            "muscle": "protects muscle endurance",
            "kidney": "protects kidney filtration",
            "skin": "protects skin elasticity",
            "immune": "supports immune defenses",
            "heart": "protects cardiovascular rhythm",
        }
        risk_map = {
            "dehydr": "the dehydration that disrupts metabolism",
            "sweat": "heat and heavy sweating",
            "diarrhea": "fluid loss from illness",
            "vomit": "fluid loss from illness",
            "heat": "heat stress",
        }
        tip_map = {
            "rehydrat": "Use oral rehydration solutions when you need a fast reset.",
            "electrolyte": "Pair water with electrolytes after hard training or illness.",
            "salts": "Keep a pinch of salts in long workout drinks to prevent cramps.",
            "volume": "Sip steadily across the day to avoid big swings in blood volume.",
            "therapy": "Clinical oral rehydration therapy works because it combines glucose with salts.",
        }

        clauses, seen_phr = [], set()
        for token in anchors_order:
            low = token.lower()
            for stem, phrase in benefit_map.items():
                if stem in low and phrase not in seen_phr:
                    clauses.append(phrase); seen_phr.add(phrase); break

        out = []
        if clauses:
            primary = clauses[:3]
            clause = primary[0] if len(primary) == 1 else (f"{primary[0]} and {primary[1]}" if len(primary)==2
                     else ", ".join(primary[:2]) + f", and {primary[2]}")
            out.append(f"{head_cap} {clause}.")
        risks, seen_r = [], set()
        for token in anchors_order:
            low = token.lower()
            for stem, phrase in risk_map.items():
                if stem in low and phrase not in seen_r:
                    risks.append(phrase); seen_r.add(phrase); break
        if risks:
            clause = risks[0] if len(risks)==1 else (f"{risks[0]} and {risks[1]}" if len(risks)==2
                     else ", ".join(risks[:2]) + f", and {risks[2]}")
            out.append(f"Stay ahead of {clause} by drinking before you feel thirsty.")
        for token in anchors_order:
            low = token.lower()
            for stem, tip in tip_map.items():
                if stem in low: out.append(tip); break
        if not out:
            out.append(f"{head_cap} keeps your cells supplied with water so circulation and temperature control stay on track.")
        cleaned = []
        for s in out:
            cleaned.extend(polish_and_validate(s, head_tok))
        return cleaned[:max(1, sentences - len(outputs))]

    def render_howto_steps(head_cap: str, alias_steps: list[str], anchors_order: list[str]) -> list[str]:
        steps = [s.format(Head=head_cap, head_lower=head_cap.lower()) for s in alias_steps] if alias_steps else []
        if not steps:
            common = []
            for tok in anchors_order:
                low = tok.lower()
                if "sensor" in low or "meter" in low: common.append("Calibrate your meters before sampling to avoid drift.")
                if "sample" in low: common.append("Label each sample with location, time, and conditions so you can trace anomalies.")
            steps = [
                "Clarify the standard you are measuring against.",
                "Collect a representative sample and keep it uncontaminated.",
                "Measure the key indicators with field kits or meters, then compare them with the thresholds."
            ] + common[:2]
        out = []
        max_steps = max(1, sentences - 1)
        for idx, step in enumerate(steps, 1):
            cand = f"Step {idx}: {step.rstrip('.')}."
            pol = polish_and_validate(cand, head_tok)
            out.extend(pol[:1])
            if len(out) >= max_steps: break
        return out

    def choose_anchors(h: str, L: dict, k_ctx=48, k_attn=48):
        ctx = top_k_context(L.get("cooc", {}), h, k=k_ctx)
        q = [h] + q_toks[:8]
        cand = candidate_pool(L, top_k=2000)
        fwd = mh_forward_scores(q, cand, lambda t,d: embed_token(t, L, d), dim=96, heads=4, seed=1729)
        bwd = mh_backward_scores([h], cand, lambda t,d: embed_token(t, L, d), dim=96, heads=4, seed=7331)
        attn = mh_posthoc_blend(fwd, bwd, alpha=attn_blend_alpha)
        scored, focus = [], wiki_focus_tokens
        universe = set(ctx) | set(cand[:800]) | focus
        for t in universe:
            if t in STOPWORDS or len(t) < 3: continue
            wiki_boost = 0.45 if t in focus else 0.0
            wiki_count = L.get("token_wiki_count", {}).get(t, 0)
            wiki_signal = 0.28 * min(1.0, wiki_count / 3)
            s = (0.65*attn.get(t, 0.0) +
                 0.35*math.tanh(0.08*L.get("tfidf", {}).get(t, 0.0)) +
                 wiki_boost + wiki_signal)
            scored.append((s, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = []
        for tok in [t for t,_ in wiki_focus_counter.most_common()]:
            if tok not in L.get("idf", {}): continue
            if tok in top: continue
            top.append(tok)
            if len(top) >= k_attn: break
        for _, t in scored:
            if t in top: continue
            top.append(t)
            if len(top) >= k_attn: break
        return top

    def accept_and_add(sentence: str):
        nonlocal used_words
        if not sentence: return False
        polished_lines = polish_and_validate(sentence, head_tok)
        accepted = False
        for s0 in polished_lines:
            if s0.lower() in used: continue
            v = semvec(s0)
            sims = [cosine(v, u) for u in vecs]
            max_sim = max(sims) if sims else 0.0
            if max_sim > 0.985: continue
            if 0.88 < max_sim <= 0.985:
                fresh = [t for t in advanced_tokenize(s0) if t not in used_words]
                if len(fresh) < 3: continue
            if quality_score(s0, head_tok) < 0.55 or word_salad_score(s0) > 0.45: continue
            outputs.append(s0); vecs.append(v); used.add(s0.lower())
            used_words.update(advanced_tokenize(s0))
            accepted = True
            if len(outputs) >= sentences: break
        return accepted

    anchors_all = [a.replace("_"," ") for a in choose_anchors(head_tok, lex)[:6]]

    if wiki_summary_pending and not how_to_prompt:
        for line in wiki_summary_pending:
            if len(outputs) >= sentences: break
            accept_and_add(line)

    if how_to_prompt and len(outputs) < sentences:
        if outputs: outputs.clear(); vecs.clear(); used.clear(); used_words.clear()
        for line in render_howto_steps(Head_cap, alias_howto_steps, anchors_all + [t for t,_ in wiki_focus_counter.most_common()]):
            if len(outputs) >= sentences: break
            accept_and_add(line)
        if len(outputs) < max(3, sentences-1):
            for line in [
                "Step 3: Compare the readings with your target limits and flag parameters that exceed them.",
                "Step 4: Note weather, nearby discharges, or odors that explain anomalies.",
                "Step 5: Schedule follow-up testing if any contaminant remains above the guideline.",
                "Step 6: Rinse and store your sampling gear so it’s ready for next time.",
                "Step 7: Review your notes and data to ensure everything is clear and complete.",
                "Step 8: Dispose of any waste materials according to local regulations."
            ]:
                if len(outputs) >= max(1, sentences - 1) or len(outputs) >= 5: break
                accept_and_add(line)
        if len(outputs) < max(3, min(5, sentences - 1)):
            accept_and_add("Quick check: record readings, tag them with location/time, and schedule lab follow-ups for anything above limit.")

    if alias_arxiv_terms and len(outputs) < sentences:
        arxiv_limit = max(0, sentences - 1)
        for line in arxiv_highlight_lines(alias_arxiv_terms, max(1, sentences - len(outputs))):
            if len(outputs) >= arxiv_limit: break
            accept_and_add(line)
    if alias_bib_entries and len(outputs) < sentences:
        bib_limit = max(0, sentences - 1)
        for line in render_bibliography_lines(alias_bib_entries, max(1, sentences - len(outputs))):
            if len(outputs) >= bib_limit: break
            accept_and_add(line)
    if alias_faq and len(outputs) < sentences:
        faq_limit = max(0, sentences - 1)
        for line in render_faq_lines(alias_faq, max(1, sentences - len(outputs))):
            if len(outputs) >= faq_limit: break
            accept_and_add(line)
    if wiki_summary_pending and how_to_prompt and len(outputs) < sentences:
        for line in wiki_summary_pending:
            if len(outputs) >= sentences: break
            accept_and_add(line)
    if alias_definition and len(outputs) < sentences:
        accept_and_add(alias_definition.format(Head=Head_cap, head_lower=Head_lower, Head_lower=Head_lower, original=Head_orig))

    if dictionary_entry and not alias_dict_override and len(outputs) < sentences and dictionary_entry.get("examples"):
        for ex in dictionary_entry["examples"][:2]:
            if len(outputs) >= sentences: break
            accept_and_add(f"Example: {clean_dictionary_text(ex)}")

    if alias_boosters and len(outputs) < sentences:
        boosters_iter = alias_boosters[:1] if how_to_prompt else alias_boosters
        booster_limit = max(0, sentences - 1)
        for template in boosters_iter:
            if len(outputs) >= booster_limit: break
            sentence = template.format(Head=Head_cap, head_lower=Head_lower, Head_lower=Head_lower, original=Head_orig)
            accept_and_add(sentence)

    if is_benefit_prompt and len(outputs) < sentences:
        for line in render_benefit_sentences(Head_cap, anchors_all + [t for t,_ in wiki_focus_counter.most_common()]):
            if len(outputs) >= sentences: break
            accept_and_add(line)

    if not how_to_prompt and is_definitional_prompt(prompt) and len(outputs) == 0:
        accept_and_add(make_definition_sentence(Head))

    dom_source = dominant_source_for_token(head_tok, lex)
    if not dom_source:
        for candidate in anchors_all:
            if dom_source: break
            for token in advanced_tokenize(candidate):
                dom_source = dominant_source_for_token(token, lex)
                if dom_source: break
    rel_suffix = ""
    if dom_source:
        src_label, ratio = dom_source
        label_map = {"arxiv":"arXiv corpus","wiki":"Wikipedia summaries","story":"persona stories"}
        rel_suffix = f" [Reliability: {label_map.get(src_label, src_label)} {ratio:.0%}]"

    if not how_to_prompt:
        if not alias_dict_override:
            if len(outputs) < sentences and anchors_all[:3]:
                body = ", ".join(anchors_all[:2]) + (f", and {anchors_all[2]}" if len(anchors_all) >= 3 else "")
                accept_and_add(f"In practice, {Head_lower} shows up in {body}.")
            if len(outputs) < sentences:
                a = anchors_all[0] if anchors_all else "a process"
                b = anchors_all[1] if len(anchors_all) > 1 else "a system"
                accept_and_add(f"Formally, you can model {Head_lower} as {a} within {b}.")
            if len(outputs) < sentences:
                x = anchors_all[2] if len(anchors_all) > 2 else "its domain"
                accept_and_add(f"Why it matters: {Head_cap} provides precision within {x}.")
        if len(outputs) < sentences:
            accept_and_add("Quick check: restate it in plain words and give one example from your field." + rel_suffix)
    else:
        if len(outputs) < sentences:
            accept_and_add("Quick check: record the readings you captured and note any site conditions that could skew future samples." + rel_suffix)

    if how_to_prompt and pending_alias_intro and len(outputs) < sentences:
        accept_and_add(pending_alias_intro)

    # ---------- final ----------
    if not outputs:
        outputs = [clean_period(f"{Head_cap} connects ideas so we can reason and act.")]
    outputs = [grammar_polish(o) for o in outputs][:max(1, sentences)]
    return " ".join(outputs)

# --------------------------
# Metrics
# --------------------------
def declare_density(lex, persona=None):
    d = lex["density"]; srcs = d.get("sources", {})
    persona_note = f" | Personas: {lex['personas']}" if lex.get("personas") else ""
    lines = [
        "[Corpus Density]",
        f"Nodes: {d['nodes']:,} | Edges(≥{lex['edge_threshold']}): {d['edges>=3']:,}",
        f"Graph density: {d['graph_density']:.6f} | Avg degree: {d['avg_degree']:.3f}",
        f"MWE count: {d['mwe_count']:,} | MWE rate: {d['mwe_rate']:.6f}",
        f"TF–IDF mass: {d['tfidf_mass']:.2f}",
        "[Reality & Persona Awareness]",
        f"Sources: {srcs} | Wikipedia ratio: {d['wiki_ratio']:.3f} | Stories ratio: {d['story_ratio']:.3f}{persona_note}"
    ]
    if persona and persona in lex.get("persona_style",{}):
        st = lex["persona_style"][persona]
        lines.append(f"[Persona '{persona}'] voice={st.get('voice')} pronoun={st.get('pronoun')} "
                     f"values={st.get('values')} style={st.get('style')}")
    return "\n".join(lines)

def declare_fairness_note(fair_alpha, fair_lambda, attn_blend_alpha):
    return (f"[Bias & Fairness]\n"
            f"C3F miscoverage α={fair_alpha:.3f} | counterfactual λ={fair_lambda:.3f} | "
            f"attn blend α_fwd={attn_blend_alpha:.2f}")

# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="Rebuild caches.")
    ap.add_argument("--sentences", type=int, default=5)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--wiki", action="store_true", help="Include Wikipedia booster.")
    ap.add_argument("--wiki_topics", type=str, default="")
    ap.add_argument("--stories_dir", type=str, default="")
    ap.add_argument("--persona", type=str, default="", help="Persona name from stories' front-matter.")
    ap.add_argument("--story_weight", type=float, default=0.45)
    ap.add_argument("--persona_weight", type=float, default=0.65)
    ap.add_argument("--bootstrap_stories", action="store_true", help="Create example stories if none exist.")

    # NEW: attention + fairness knobs
    ap.add_argument("--attn_blend_alpha", type=float, default=0.65, help="Forward vs backward blend (toward forward).")
    ap.add_argument("--fair_alpha", type=float, default=0.10, help="Global miscoverage for C3F.")
    ap.add_argument("--fair_lambda", type=float, default=0.25, help="Counterfactual regularization strength.")

    args = ap.parse_args()

    print("\n=== Faruk Alpay Persona + Reality Dense Reflective Engine (C3F) ===")
    print("Enter a prompt (press Enter twice to run)\n")
    lines=[]
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line: break
        lines.append(line)
    prompt = " ".join(lines).strip() or "explain fixed point theorem with a human narrative"

    # Ensure stories present
    if args.stories_dir:
        os.makedirs(args.stories_dir, exist_ok=True)
        if args.bootstrap_stories and not any(fn.lower().endswith((".txt",".md",".markdown")) for fn in os.listdir(args.stories_dir)):
            bootstrap_stories(args.stories_dir)

    # Fetch corpora
    arxiv = fetch_arxiv_corpus(rebuild=args.rebuild)
    wiki  = fetch_wiki_corpus(rebuild=args.rebuild, topics=[t.strip() for t in args.wiki_topics.split(",") if t.strip()]) if args.wiki else []
    stories = load_stories(args.stories_dir) if args.stories_dir else []

    combined = combine_entries(arxiv, wiki, stories)
    lex = build_lexicon(combined, rebuild=True if args.rebuild else False)

    print("\n" + declare_density(lex, persona=args.persona))
    print("\n" + declare_fairness_note(args.fair_alpha, args.fair_lambda, args.attn_blend_alpha) + "\n")

    out = transfinite_generate(
        prompt, lex,
        sentences=max(1,args.sentences),
        depth=max(1,args.depth),
        persona=(args.persona or None),
        story_weight=max(0.0, min(1.0, args.story_weight)),
        persona_weight=max(0.0, min(1.0, args.persona_weight)),
        attn_blend_alpha=max(0.0, min(1.0, args.attn_blend_alpha)),
        fair_alpha=max(0.0, min(1.0, args.fair_alpha)),
        fair_lambda=max(0.0, min(3.0, args.fair_lambda))
    )
    print("[Reflective Output]\n")
    print(out)

if __name__ == "__main__":
    main()
# =====================[ END OF FILE ]==========================
