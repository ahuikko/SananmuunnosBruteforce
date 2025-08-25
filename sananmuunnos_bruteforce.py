'''
MIT License

Copyright (c) 2025 Alex Huikko

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# -*- coding: utf-8 -*-
import argparse
import itertools
import os
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from typing import List, Optional, Tuple

# Accept only Finnish-style single tokens by default (letters incl. ÅÄÖ).
VALID_CHARS_RE = re.compile(r"^[A-Za-zÄÖÅäöå]+$")

# Vowel set used by split_onset (optional mode).
VOWELS = set("AEIOUYÄÖÅaeiouyäöå")


# ----------- Normalization & validation -----------

def nfc_lower(s: str) -> str:
    """Normalize to NFC and lowercase; strip surrounding whitespace."""
    return unicodedata.normalize("NFC", s).strip().lower()


def is_valid_token(w: str, k: int) -> bool:
    """
    A valid token for building:
      - only letters (incl. ä/ö/å)
      - length >= k+1 so there's a non-empty suffix after the k-prefix
    """
    return bool(VALID_CHARS_RE.match(w)) and len(w) >= k + 1


# ----------- Splitting helpers -----------

def split_k(word: str, k: int) -> Tuple[str, str]:
    """Fixed-length prefix (k) and the rest as suffix."""
    return word[:k], word[k:]


def split_onset(word: str) -> Tuple[str, str]:
    """
    Alternative: split at first vowel (consonant onset as 'prefix').
    Not used by default; left here for experiments.
    """
    i = 0
    while i < len(word) and word[i] not in VOWELS:
        i += 1
    return word[:i], word[i:]


# ----------- Schema management -----------

def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY,
            word TEXT UNIQUE
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS swaps (
            id INTEGER PRIMARY KEY,
            k INTEGER NOT NULL,
            w1_id INTEGER NOT NULL,
            w2_id INTEGER NOT NULL,
            sw1_id INTEGER NOT NULL,
            sw2_id INTEGER NOT NULL,
            p1 TEXT NOT NULL,
            p2 TEXT NOT NULL,
            s1 TEXT NOT NULL,
            s2 TEXT NOT NULL,
            UNIQUE(k, w1_id, w2_id, sw1_id, sw2_id)
        );
    """)
    # Helpful indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_words_word ON words(word);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_swaps_suffs ON swaps(s1, s2);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_swaps_w ON swaps(w1_id, w2_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_swaps_sw ON swaps(sw1_id, sw2_id);")
    conn.commit()


# ----------- Word ID helpers -----------

def get_word_id(conn: sqlite3.Connection, cache: dict, word: str) -> int:
    if word in cache:
        return cache[word]
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO words(word) VALUES (?)", (word,))
    conn.commit()
    cur.execute("SELECT id FROM words WHERE word=?", (word,))
    row = cur.fetchone()
    wid = row[0]
    cache[word] = wid
    return wid


def insert_swap(conn: sqlite3.Connection, k: int,
                w1: str, w2: str, sw1: str, sw2: str,
                p1: str, p2: str, s1: str, s2: str,
                id_cache: dict) -> None:
    w1_id = get_word_id(conn, id_cache, w1)
    w2_id = get_word_id(conn, id_cache, w2)
    sw1_id = get_word_id(conn, id_cache, sw1)
    sw2_id = get_word_id(conn, id_cache, sw2)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO swaps(k, w1_id, w2_id, sw1_id, sw2_id, p1, p2, s1, s2)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (k, w1_id, w2_id, sw1_id, sw2_id, p1, p2, s1, s2))


# ----------- Loading & indexing -----------

def load_words(wordlist_path: str, k: int) -> List[str]:
    words: List[str] = []
    with open(wordlist_path, "r", encoding="utf-8") as f:
        for line in f:
            w = nfc_lower(line)
            if not w:
                continue
            if is_valid_token(w, k):
                words.append(w)
    return words


def build_index(words: List[str], k: int, splitter=split_k):
    """
    Build:
      - word_set for O(1) existence checks
      - prefixes_by_suffix: suffix -> set(prefixes)
    """
    word_set = set(words)
    prefixes_by_suffix = defaultdict(set)
    for w in words:
        p, s = splitter(w, k) if splitter is split_k else splitter(w)
        prefixes_by_suffix[s].add(p)
    return word_set, prefixes_by_suffix


# ----------- Swap generation -----------

def generate_swaps(conn: sqlite3.Connection, words: List[str],
                   k: int, max_pairs_hint: Optional[int] = None) -> int:
    """
    Efficient generation:
      - Group by suffix (after k-prefix).
      - For two suffixes s1, s2, look at common prefixes C = P[s1] ∩ P[s2].
      - For ordered prefix pairs (p1, p2) with p1 != p2:
           w1 = p1+s1, w2 = p2+s2, sw1 = p2+s1, sw2 = p1+s2
        Insert if all four exist in the word list.
    """
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = WAL;")

    word_set, pref_by_suf = build_index(words, k)
    id_cache: dict = {}
    seen = set()

    suf_list = list(pref_by_suf.keys())
    suf_list.sort(key=lambda s: (len(s), s))

    batch = 0
    total = 0

    for i in range(len(suf_list)):
        s1 = suf_list[i]
        P1 = pref_by_suf[s1]
        if len(P1) < 2:
            continue

        for j in range(i + 1, len(suf_list)):
            s2 = suf_list[j]
            P2 = pref_by_suf[s2]
            if len(P2) < 2:
                continue

            common = P1 & P2
            if len(common) < 2:
                continue

            for p1, p2 in itertools.permutations(common, 2):
                if p1 == p2:
                    continue

                w1 = p1 + s1
                w2 = p2 + s2
                sw1 = p2 + s1
                sw2 = p1 + s2

                if not (w1 in word_set and w2 in word_set and sw1 in word_set and sw2 in word_set):
                    continue

                key = (tuple(sorted((w1, w2))),
                       tuple(sorted((sw1, sw2))),
                       k)
                if key in seen:
                    continue
                seen.add(key)

                insert_swap(conn, k, w1, w2, sw1, sw2, p1, p2, s1, s2, id_cache)
                batch += 1
                total += 1

                if batch >= 2000:
                    conn.commit()
                    batch = 0

                if max_pairs_hint is not None and total >= max_pairs_hint:
                    conn.commit()
                    return total

    if batch:
        conn.commit()
    return total


# ----------- Query (multi-term + regex/like/prefix/suffix) -----------

def _scope_fields(scope: str) -> List[str]:
    if scope == "swapped":
        return ["sw1.word", "sw2.word"]
    if scope == "normal":
        return ["w1.word", "w2.word"]
    return ["w1.word", "w2.word", "sw1.word", "sw2.word"]


def _like_escape(s: str) -> str:
    # Escape % and _ for LIKE when we build patterns
    return s.replace("%", r"\%").replace("_", r"\_")


def _register_regexp(conn: sqlite3.Connection, case_sensitive: bool) -> None:
    flags = 0 if case_sensitive else re.IGNORECASE

    def regexp(pattern: str, value: Optional[str]) -> int:
        if value is None:
            return 0
        try:
            return 1 if re.search(pattern, value, flags) else 0
        except re.error:
            # Invalid pattern -> no match
            return 0

    # Provide REGEXP operator to SQLite: column REGEXP pattern
    conn.create_function("REGEXP", 2, regexp)


def _build_where(terms: List[str], scope: str, logic: str, k: Optional[int],
                 mode: str, case_sensitive: bool) -> Tuple[str, List]:
    """
    Build WHERE SQL and params for multi-term matching.

    mode:
      - equals : field = ?
      - prefix : field LIKE ? ESCAPE '\\'  (term%)
      - suffix : field LIKE ? ESCAPE '\\'  (%term)
      - like   : field LIKE ? ESCAPE '\\'  (as-is)
      - regex  : field REGEXP ?
    """
    fields = _scope_fields(scope)
    params: List = []
    clauses: List[str] = []

    # Normalize terms; for regex we keep user’s case but the UDF handles case sensitivity.
    def norm(t: str) -> str:
        return unicodedata.normalize("NFC", t).strip()

    words = [norm(t) for t in terms if t and t.strip()]
    if not case_sensitive and mode != "regex":
        # For equals/like/prefix/suffix we store lowercase; our DB words are lowercase.
        words = [t.lower() for t in words]

    for term in words:
        per_field = []
        if mode == "equals":
            per_field = [f"{f} = ?" for f in fields]
            params.extend([term] * len(fields))
        elif mode == "prefix":
            patt = _like_escape(term) + "%"
            per_field = [f"{f} LIKE ? ESCAPE '\\'" for f in fields]
            params.extend([patt] * len(fields))
        elif mode == "suffix":
            patt = "%" + _like_escape(term)
            per_field = [f"{f} LIKE ? ESCAPE '\\'" for f in fields]
            params.extend([patt] * len(fields))
        elif mode == "like":
            # User supplies raw LIKE pattern (honor %/_)
            per_field = [f"{f} LIKE ? ESCAPE '\\'" for f in fields]
            params.extend([term] * len(fields))
        elif mode == "regex":
            per_field = [f"{f} REGEXP ?" for f in fields]
            params.extend([term] * len(fields))
        else:
            raise ValueError("Unsupported mode")

        clause = "(" + " OR ".join(per_field) + ")"
        clauses.append(clause)

    where_parts: List[str] = []
    if clauses:
        joiner = " AND " if logic == "all" else " OR "
        where_parts.append("(" + joiner.join(clauses) + ")")

    if k is not None:
        where_parts.append("s.k = ?")
        params.append(k)

    where_sql = " AND ".join(where_parts) if where_parts else "1=1"
    return where_sql, params


def query_swaps(conn: sqlite3.Connection,
                words: List[str],
                scope: str = "swapped",
                logic: str = "any",
                limit: Optional[int] = None,
                k: Optional[int] = None,
                mode: str = "equals",
                case_sensitive: bool = False,
                count_only: bool = False):
    """
    Multi-term query with several match modes (equals/prefix/suffix/like/regex).
    """
    if mode == "regex":
        _register_regexp(conn, case_sensitive)

    base_from = """
    FROM swaps s
    JOIN words w1  ON s.w1_id  = w1.id
    JOIN words w2  ON s.w2_id  = w2.id
    JOIN words sw1 ON s.sw1_id = sw1.id
    JOIN words sw2 ON s.sw2_id = sw2.id
    """
    where_sql, params = _build_where(words, scope, logic, k, mode, case_sensitive)
    cur = conn.cursor()

    if count_only:
        sql = f"SELECT COUNT(*) {base_from} WHERE {where_sql}"
        cur.execute(sql, tuple(params))
        row = cur.fetchone()
        return row[0] if row else 0

    sql = f"""
    SELECT w1.word, w2.word, sw1.word, sw2.word, s.k, s.p1, s.p2, s.s1, s.s2
    {base_from}
    WHERE {where_sql}
    ORDER BY s.k, w1.word, w2.word
    """
    if limit is not None and limit > 0:
        sql += f" LIMIT {int(limit)}"

    cur.execute(sql, tuple(params))
    return cur.fetchall()


# ----------- CLI commands -----------

def cli_build(args: argparse.Namespace) -> None:
    if args.k < 1:
        print("k täytyy olla vähintään 1.", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.wordlist):
        print(f"Sanalistaa ei löytynyt: {args.wordlist}", file=sys.stderr)
        sys.exit(2)

    words = load_words(args.wordlist, args.k)
    if not words:
        print("Sanalista tyhjä tai kaikki sanat suodatettiin.", file=sys.stderr)
        sys.exit(2)

    conn = sqlite3.connect(args.db)
    ensure_schema(conn)
    total = generate_swaps(conn, words, k=args.k, max_pairs_hint=args.limit)
    print(f"Valmiit muunnokset talletettu: {total} kpl (k={args.k}). Tietokanta: {args.db}")


def cli_query(args: argparse.Namespace) -> None:
    conn = sqlite3.connect(args.db)
    ensure_schema(conn)

    # Back-compat: --word or new --words
    terms: List[str] = []
    if getattr(args, "word", None):
        terms.append(args.word)
    if getattr(args, "words", None):
        terms.extend(args.words)

    if not terms:
        print("Anna vähintään yksi hakutermi (--words ... tai --word ...).", file=sys.stderr)
        sys.exit(2)

    total = query_swaps(conn,
                        words=terms,
                        scope=args.scope,
                        logic=args.logic,
                        k=args.k,
                        mode=args.mode,
                        case_sensitive=args.case_sensitive,
                        count_only=True)

    if args.count_only:
        print(f"Total: {total} matches")
        return

    rows = query_swaps(conn,
                       words=terms,
                       scope=args.scope,
                       logic=args.logic,
                       k=args.k,
                       mode=args.mode,
                       case_sensitive=args.case_sensitive,
                       limit=args.limit,
                       count_only=False)

    if not rows:
        kval = f", k={args.k}" if args.k is not None else ""
        print(f"No matches (scope={args.scope}, logic={args.logic}, mode={args.mode}{kval}) "
              f"for: {', '.join(terms)}")
        return

    shown = 0
    for (w1, w2, sw1, sw2, k, p1, p2, s1, s2) in rows:
        print(f"[{w1} {w2}]  ->  [{sw1} {sw2}]  (k={k}; p1={p1}, p2={p2})")
        shown += 1

    if args.limit is not None and shown < total:
        print(f"Shown: {shown} / Total: {total} matches (increase --limit to see more)")
    else:
        print(f"Total: {total} matches")


def main() -> None:
    ap = argparse.ArgumentParser(description="Finnish wordplay (sananmuunnos) – generate and search in an SQLite database.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # build
    ap_build = sub.add_parser("build", help="Build the swap index from a word list.")
    ap_build.add_argument("--wordlist", required=True, help="Word list, one word per line (utf-8).")
    ap_build.add_argument("--db", default="muunnokset.sqlite", help="Target SQLite database file.")
    ap_build.add_argument("--k", type=int, default=2, help="Length of the swapped prefix (default 2).")
    ap_build.add_argument("--limit", type=int, default=None, help="Stop after this many swaps have been stored (debug).")
    ap_build.set_defaults(func=cli_build)

    # query
    ap_query = sub.add_parser("query", help="Search swaps for given words/patterns.")
    ap_query.add_argument("--db", default="muunnokset.sqlite", help="SQLite database file.")

    # Multi-term inputs (keep --word for backward compatibility)
    ap_query.add_argument("--words", nargs="+", help="One or more search terms (e.g. lasse kulli).")
    ap_query.add_argument("--word", help="Single search term (legacy option).")

    ap_query.add_argument("--scope", choices=["swapped", "normal", "any"], default="swapped",
                          help="Where to search: swapped=[sw1,sw2], normal=[w1,w2], any=[all four words].")
    ap_query.add_argument("--logic", choices=["any", "all"], default="any",
                          help="Match logic: any=at least one term must match; all=all terms must match.")

    # Match modes
    ap_query.add_argument("--mode", choices=["equals", "prefix", "suffix", "like", "regex"],
                          default="equals",
                          help="Match mode: equals|prefix|suffix|like|regex. 'like' uses SQL LIKE syntax (%, _).")
    ap_query.add_argument("--case-sensitive", action="store_true",
                          help="Make the search case-sensitive. For regex, disables re.IGNORECASE.")

    ap_query.add_argument("--k", type=int, default=None, help="Restrict to swaps with this prefix length.")
    ap_query.add_argument("--limit", type=int, default=None, help="Maximum number of results to return.")
    ap_query.add_argument("--count-only", action="store_true", help="Print only the total number of matches.")
    ap_query.set_defaults(func=cli_query)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
