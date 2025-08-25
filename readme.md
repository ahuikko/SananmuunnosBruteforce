# Sananmuunnos Bruteforcer 🔀🇫🇮

This project builds a database of **sananmuunnos** (a Finnish wordplay technique where the beginnings of two words are swapped).  
For example:

pirisevä sumppi → suriseva pimppi = (hilarious)


It generates all valid swaps from a given Finnish wordlist and stores them into an SQLite database.  
Then you can query the database in flexible ways — by exact word, multiple words, prefixes, SQL‐LIKE patterns, or even full regex.

---

## Features ✨

- Efficient swap generation: avoids naïve `n²` loops by grouping words by suffix.
- Supports configurable prefix length (`--k`).
- Stores results in **SQLite** with indexes for fast search.
- Powerful query modes:
  - **equals** (exact match)
  - **prefix** (e.g. `kul*`)
  - **suffix** (e.g. `*kki`)
  - **like** (SQL LIKE patterns `%` / `_`)
  - **regex** (full Python regular expressions)
- Multi-term search with `--logic any|all`.
- Scope control: search in swapped pairs, normal pairs, or anywhere.
- `--count-only` mode for fast result counts.
- Case sensitivity toggle for LIKE/regex searches.

---

## Requirements

- Python 3.9+
- No external dependencies (just `sqlite3` from the stdlib)

---

## Usage

### 1. Build the database

```bash
python sananmuunnos_bruteforce.py build --wordlist kaikkisanat_fusion.txt --db muunnokset.sqlite --k 2
```
If using the kaikkisanat_fusion.txt, generation can take couple of minutes and the db is quite large ~700Mb

This will parse ~100k words in a few minutes and generate hundreds of thousands of swaps.
### 2. Query the database

Basic exact match:
```bash
python sananmuunnos_bruteforce.py query --db muunnokset.sqlite --words kulli --scope swapped
```
Multiple words (must all appear in swapped pair):
```bash
python sananmuunnos_bruteforce.py query --db muunnokset.sqlite --words tina murahtaa --logic all --scope swapped
```
Prefix search (fast, uses index):
```bash
python sananmuunnos_bruteforce.py query --db muunnokset.sqlite --words kul --mode prefix --scope any
```
Regex search (powerful but slower):
```bash
python sananmuunnos_bruteforce.py query --db muunnokset.sqlite --words "k(ul|yl)li" --mode regex --scope any
```
Count only (no rows, just a number):
```bash
python sananmuunnos_bruteforce.py query --db muunnokset.sqlite --words "k.*i$" --mode regex --count-only
```
CLI Options Summary
build

    --wordlist PATH : input file, one word per line.

    --db PATH : SQLite output (default muunnokset.sqlite).

    --k N : prefix length to swap (default 2).

    --limit N : stop after N swaps (for testing).

query

    --db PATH : SQLite database (default muunnokset.sqlite).

    --words ... : one or more search terms.

    --word WORD : single search term (legacy).

    --scope : swapped|normal|any (default swapped).

    --logic : any|all (default any).

    --mode : equals|prefix|suffix|like|regex (default equals).

    --case-sensitive: toggle case sensitivity for LIKE/regex.

    --k N : restrict to swaps of this prefix length.

    --limit N : cap result rows.

    --count-only : print only the total match count.



## Now you ask, why?

Because Finnish wordplay is hilarious and systematically mining sananmuunnos pairs reveals lots of gems.
This tool is equal parts linguistic toy, cultural archaeology kit, and profanity generator.
