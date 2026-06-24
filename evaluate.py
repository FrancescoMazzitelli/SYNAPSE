"""
USAGE
-----
Batch mode (auto-discover all models under results/ for data_MyDailyTravelData only):
  python evaluate2.py --results-dir results/ --out-dir evaluations/

Single-run mode (manual file paths, backward compatible):
  python evaluate2.py \\
      --results  results.csv \\
      --oracle   sampled_oracle.csv \\
      --log      log.txt \\
      --out      report.txt
"""

import argparse
import math
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from scipy.stats import entropy as scipy_entropy
from tabulate import tabulate

warnings.filterwarnings("ignore")

DATASET_NAME = "data_lyon_EDGT"  # default, overridden per-run when scanning multiple datasets

# ── Bilingual EN → FR aliases for dictionary label matching ──────────────
EN_FR_MAP = {
    "yes": "oui", "no": "non", "y": "oui", "n": "non",
    "work": "travail", "home": "domicile", "office": "bureau",
    "always": "toujours", "often": "souvent", "sometimes": "parfois",
    "never": "jamais", "rarely": "rarement",
    "other": "autre", "none": "aucun", "all": "tous",
    "male": "homme", "female": "femme", "man": "homme", "woman": "femme",
    "morning": "matin", "afternoon": "apres-midi", "evening": "soir",
    "monday": "lundi", "tuesday": "mardi", "wednesday": "mercredi",
    "thursday": "jeudi", "friday": "vendredi", "saturday": "samedi", "sunday": "dimanche",
    "private": "prive", "public": "public",
    "car": "voiture", "bus": "bus", "train": "train", "walk": "marche",
    "bike": "velo", "bicycle": "velo", "cycling": "velo",
    "telework": "teletravail", "teleworking": "teletravail",
    "student": "etudiant", "employed": "emploi", "unemployed": "chomeur",
    "retired": "retraite",
    "primary": "primaire", "secondary": "secondaire", "superior": "superieur",
    "university": "universite", "school": "ecole",
    "every": "chaque", "day": "jour", "week": "semaine", "month": "mois", "year": "an",
}


def _add_bilingual_aliases(label_map: dict) -> dict:
    """For each dictionary column, add English→French alias entries."""
    out = {}
    for col, fr_map in label_map.items():
        en_map = {}
        for fr_label, code in fr_map.items():
            en_map[fr_label] = code
            fr_tokens = fr_label.split()
            en_aliases = []
            for t in fr_tokens:
                for en_word, fr_word in EN_FR_MAP.items():
                    if t == fr_word:
                        en_aliases.append(fr_label.replace(t, en_word))
            for alias in en_aliases:
                en_map[alias.lower().strip()] = code
        out[col] = en_map
    return out


# ==============================================================================
# STEP 1 — LOAD & AUTO-MAP COLUMNS
# ==============================================================================

def load_and_map(results_path: str, oracle_path: str):
    """
    Load both CSVs.  Build a mapping {results_col -> oracle_col} via
    case-insensitive name matching.  Metadata-only columns in results
    (those with no counterpart in oracle) are collected separately.
    Returns:
        res_aligned : DataFrame — only the matched columns, normalised strings
        ora_aligned : DataFrame — counterpart oracle columns, same order
        col_map     : dict {res_col -> ora_col}
        meta_cols   : list of results columns that have no oracle match
        n           : number of rows used
    """
    res_raw = pd.read_csv(results_path)
    ora_raw = pd.read_csv(oracle_path)

    # Build lookup: normalised oracle col name -> original oracle col name
    def _key(s):
        return s.lower().replace(" ", "_")

    ora_lookup = {_key(c): c for c in ora_raw.columns}

    col_map   = {}   # res_col -> ora_col
    meta_cols = []   # res_col with no oracle match

    for rc in res_raw.columns:
        if _key(rc) in ora_lookup:
            col_map[rc] = ora_lookup[_key(rc)]
        else:
            meta_cols.append(rc)


    if not col_map:
        sys.exit(
            "[ERROR] No column overlap found between results and oracle. "
            "Check that the files share question columns (even with different case)."
        )

    # Align rows
    n = min(len(res_raw), len(ora_raw))
    res_raw = res_raw.iloc[:n].reset_index(drop=True)
    ora_raw = ora_raw.iloc[:n].reset_index(drop=True)

    # Normalise string content (uppercase, strip whitespace)
    def _norm_df(df):
        out = df.copy()
        for c in out.columns:
            if out[c].dtype == object or str(out[c].dtype) == "string":
                out[c] = out[c].apply(
                    lambda x: str(x).strip().upper() if not pd.isna(x) else x
                )
        return out

    res_norm = _norm_df(res_raw)
    ora_norm = _norm_df(ora_raw)

    # Build aligned sub-DataFrames — oracle columns renamed to match results
    res_cols = list(col_map.keys())
    ora_cols = [col_map[c] for c in res_cols]

    res_aligned = res_norm[res_cols].copy()
    ora_aligned = ora_norm[ora_cols].copy()
    ora_aligned.columns = res_cols   # same col names so we can zip by name

    return res_aligned, ora_aligned, col_map, meta_cols, n


# ==============================================================================
# STEP 1b — RESOLVE "CODE: LABEL" FORMAT IN AGENT RESPONSES
# ==============================================================================

NR_SENTINEL = "<NR>"

NON_RESPONSE_CODES = frozenset({"-1", "-7", "-8", "-9"})


def _strip_float(s: str) -> str:
    """Strip trailing '.0' from a numeric string."""
    m = re.match(r'^(-?\d+)\.0+$', s)
    return m.group(1) if m else s


# ── Survey dictionary (label → code) ─────────────────────────────────────
def load_label_map(dict_path: str = None) -> dict:
    """
    Load the survey dictionary CSV and return a dict:
      {col_name_lower: {normalised_label: code}}

    Each row's *Responses* column contains comma‑separated ``code: label``
    entries which are parsed into a forward label→code lookup.
    """
    if dict_path is None:
        dict_path = str(Path(__file__).parent /
                        f"configs/Generic/{DATASET_NAME}/dictionary.csv")
    try:
        import csv
        label_map = {}
        with open(dict_path, encoding="utf-8") as f:
            # Auto-detect delimiter: check first line for semicolon
            first_line = f.readline()
            delimiter = ";" if ";" in first_line else ","
            f.seek(0)
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                name = row.get("Name", "").strip().lower()
                resp = row.get("Responses", "")
                if not name or not resp:
                    continue
                col_map = {}
                # Split on commas followed by a digit+colon (code boundary),
                # so commas inside labels (e.g. "OUI, AVEC ACCES") are preserved.
                for part in re.split(r",\s*(?=\d+\s*:)", resp):
                    part = part.strip()
                    m = re.match(r"^(-?\d+)\s*:\s*(.+)$", part)
                    if m:
                        code = m.group(1)
                        lbl = re.sub(r"^\d+[\.\:\)]+\s*", "", m.group(2)).strip().lower()
                        lbl = re.sub(r"\s+", " ", lbl)
                        col_map[lbl] = code
                if col_map:
                    label_map[name] = col_map
        label_map = _add_bilingual_aliases(label_map)
        return label_map
    except FileNotFoundError:
        print(f"  [WARN] Dictionary not found at {dict_path}", file=sys.stderr)
        return {}


_LABEL_MAP = None


def _get_label_map():
    global _LABEL_MAP
    if _LABEL_MAP is None:
        _LABEL_MAP = load_label_map()
    return _LABEL_MAP


def _resolve_agent(val, col: str = ""):
    """
    Normalise a single agent response value.
    Steps:
      A) Extract code from parenthetical description  (valeur fixe = 4)  → 4
      B) Try dictionary FIRST (label → code mapping)
      C) Parse Python‑list format  ['code: text', ...]  → first code
      D) Parse  code: label  → code
      E) Bare number → keep
      F) Fallback dictionary lookup (substring)
      G) Normalise non‑response codes  (-1, -7, -8, -9)  → <NR>
    """
    if pd.isna(val):
        return val
    s = str(val).strip()

    # A) Extract code from parenthetical descriptions like "(valeur fixe = 4)"
    m_paren = re.search(r'\([^)]*?[=:][^)]*?(-?\d+)[^)]*\)', s)
    if m_paren:
        code = m_paren.group(1)
        if code in NON_RESPONSE_CODES:
            return NR_SENTINEL
        return code

    # B) Try dictionary FIRST (label → code mapping)
    if col:
        lmap = _get_label_map().get(col.lower(), {})
        if lmap:
            q = re.sub(r"^\d+[\.\:\)]+\s*", "", s.lower()).strip()
            q = re.sub(r"\s+", " ", q)
            if q:
                code = lmap.get(q)
                if code:
                    return code
                # Substring match
                for lbl, code in lmap.items():
                    if q in lbl or lbl in q:
                        return code

    # C) Python list of quoted strings from multi‑select
    #    e.g. "['1: I had work', '2: I had an appointment']"
    m_list = re.match(r"\[\s*'(-?\d+)", s)
    if m_list:
        code = m_list.group(1)
        if code in NON_RESPONSE_CODES:
            return NR_SENTINEL
        return code

    # D) "code: label"  format → extract code directly
    m_code = re.match(r"^(-?\d+)\s*:", s)
    if m_code:
        code = m_code.group(1)
        if code in NON_RESPONSE_CODES:
            return NR_SENTINEL
        return code

    # E) Bare number
    s_clean = _strip_float(s)
    if s_clean in NON_RESPONSE_CODES:
        return NR_SENTINEL

    # F) Fallback dictionary substring lookup (for partial matches)
    if col:
        lmap = _get_label_map().get(col.lower(), {})
        if lmap:
            q = re.sub(r"^\d+[\.\:\)]+\s*", "", s.lower()).strip()
            q = re.sub(r"\s+", " ", q)
            if q:
                for lbl, code in lmap.items():
                    if q in lbl or lbl in q:
                        return code

    return s


def _normalise_oracle(val):
    """Normalise an oracle value: strip .0 and map non‑response → <NR>."""
    if pd.isna(val):
        return val
    s = str(val).strip()
    s = _strip_float(s)
    if s in NON_RESPONSE_CODES:
        return NR_SENTINEL
    return s


def _parse_py_list(s: str) -> list:
    """Extract numeric codes from a Python‑list string like \"['1: text', '2: text']\"."""
    return re.findall(r"'(-?\d+)\s*:", s)


def _resolve_flag_col(val, expected_code: str, col: str = ""):
    """
    For multi‑select flag columns (e.g. trip_appt_why_1).
    If the agent wrote a Python list, check whether *expected_code* is in it.
    """
    if pd.isna(val):
        return val
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        codes = _parse_py_list(s)
        if codes:
            return expected_code if expected_code in codes else NR_SENTINEL
    # fall through to regular resolution
    return _resolve_agent(val, col=col)


def resolve_code_label_format(res: pd.DataFrame, ora: pd.DataFrame):
    """
    Convert agent responses to canonical form for fair comparison with oracle.

    Handles:
      - 'code: label'  →  code  (e.g. '1: Yes' → '1')
      - Python multi‑select lists  →  first code
      - Multi‑select *flag* columns (name ends in _N) → check membership
      - Non‑response codes -1/-7/-8/-9  →  <NR> sentinel (all match each other)
      - Numeric oracle codes stripped of trailing '.0'
      - Text columns → uppercase + collapse whitespace
    """
    for col in res.columns:
        ora_col = ora[col]

        # Decide whether this is a numeric‑coded column based on actual data
        ora_as_str = ora_col.dropna().astype(str)
        if len(ora_as_str) == 0:
            continue
        num_frac = pd.to_numeric(ora_as_str, errors="coerce").notna().mean()

        # ── Numeric‑coded column ──────────────────────────────────
        if num_frac > 0.5:
            # Check for multi‑select flag columns: name ends with _DIGIT
            flag_match = re.match(r".+_(-?\d+)$", col)
            if flag_match:
                expected = flag_match.group(1)
                res[col] = res[col].apply(
                    lambda v, e=expected, c=col: _resolve_flag_col(v, e, col=c)
                )
            else:
                res[col] = res[col].apply(lambda v, c=col: _resolve_agent(v, col=c))
            ora[col] = ora[col].apply(_normalise_oracle)
            continue

        # ── Text column (e.g. source = survey / gps) ──────────────
        def _clean_text(val):
            if pd.isna(val):
                return val
            return re.sub(r"\s+", " ", str(val).strip().upper())

        res[col] = res[col].apply(_clean_text)
        ora[col] = ora[col].apply(_clean_text)

    return res, ora


def binarize_flag_cols(res: pd.DataFrame, ora: pd.DataFrame):
    """
    Multi‑select indicator columns have only <NR> (not selected) and one
    valid code (e.g. '1' = selected).  Map <NR> → '0' so that 'not
    selected' becomes a real response category.  This matches standard
    survey practice (binarize to {0, 1}).
    """
    binarized = []
    for col in res.columns:
        o = ora[col].dropna().astype(str)
        u = o.unique()
        vals_no_nr = set(u) - {NR_SENTINEL}
        if len(vals_no_nr) <= 1 and NR_SENTINEL in u:
            ora[col] = ora[col].astype(str).replace({NR_SENTINEL: "0"})
            res[col] = res[col].astype(str).replace({NR_SENTINEL: "0"})
            binarized.append(col)
    return binarized


def identify_nr_columns(ora: pd.DataFrame, threshold: float = None) -> set:
    """
    Return the set of column names where the combined
    <NR> + NA rate in the oracle exceeds *threshold*.
    """
    if threshold is None:
        threshold = NR_THRESHOLD
    drop_keys = {NR_SENTINEL, "NAN", "NONE", "NAT", ""}
    nr_cols = set()
    for col in ora.columns:
        s = ora[col]
        nr_na = s.isna() | s.astype(str).isin(drop_keys)
        if nr_na.mean() > threshold:
            nr_cols.add(col)
    return nr_cols


def identify_agent_constant_columns(res: pd.DataFrame, ora: pd.DataFrame,
                                    min_unique: int = None) -> set:
    """
    Return the set of column names where the agent's output has
    ≤ *min_unique* distinct normalised values while the oracle has
    more than that (i.e. the oracle genuinely varies).
    """
    if min_unique is None:
        min_unique = AGENT_VAR_MIN_UNIQUE

    def _norm(v):
        if pd.isna(v):
            return v
        return str(v).strip().upper()

    constant_cols = set()
    for col in res.columns:
        rv = res[col].dropna().apply(_norm)
        ov = ora[col].dropna().apply(_norm)
        r_uniq = rv.nunique() if len(rv) > 0 else 0
        o_uniq = ov.nunique() if len(ov) > 0 else 0
        if r_uniq <= min_unique and o_uniq > min_unique:
            constant_cols.add(col)
    return constant_cols


def identify_zero_overlap_columns(res: pd.DataFrame, ora: pd.DataFrame) -> set:
    """
    Return the set of column names where the agent's output codes have
    absolutely no overlap with the oracle's answer codes (excluding sentinel
    non‑response values).  These are columns where the agent is systematically
    answering a different question or using a different code space.
    """
    drop_keys = frozenset({NR_SENTINEL, "NAN", "NONE", "NAT", ""})
    no_overlap = set()
    for col in res.columns:
        r_set = set()
        o_set = set()
        for rv, ov in zip(res[col], ora[col]):
            rs = str(rv).strip().lower()
            os = str(ov).strip().lower()
            if pd.isna(rv) or rs in drop_keys:
                pass
            else:
                r_set.add(rs)
            if pd.isna(ov) or os in drop_keys:
                pass
            else:
                o_set.add(os)
        if not r_set or not o_set:
            continue
        if r_set.isdisjoint(o_set):
            no_overlap.add(col)
    return no_overlap


# ==============================================================================
# STEP 2 — AUTO-CLASSIFY COLUMNS
# ==============================================================================

# Thresholds (overridable via CLI)
CATEGORICAL_MAX_UNIQUE = 20    # <= N distinct values in oracle -> categorical
FREE_TEXT_MIN_UNIQUE   = 5     # >= N distinct values in oracle -> free-text
NR_THRESHOLD           = 0.9   # drop columns where oracle NR/NA rate exceeds this

# Excluded columns — agent and oracle content types differ fundamentally
# (e.g. agent generates narrative bio text while oracle has categorical codes)
EXCLUDED_COLS          = set()   # Lyon EDGT has no persona columns

# Agent variability filter — drop columns where the agent outputs the same
# text for nearly every row (≤ AGENT_VAR_MIN_UNIQUE unique values) while the
# oracle genuinely varies (> AGENT_VAR_MIN_UNIQUE unique values).
AGENT_VAR_MIN_UNIQUE   = 2       # agent must have > this many unique values


def classify_columns(res: pd.DataFrame, ora: pd.DataFrame):
    """
    Classify each shared column into one of:
      'numeric'     — both sides can be cast to numbers
      'categorical' — small-cardinality string field
      'free_text'   — open-ended string with high cardinality

    Classification is driven by the ORACLE distribution (ground truth).
    Returns four lists: numeric_cols, categorical_cols, free_text_cols
    """
    numeric_cols     = []
    categorical_cols = []
    free_text_cols   = []

    for col in res.columns:
        ora_col = ora[col]
        res_col = res[col]

        # Try numeric on oracle
        ora_num = pd.to_numeric(ora_col, errors="coerce")
        res_num = pd.to_numeric(res_col, errors="coerce")
        if ora_num.notna().mean() > 0.8 and res_num.notna().mean() > 0.8:
            numeric_cols.append(col)
            continue

        # String classification based on oracle cardinality
        ora_vals  = ora_col.dropna().astype(str)
        n_unique  = ora_vals.nunique()
        fill_rate = ora_col.notna().mean()

        if n_unique <= CATEGORICAL_MAX_UNIQUE:
            categorical_cols.append(col)
        elif n_unique >= FREE_TEXT_MIN_UNIQUE:
            free_text_cols.append(col)

    return numeric_cols, categorical_cols, free_text_cols


def select_entropy_kl_columns(res: pd.DataFrame, ora: pd.DataFrame) -> list:
    """
    Return columns where the oracle has between 2 and 9 valid categories
    (after excluding sentinel non‑response values).  These columns are used
    for entropy and KL‑divergence evaluation.
    """
    drop_keys = {NR_SENTINEL, "NAN", "NONE", "NAT", ""}
    cols = []
    for col in res.columns:
        clean = ora[col].astype(str)
        clean = clean[~clean.isin(drop_keys)]
        n_cats = clean.nunique()
        if 2 <= n_cats <= CATEGORICAL_MAX_UNIQUE:
            cols.append(col)
    return cols


# ==============================================================================
# UTILITIES
# ==============================================================================

def tokens_overlap(rv, ov) -> bool:
    """True iff the token sets of *rv* and *ov* have non‑empty intersection."""
    r_tok = set(tokenise(rv))
    o_tok = set(tokenise(ov))
    return bool(r_tok & o_tok)

def norm_str(s) -> str:
    """Lowercase + accent fold + collapse whitespace."""
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    if s in {"<nr>", "nan", "none", "nat", ""}:
        return ""
    for src, dst in [("éèêë","e"),("àâä","a"),("îï","i"),("ôö","o"),("ùûü","u")]:
        for ch in src:
            s = s.replace(ch, dst)
    for sep in (";", ",", "|", "/"):
        s = s.replace(sep, " ")
    return re.sub(r"\s+", " ", s)

def tokenise(text) -> list:
    t = norm_str(text)
    # Split on common multi-code delimiters
    for sep in (";", ",", "|", "/"):
        t = t.replace(sep, " ")
    return t.split()

def get_dist(series: pd.Series) -> dict:
    counts = series.dropna().value_counts()
    # Exclude sentinel non‑response values from the distribution
    drop_keys = {NR_SENTINEL, "NAN", "NONE", "NAT", ""}
    counts   = counts[~counts.index.isin(drop_keys)]
    total    = counts.sum()
    return {k: v / total for k, v in counts.items()} if total else {}

def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    va   = np.array([a.get(k, 0) for k in keys], dtype=float)
    vb   = np.array([b.get(k, 0) for k in keys], dtype=float)
    d    = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 0 else 0.0

def load_log(path: str) -> dict:
    info = {}
    if not path or not Path(path).exists():
        return info
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                k, _, v = line.partition(":")
                if k.strip() and v.strip():
                    info[k.strip()] = v.strip()
    return info


def entropy_norm(dist: dict) -> float:
    if len(dist) <= 1:
        return 0.0
    probs = np.array(list(dist.values()))
    probs = probs[probs > 0]
    H     = -np.sum(probs * np.log(probs))
    Hmax  = math.log(len(dist))
    return H / Hmax if Hmax > 0 else 0.0

def compute_entropy_table(res, ora, categorical_cols, min_cats=2, max_cats=9) -> pd.DataFrame:
    rows = []
    for col in categorical_cols:
        d_res = get_dist(res[col])
        d_ora = get_dist(ora[col])
        n_cats = len(d_ora)
        if n_cats < min_cats or n_cats > max_cats:
            continue
        if len(d_res) < 2:
            continue
        h_res = entropy_norm(d_res)
        h_ora = entropy_norm(d_ora)
        rows.append({
            "Variable":         col,
            "# Categories":     n_cats,
            "Entropy (Oracle)": round(h_ora, 4),
            "Entropy (Agent)":  round(h_res, 4),
            "Delta Entropy":    round(h_res - h_ora, 4),
        })
    return pd.DataFrame(rows)

# ==============================================================================
# METRIC B — KL DIVERGENCE
# ==============================================================================

def kl_div(p: dict, q: dict, eps: float = 1e-8) -> float:
    """KL(P || Q) where P=oracle, Q=agent."""
    keys  = set(p) | set(q)
    p_arr = np.array([p.get(k, 0) + eps for k in keys])
    q_arr = np.array([q.get(k, 0) + eps for k in keys])
    p_arr /= p_arr.sum()
    q_arr /= q_arr.sum()
    return float(scipy_entropy(p_arr, q_arr))

def compute_kl_table(res, ora, categorical_cols, min_cats=2, max_cats=9) -> pd.DataFrame:
    rows = []
    for col in categorical_cols:
        d_ora = get_dist(ora[col])
        d_res = get_dist(res[col])
        if not d_ora or not d_res:
            continue
        n_cats = len(d_ora)
        if n_cats < min_cats or n_cats > max_cats:
            continue
        if len(d_res) < 2:
            continue
        kl = kl_div(d_ora, d_res)
        rows.append({
            "Variable":                      col,
            "KL Divergence (Oracle || Agent)": round(kl, 4),
            "Rating": ("approx_perfect" if kl < 0.05 else
                       "good"           if kl < 0.5  else
                       "moderate"       if kl < 1.5  else "poor"),
        })
    return pd.DataFrame(rows)


# ==============================================================================
# METRIC C — NUMERIC COMPARISON
# ==============================================================================

def compute_numeric_table(res, ora, numeric_cols) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        r = pd.to_numeric(res[col], errors="coerce").dropna()
        o = pd.to_numeric(ora[col], errors="coerce").dropna()
        if r.empty or o.empty:
            continue
        n   = min(len(r), len(o))
        mae = round((r.iloc[:n] - o.iloc[:n]).abs().mean(), 3)
        rows.append({
            "Variable":      col,
            "Oracle Mean":   round(o.mean(), 2),
            "Agent Mean":    round(r.mean(), 2),
            "Oracle Median": round(o.median(), 2),
            "Agent Median":  round(r.median(), 2),
            "Oracle Std":    round(o.std(), 2),
            "Agent Std":     round(r.std(), 2),
            "MAE":           mae,
        })
    return pd.DataFrame(rows)


# ==============================================================================
# METRIC D — BLEU
# ==============================================================================

def compute_bleu(res, ora, free_text_cols) -> dict:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    sf = SmoothingFunction().method3

    per_col  = {}
    all_refs = []
    all_hyps = []

    for col in free_text_cols:
        refs, hyps = [], []
        for rv, ov in zip(res[col], ora[col]):
            r_tok = tokenise(rv)
            o_tok = tokenise(ov)
            if r_tok and o_tok:
                refs.append([o_tok])
                hyps.append(r_tok)
        if refs:
            per_col[col] = round(corpus_bleu(refs, hyps, smoothing_function=sf), 4)
            all_refs.extend(refs)
            all_hyps.extend(hyps)

    global_bleu = round(
        corpus_bleu(all_refs, all_hyps, smoothing_function=sf), 4
    ) if all_refs else 0.0
    return {"per_column": per_col, "global": global_bleu}

# ==============================================================================
# METRIC E — ROUGE
# ==============================================================================

def compute_rouge(res, ora, free_text_cols) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=False)
    per_col = {}
    agg     = {"rouge1":[], "rouge2":[], "rougeL":[]}

    for col in free_text_cols:
        r1s, r2s, rLs = [], [], []
        for rv, ov in zip(res[col], ora[col]):
            ref = norm_str(ov)
            hyp = norm_str(rv)
            if ref and hyp:
                s = scorer.score(ref, hyp)
                r1s.append(s["rouge1"].fmeasure)
                r2s.append(s["rouge2"].fmeasure)
                rLs.append(s["rougeL"].fmeasure)
        if r1s:
            per_col[col] = {
                "ROUGE-1": round(np.mean(r1s), 4),
                "ROUGE-2": round(np.mean(r2s), 4),
                "ROUGE-L": round(np.mean(rLs), 4),
            }
            agg["rouge1"].extend(r1s)
            agg["rouge2"].extend(r2s)
            agg["rougeL"].extend(rLs)

    return {
        "per_column": per_col,
        "global": {
            "ROUGE-1": round(np.mean(agg["rouge1"]), 4) if agg["rouge1"] else 0,
            "ROUGE-2": round(np.mean(agg["rouge2"]), 4) if agg["rouge2"] else 0,
            "ROUGE-L": round(np.mean(agg["rougeL"]), 4) if agg["rougeL"] else 0,
        },
    }


# ==============================================================================
# METRIC F — METEOR  (lightweight, no external corpus)
# ==============================================================================

def _meteor_pair(ref: str, hyp: str, alpha=0.9, gamma=0.5, beta=3) -> float:
    r_toks = tokenise(ref)
    h_toks = tokenise(hyp)
    if not r_toks or not h_toks:
        return 0.0
    r_cnt   = Counter(r_toks)
    h_cnt   = Counter(h_toks)
    matches = sum((r_cnt & h_cnt).values())
    if matches == 0:
        return 0.0
    P     = matches / len(h_toks)
    R     = matches / len(r_toks)
    denom = alpha * R + (1 - alpha) * P
    F     = (P * R) / denom if denom > 0 else 0.0
    # Chunk penalty
    used, pos = {}, []
    for i, t in enumerate(h_toks):
        if r_cnt.get(t, 0) > used.get(t, 0):
            used[t] = used.get(t, 0) + 1
            pos.append(i)
    chunks  = sum(1 for i in range(1, len(pos)) if pos[i] != pos[i-1]+1) + (1 if pos else 0)
    penalty = gamma * (chunks / matches) ** beta
    return max(0.0, F * (1 - penalty))

def compute_meteor(res, ora, free_text_cols) -> dict:
    per_col  = {}
    all_vals = []
    for col in free_text_cols:
        scores = [_meteor_pair(norm_str(ov), norm_str(rv))
                  for rv, ov in zip(res[col], ora[col])]
        if scores:
            per_col[col] = round(np.mean(scores), 4)
            all_vals.extend(scores)
    return {
        "per_column": per_col,
        "global": round(np.mean(all_vals), 4) if all_vals else 0.0,
    }


# ==============================================================================
# METRIC G — BERTScore
# ==============================================================================

def _char_ngram(text: str, n: int = 1) -> Counter:
    t = norm_str(text).replace(" ", "")
    return Counter(t[i:i+n] for i in range(len(t) - n + 1))

def compute_bertscore(res, ora, free_text_cols) -> dict:
    try:
        from bert_score import score as bert_score_fn
        USE_NEURAL = True
    except ImportError:
        USE_NEURAL = False

    per_col = {}
    all_F   = []

    for col in free_text_cols:
        pairs = [
            (norm_str(rv), norm_str(ov))
            for rv, ov in zip(res[col], ora[col])
            if norm_str(rv) and norm_str(ov)
        ]
        if not pairs:
            continue
        h_list, r_list = zip(*pairs)

        if USE_NEURAL:
            try:
                _, _, F_ = bert_score_fn(list(h_list), list(r_list),
                                         lang="en", verbose=False)
                f_vals = F_.tolist()
            except Exception:
                f_vals = [_cosine(_char_ngram(h, 1), _char_ngram(r, 1)) for h, r in pairs]
        else:
            f_vals = [_cosine(_char_ngram(h, 1), _char_ngram(r, 1)) for h, r in pairs]

        per_col[col] = round(np.mean(f_vals), 4)
        all_F.extend(f_vals)

    note = ("Neural BERTScore" if USE_NEURAL
            else "BERTScore approx. via char-1gram cosine")

    return {
        "per_column": per_col,
        "global": round(np.mean(all_F), 4) if all_F else 0.0,
        "note": note,
    }

# ==============================================================================
# METRIC H — SUCCESS RATE  (SR)
# ==============================================================================

SR_THRESHOLD = 0.70

def compute_sr(res) -> dict:
    """
    An agent 'succeeds' if >= SR_THRESHOLD of all matched survey columns
    have a non-null, non-empty response.
    """
    cols      = list(res.columns)
    per_agent = []
    for _, row in res.iterrows():
        filled = sum(
            1 for c in cols
            if not pd.isna(row[c])
            and str(row[c]).strip() not in ("", "NAN", "NONE", "NAT")
        )
        per_agent.append(filled / len(cols) if cols else 0)

    successes = [1 if s >= SR_THRESHOLD else 0 for s in per_agent]
    return {
        "Success Rate (SR)":   round(np.mean(successes), 4),
        "SR Threshold":        SR_THRESHOLD,
        "N Agents":            len(res),
        "N Successful":        int(sum(successes)),
        "Avg Field Coverage":  round(np.mean(per_agent), 4),
        "Min Field Coverage":  round(float(np.min(per_agent)), 4),
        "Max Field Coverage":  round(float(np.max(per_agent)), 4),
    }

# ==============================================================================
# METRIC I — TASK GOAL COMPLETION  (TGC)
# ==============================================================================

def compute_tgc(res, ora, categorical_cols, numeric_cols) -> dict:
    """
    TGC = weighted fraction of correctly answered fields per agent.
      - categorical: weight 1.0, exact normalised match
      - numeric:     weight 0.5, proximity score 1 - min(|a-o|/max(|o|,1), 1)
    """
    all_cols = categorical_cols + numeric_cols
    cat_set  = set(categorical_cols)
    total_w  = len(categorical_cols) * 1.0 + len(numeric_cols) * 0.5

    if total_w == 0:
        return {"Task Goal Completion (TGC)": "N/A"}

    per_agent = []
    for i, row in res.iterrows():
        ora_row = ora.iloc[i]
        score   = 0.0
        for col in categorical_cols:
            rv = row[col]
            ov = ora_row[col]
            ovn = norm_str(ov)
            if ovn:
                score += 1.0 if tokens_overlap(rv, ov) else 0.0
        for col in numeric_cols:
            rv = pd.to_numeric(row[col], errors="coerce")
            ov = pd.to_numeric(ora_row[col], errors="coerce")
            if not pd.isna(ov) and not pd.isna(rv):
                score += 0.5 * max(0.0, 1.0 - abs(rv - ov) / max(abs(ov), 1))
        per_agent.append(score / total_w)

    return {
        "Task Goal Completion (TGC)":  round(np.mean(per_agent), 4),
        "Median TGC":                  round(float(np.median(per_agent)), 4),
        "Std TGC":                     round(float(np.std(per_agent)), 4),
        "Min TGC":                     round(float(np.min(per_agent)), 4),
        "Max TGC":                     round(float(np.max(per_agent)), 4),
        "Cols evaluated":              len(all_cols),
        "Total weight":                round(total_w, 2),
    }


# ==============================================================================
# METRIC J — FACTUAL CORRECTNESS
# ==============================================================================

def compute_factual(res, ora, categorical_cols) -> dict:
    """Exact normalised-match accuracy for every categorical column.

    Rows where the oracle is NA or contains the NR sentinel are *skipped*
    (neither counted as correct nor incorrect).
    """
    per_col  = {}
    all_accs = []
    skip_vals = frozenset({"", "nan", NR_SENTINEL.lower()})

    for col in categorical_cols:
        correct = 0
        total   = 0
        for rv, ov in zip(res[col], ora[col]):
            ov_str = str(ov).strip().lower()
            if pd.isna(ov) or ov_str in skip_vals:
                continue
            total   += 1
            correct += int(tokens_overlap(rv, ov))
        acc = correct / total if total else 0.0
        per_col[col] = {"Accuracy": round(acc, 4), "N_oracle": total}
        if total > 0:
            all_accs.append(acc)

    return {
        "per_column":      per_col,
        "global_accuracy": round(np.mean(all_accs), 4) if all_accs else 0.0,
        "median_accuracy": round(float(np.median(all_accs)), 4) if all_accs else 0.0,
    }


# ==============================================================================
# METRIC K — RESPONSE RELEVANCE  (TF-IDF cosine)
# ==============================================================================

def _tfidf(corpus: list) -> list:
    N    = len(corpus)
    toks = [tokenise(d) for d in corpus]
    df   = Counter(t for doc in toks for t in set(doc))
    vecs = []
    for doc in toks:
        tf  = Counter(doc)
        tot = max(len(doc), 1)
        vec = Counter({
            t: (c / tot) * math.log((N + 1) / (df[t] + 1))
            for t, c in tf.items()
        })
        vecs.append(vec)
    return vecs

def compute_relevance(res, ora, free_text_cols) -> dict:
    per_col  = {}
    all_sims = []
    for col in free_text_cols:
        r_vals = [norm_str(v) for v in res[col]]
        o_vals = [norm_str(v) for v in ora[col]]
        corpus = r_vals + o_vals
        vecs   = _tfidf(corpus)
        n      = len(r_vals)
        sims   = [_cosine(vecs[i], vecs[i + n]) for i in range(n)]
        per_col[col] = round(np.mean(sims), 4)
        all_sims.extend(sims)
    return {
        "per_column":    per_col,
        "global_mean":   round(np.mean(all_sims), 4)          if all_sims else 0.0,
        "global_median": round(float(np.median(all_sims)), 4) if all_sims else 0.0,
    }

# ==============================================================================
# REPORT HELPERS
# ==============================================================================

SEP  = "=" * 80
SEP2 = "-" * 80

def tbl(df, fmt="simple") -> str:
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return "(no data)"
    return tabulate(df, headers="keys", tablefmt=fmt, showindex=False)

def _kv(d: dict) -> str:
    return "\n".join(f"  {str(k):<40} {v}" for k, v in d.items())

def build_summary(log_info, n_agents, col_map,
                  categorical_cols, numeric_cols, free_text_cols,
                  entropy_df, kl_df, bleu, rouge, meteor, bertscore,
                  sr, tgc, factual, relevance) -> pd.DataFrame:

    kl_vals = kl_df["KL Divergence (Oracle || Agent)"] if not kl_df.empty else pd.Series([0])
    en_ora  = entropy_df["Entropy (Oracle)"]             if not entropy_df.empty else pd.Series([0])
    en_agt  = entropy_df["Entropy (Agent)"]              if not entropy_df.empty else pd.Series([0])

    rows = [
        ("Run Info",      "Model",                  log_info.get("chat_model",      "—")),
        ("Run Info",      "Temperature",            log_info.get("temperature",     "—")),
        ("Run Info",      "N Agents",               n_agents),
        ("Run Info",      "Duration (h)",           log_info.get("Duration (hours)","—")),
        ("Run Info",      "Start Time",             log_info.get("Start Time",      "—")),
        ("Run Info",      "End Time",               log_info.get("End Time",        "—")),
        ("Run Info",      "Sample Size",            log_info.get("sample_size",     "—")),
        ("Run Info",      "Batch Size",             log_info.get("batch_size",      "—")),
        ("Run Info",      "Source",                 log_info.get("source",          "—")),
        ("Run Info",      "Sim Year",               log_info.get("sim_year",        "—")),
        ("Columns",       "Matched (results<->oracle)", len(col_map)),
        ("Columns",       "  -> Categorical",       len(categorical_cols)),
        ("Columns",       "  -> Numeric",           len(numeric_cols)),
        ("Columns",       "  -> Free-text",         len(free_text_cols)),
        (SEP2[:12], SEP2[:22], SEP2[:28]),
        ("Entropy",       "Oracle Median",          round(en_ora.median(),  4)),
        ("Entropy",       "Agent Median",           round(en_agt.median(),  4)),
        ("Entropy",       "Agent Mean",             round(en_agt.mean(),    4)),
        ("KL Divergence", "Median KL",              round(kl_vals.median(), 4)),
        ("KL Divergence", "Mean KL",                round(kl_vals.mean(),   4)),
        (SEP2[:12], SEP2[:22], SEP2[:28]),
        ("NLP - BLEU",       "Corpus BLEU",         bleu["global"]),
        ("NLP - ROUGE",      "ROUGE-1",             rouge["global"]["ROUGE-1"]),
        ("NLP - ROUGE",      "ROUGE-2",             rouge["global"]["ROUGE-2"]),
        ("NLP - ROUGE",      "ROUGE-L",             rouge["global"]["ROUGE-L"]),
        ("NLP - METEOR",     "METEOR",              meteor["global"]),
        ("NLP - BERTScore",  "F1",                  bertscore["global"]),
        (SEP2[:12], SEP2[:22], SEP2[:28]),
        ("Task Metrics",  "Success Rate (SR)",      sr.get("Success Rate (SR)",          "—")),
        ("Task Metrics",  "N Successful / N Total", f"{sr.get('N Successful','?')} / {sr.get('N Agents','?')}"),
        ("Task Metrics",  "Avg Field Coverage",     sr.get("Avg Field Coverage",         "—")),
        ("Task Metrics",  "Task Goal Completion",   tgc.get("Task Goal Completion (TGC)","—")),
        ("Task Metrics",  "Median TGC",             tgc.get("Median TGC",                "—")),
        (SEP2[:12], SEP2[:22], SEP2[:28]),
        ("Factual",       "Global Accuracy",        factual["global_accuracy"]),
        ("Factual",       "Median Accuracy",        factual["median_accuracy"]),
        ("Relevance",     "Global Mean (TF-IDF)",   relevance["global_mean"]),
        ("Relevance",     "Global Median (TF-IDF)", relevance["global_median"]),
    ]
    return pd.DataFrame(rows, columns=["Category", "Metric", "Value"])


def build_nlp_table(bleu, rouge, meteor, bertscore) -> pd.DataFrame:
    all_cols = sorted(
        set(bleu["per_column"]) | set(rouge["per_column"])
        | set(meteor["per_column"]) | set(bertscore["per_column"])
    )
    rows = []
    for col in all_cols:
        rows.append({
            "Field":        col,
            "BLEU":         bleu["per_column"].get(col, "—"),
            "ROUGE-1":      rouge["per_column"].get(col, {}).get("ROUGE-1", "—"),
            "ROUGE-2":      rouge["per_column"].get(col, {}).get("ROUGE-2", "—"),
            "ROUGE-L":      rouge["per_column"].get(col, {}).get("ROUGE-L", "—"),
            "METEOR":       meteor["per_column"].get(col, "—"),
            "BERTScore F1": bertscore["per_column"].get(col, "—"),
        })
    return pd.DataFrame(rows)


def build_factual_table(factual) -> pd.DataFrame:
    rows = [
        {"Field": col, "Accuracy": v["Accuracy"], "N (oracle)": v["N_oracle"]}
        for col, v in factual["per_column"].items()
    ]
    df = pd.DataFrame(rows)
    return df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

def build_relevance_table(relevance) -> pd.DataFrame:
    rows = [{"Field": col, "TF-IDF Cosine Sim.": v}
            for col, v in relevance["per_column"].items()]
    if not rows:
        return pd.DataFrame(columns=["Field", "TF-IDF Cosine Sim."])
    df = pd.DataFrame(rows)
    return df.sort_values("TF-IDF Cosine Sim.", ascending=False).reset_index(drop=True)


def write_report(out_path: str, sections: list):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for title, content in sections:
            f.write(f"\n{SEP}\n{title}\n{SEP}\n{content}\n")
    print(f"  -> Report saved: {out_path}")


def _find_file(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None


def discover_runs(results_dir: str, target_dataset: str = None) -> list[dict]:
    """Scan results/<model>/<dataset>/ for evaluation runs.

    If *target_dataset* is given, only collect runs for that dataset.
    Returns a list of dicts with keys: model, dataset, results, oracle, log
    """
    base = Path(results_dir)
    if not base.is_dir():
        return []

    runs = []
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        model_name = model_dir.name

        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
                continue
            dataset_name = dataset_dir.name
            if target_dataset and dataset_name != target_dataset:
                continue

            oracle_dir = dataset_dir / "oracle"
            agents_dir = dataset_dir / "agents"

            oracle_file = _find_file(oracle_dir, "sampled_oracle.csv")
            results_file = _find_file(agents_dir, "*_results.csv")
            if results_file is None:
                results_file = _find_file(dataset_dir, "*_results.csv")
            log_file = _find_file(agents_dir, "log.txt")
            if log_file is None:
                log_file = _find_file(dataset_dir, "log.txt")

            if oracle_file and results_file:
                runs.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "results": str(results_file),
                    "oracle": str(oracle_file),
                    "log": str(log_file) if log_file else "",
                })

    return runs


def _load_dataset_dictionary(dataset: str):
    """Load label map for a specific dataset and reset the global cache."""
    global _LABEL_MAP
    dict_path = str(Path(__file__).parent /
                    f"configs/Generic/{dataset}/dictionary.csv")
    _LABEL_MAP = load_label_map(dict_path)


def run_single_evaluation(run_info: dict, out_dir: Path) -> dict:
    """Run the full evaluation pipeline for a single model/dataset pair."""
    global CATEGORICAL_MAX_UNIQUE, FREE_TEXT_MIN_UNIQUE, NR_THRESHOLD, EXCLUDED_COLS, AGENT_VAR_MIN_UNIQUE

    results_path = run_info["results"]
    oracle_path = run_info["oracle"]
    log_path = run_info["log"]
    model = run_info["model"]
    dataset = run_info["dataset"]

    # Load dataset-specific dictionary
    _load_dataset_dictionary(dataset)

    print(f"\n{'='*60}")
    print(f"  Evaluating: {model} / {dataset}")
    print(f"{'='*60}")

    print("[1/11] Loading and mapping columns ...")
    res, ora, col_map, meta_cols, n = load_and_map(results_path, oracle_path)
    log_info = load_log(log_path)
    print(f"       {n} rows  |  {len(col_map)} shared cols  |  "
          f"excluded metadata: {meta_cols}")

    print("[1b/11] Resolving code:label format in agent responses ...")
    res, ora = resolve_code_label_format(res, ora)

    print("[1c/11] Binarizing multi-select indicator columns ...")
    binarized = binarize_flag_cols(res, ora)
    if binarized:
        print(f"       Binarized {len(binarized)} flag columns")

    print("[1d/11] Identifying and removing NR/NA-heavy columns ...")
    nr_cols = identify_nr_columns(ora, NR_THRESHOLD)
    if nr_cols:
        print(f"       Dropping {len(nr_cols)} columns with >{NR_THRESHOLD:.0%} NR/NA")
        res.drop(columns=[c for c in nr_cols if c in res.columns], inplace=True)
        ora.drop(columns=[c for c in nr_cols if c in ora.columns], inplace=True)
    else:
        print(f"       No columns exceed {NR_THRESHOLD:.0%} NR/NA")

    print("[1e/11] Removing known mismatched (persona) columns ...")
    to_drop = [c for c in EXCLUDED_COLS if c in ora.columns]
    if to_drop:
        print(f"       Dropping {len(to_drop)} columns with mismatched content: {to_drop}")
        res.drop(columns=to_drop, inplace=True)
        ora.drop(columns=to_drop, inplace=True)
    else:
        print(f"       No mismatched columns found")

    # Snapshot pre-filter copies for NLP metrics (before agent-constant removal)
    res_nlp = res.copy()
    ora_nlp = ora.copy()
    _, _, free_text_pre = classify_columns(res_nlp, ora_nlp)
    nlp_all_cols = [c for c in res_nlp.columns if c in ora_nlp.columns]

    print("[1f/11] Removing agent-constant (non-varying) columns ...")
    agent_const = identify_agent_constant_columns(res, ora)
    if agent_const:
        print(f"       Dropping {len(agent_const)} columns where agent output does not vary")
        res.drop(columns=[c for c in agent_const if c in res.columns], inplace=True)
        ora.drop(columns=[c for c in agent_const if c in ora.columns], inplace=True)
    else:
        print(f"       All columns pass the agent-variability check")

    print("[1g/11] Removing zero-overlap columns (agent/oracle code sets disjoint) ...")
    zero_ov = identify_zero_overlap_columns(res, ora)
    if zero_ov:
        print(f"       Dropping {len(zero_ov)} columns where agent code space does not overlap oracle")
        res.drop(columns=[c for c in zero_ov if c in res.columns], inplace=True)
        ora.drop(columns=[c for c in zero_ov if c in ora.columns], inplace=True)
    else:
        print(f"       All columns pass the zero-overlap check")

    print("[2/11] Auto-classifying columns ...")
    numeric_cols, categorical_cols, free_text_cols = classify_columns(res, ora)

    print(f"[2b/11] Selecting columns for entropy/KL (2-{CATEGORICAL_MAX_UNIQUE} oracle categories) ...")
    entropy_kl_cols = select_entropy_kl_columns(res, ora)
    print(f"       {len(entropy_kl_cols)} columns selected")

    print("[3/11] Shannon Entropy ...")
    entropy_df = compute_entropy_table(res, ora, entropy_kl_cols, max_cats=CATEGORICAL_MAX_UNIQUE)

    print("[4/11] KL Divergence ...")
    kl_df = compute_kl_table(res, ora, entropy_kl_cols, max_cats=CATEGORICAL_MAX_UNIQUE)

    print("[5/11] Numeric comparison ...")
    num_df = compute_numeric_table(res, ora, numeric_cols)

    print("[6/11] BLEU (pre-filter free-text columns) ...")
    bleu = compute_bleu(res_nlp, ora_nlp, free_text_pre)
    print("[7/11] ROUGE (pre-filter free-text columns) ...")
    rouge = compute_rouge(res_nlp, ora_nlp, free_text_pre)
    print("[8/11] METEOR (pre-filter free-text columns) ...")
    meteor = compute_meteor(res_nlp, ora_nlp, free_text_pre)
    print(f"[9/11] BERTScore (all {len(nlp_all_cols)} pre-filter columns) ...")
    bertscore = compute_bertscore(res_nlp, ora_nlp, nlp_all_cols)

    print("[10/11] SR / TGC / Factual / Relevance ...")
    sr = compute_sr(res)
    tgc = compute_tgc(res, ora, categorical_cols, numeric_cols)
    factual = compute_factual(res, ora, categorical_cols)
    relevance = compute_relevance(res, ora, free_text_cols)

    report_dir = out_dir / model / dataset
    report_path = report_dir / "evaluation_report.txt"

    print("[11/11] Assembling report ...")
    summary_df = build_summary(
        log_info, n, col_map,
        categorical_cols, numeric_cols, free_text_cols,
        entropy_df, kl_df, bleu, rouge, meteor, bertscore,
        sr, tgc, factual, relevance,
    )
    nlp_df = build_nlp_table(bleu, rouge, meteor, bertscore)
    factual_df = build_factual_table(factual)
    relevance_df = build_relevance_table(relevance)

    col_class_txt = (
        f"Categorical      ({len(categorical_cols)}): {categorical_cols}\n"
        f"Numeric          ({len(numeric_cols)}):     {numeric_cols}\n"
        f"Free-text        ({len(free_text_cols)}):   {free_text_cols}\n"
        f"Entropy/KL cols  ({len(entropy_kl_cols)}):  {entropy_kl_cols}\n"
    )

    header_txt = (
        f"Model   : {model}\n"
        f"Dataset : {dataset}\n"
        f"Results : {Path(results_path).name}\n"
        f"Oracle  : {Path(oracle_path).name}\n"
        f"Log     : {Path(log_path).name if log_path else 'N/A'}\n"
        f"{SEP2}"
    )

    write_report(str(report_path), [
        ("HEADER", header_txt),
        ("TABLE 0 — GLOBAL SUMMARY", tbl(summary_df)),
        ("COLUMN CLASSIFICATION  (auto-inferred from data)", col_class_txt),
        ("TABLE 1 — NORMALIZED SHANNON ENTROPY", tbl(entropy_df)),
        ("TABLE 2 — KL DIVERGENCE  P(Oracle) || Q(Agent)", tbl(kl_df)),
        ("TABLE 3 — NUMERIC VARIABLE COMPARISON", tbl(num_df)),
        ("TABLE 4 — NLP METRICS PER FREE-TEXT FIELD\n"
         "           BLEU / ROUGE / METEOR / BERTScore", tbl(nlp_df)),
        ("TABLE 5 — SUCCESS RATE  (SR)", _kv(sr)),
        ("TABLE 6 — TASK GOAL COMPLETION  (TGC)", _kv(tgc)),
        ("TABLE 7 — FACTUAL CORRECTNESS  (exact norm. match)", tbl(factual_df)),
        ("TABLE 8 — RESPONSE RELEVANCE  (TF-IDF cosine)", tbl(relevance_df)),
    ])

    print(f"\n  GLOBAL SUMMARY — {model} / {dataset}")
    print(tbl(summary_df))

    return {
        "model": model,
        "dataset": dataset,
        "n_agents": n,
        "n_matched_cols": len(col_map),
        "n_categorical": len(categorical_cols),
        "n_numeric": len(numeric_cols),
        "n_free_text": len(free_text_cols),
        "entropy_oracle_median": round(entropy_df["Entropy (Oracle)"].median(), 4) if not entropy_df.empty else 0,
        "entropy_agent_median": round(entropy_df["Entropy (Agent)"].median(), 4) if not entropy_df.empty else 0,
        "kl_median": round(kl_df["KL Divergence (Oracle || Agent)"].median(), 4) if not kl_df.empty else 0,
        "bleu_global": bleu["global"],
        "rouge1": rouge["global"]["ROUGE-1"],
        "rouge2": rouge["global"]["ROUGE-2"],
        "rougeL": rouge["global"]["ROUGE-L"],
        "meteor": meteor["global"],
        "bertscore_f1": bertscore["global"],
        "success_rate": sr.get("Success Rate (SR)", 0),
        "avg_field_coverage": sr.get("Avg Field Coverage", 0),
        "tgc": tgc.get("Task Goal Completion (TGC)", 0),
        "median_tgc": tgc.get("Median TGC", 0),
        "factual_accuracy": factual["global_accuracy"],
        "median_factual_accuracy": factual["median_accuracy"],
        "relevance_mean": relevance["global_mean"],
        "relevance_median": relevance["global_median"],
        "report_path": str(report_path),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    global CATEGORICAL_MAX_UNIQUE, FREE_TEXT_MIN_UNIQUE, AGENT_VAR_MIN_UNIQUE
    ap = argparse.ArgumentParser(description="Dynamic LLM Agent Evaluation Suite (multi-dataset)")

    ap.add_argument("--results-dir", type=str, default=None,
                    help="Auto-discover runs under this directory (default: results/)")
    ap.add_argument("--out-dir", type=str, default="evaluations",
                    help="Output directory for reports and summary (default: evaluations/)")
    ap.add_argument("--dataset", type=str, default=None,
                    help="Limit to a specific dataset (e.g. data_lyon_EDGT)")

    ap.add_argument("--results", default="",
                    help="Single-run: results CSV path")
    ap.add_argument("--oracle", default="",
                    help="Single-run: oracle CSV path")
    ap.add_argument("--log", default="",
                    help="Single-run: log file path")
    ap.add_argument("--out", default="",
                    help="Single-run: output report path")

    ap.add_argument("--cat-max-unique", type=int, default=CATEGORICAL_MAX_UNIQUE,
                    help=f"Max distinct values to treat a column as categorical (default {CATEGORICAL_MAX_UNIQUE})")
    ap.add_argument("--free-text-min-unique", type=int, default=FREE_TEXT_MIN_UNIQUE,
                    help=f"Min distinct values needed to treat a column as free-text (default {FREE_TEXT_MIN_UNIQUE})")
    ap.add_argument("--nr-threshold", type=float, default=0.9,
                    help="Drop columns where oracle NR/NA rate exceeds this (default 0.9)")
    ap.add_argument("--agent-var-min-unique", type=int, default=AGENT_VAR_MIN_UNIQUE,
                    help=f"Min unique values for agent output to keep column (default {AGENT_VAR_MIN_UNIQUE})")
    args = ap.parse_args()

    CATEGORICAL_MAX_UNIQUE = args.cat_max_unique
    FREE_TEXT_MIN_UNIQUE = args.free_text_min_unique
    NR_THRESHOLD = args.nr_threshold
    AGENT_VAR_MIN_UNIQUE = args.agent_var_min_unique

    print(f"\n{SEP}")
    print("  LLM TRANSPORTATION AGENT — DYNAMIC EVALUATION SUITE (multi-dataset)")
    print(SEP)

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # ── Batch mode ──────────────────────────────────────────────────────
    if args.results_dir:
        runs = discover_runs(args.results_dir, target_dataset=args.dataset)
        if not runs:
            ds = f" for '{args.dataset}'" if args.dataset else ""
            sys.exit(f"[ERROR] No evaluation runs found{ds} under {args.results_dir}/")

        # Group by dataset
        datasets_seen = sorted(set(r["dataset"] for r in runs))
        print(f"\n  Discovered {len(runs)} evaluation run(s) across {len(datasets_seen)} dataset(s):")
        for ds in datasets_seen:
            models_ds = [r["model"] for r in runs if r["dataset"] == ds]
            print(f"    {ds} ({len(models_ds)} models): {', '.join(models_ds)}")
        print()

        all_results = []
        for run_info in runs:
            try:
                result = run_single_evaluation(run_info, out_base)
                all_results.append(result)
            except Exception as e:
                model = run_info["model"]
                dataset = run_info["dataset"]
                print(f"\n  [ERROR] Evaluation failed for {model}/{dataset}: {e}")
                import traceback; traceback.print_exc()
                all_results.append({
                    "model": model,
                    "dataset": dataset,
                    "error": str(e),
                })

        # Build per-dataset summary CSVs
        for ds in datasets_seen:
            ds_results = [r for r in all_results if r.get("dataset") == ds]
            if not ds_results:
                continue
            summary_rows = []
            for r in ds_results:
                row = {k: r.get(k, "") for k in [
                    "model", "dataset", "n_agents", "n_matched_cols",
                    "n_categorical", "n_numeric", "n_free_text",
                    "entropy_oracle_median", "entropy_agent_median",
                    "kl_median", "bleu_global", "rouge1", "rouge2", "rougeL",
                    "meteor", "bertscore_f1", "success_rate", "avg_field_coverage",
                    "tgc", "median_tgc", "factual_accuracy", "median_factual_accuracy",
                    "relevance_mean", "relevance_median", "report_path",
                ]}
                if "error" in r:
                    row["error"] = r["error"]
                summary_rows.append(row)

            summary_csv = out_base / f"summary_{ds}.csv"
            pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
            print(f"\n  Global summary saved: {summary_csv}")

            # Console comparison table
            print(f"\n{'='*60}")
            print(f"  GLOBAL COMPARISON  ({ds})")
            print(f"{'='*60}")
            comp_cols = ["model", "dataset", "n_agents", "success_rate", "tgc",
                         "factual_accuracy", "bleu_global", "rougeL", "meteor", "bertscore_f1"]
            comp_df = pd.DataFrame(summary_rows)[comp_cols]
            print(tbl(comp_df))
        return

    # ── Single-run mode (backward compatible) ──────────────────────────
    if not args.results or not args.oracle:
        sys.exit("[ERROR] Either --results-dir (batch) or --results + --oracle (single) required.")

    # Infer dataset from results path for dictionary loading
    results_path = args.results
    oracle_path = args.oracle
    log_path = args.log if args.log else ""
    out_path = args.out if args.out else "evaluation_report.txt"

    inferred_dataset = args.dataset
    if not inferred_dataset:
        for known_dataset in ["data_lyon_EDGT", "data_MyDailyTravelData", "data_NYC_mobility",
                               "data_VTC_survey", "data_pmus_yaounde", "data_hanoi"]:
            if known_dataset in results_path:
                inferred_dataset = known_dataset
                break
    if not inferred_dataset:
        inferred_dataset = "data_MyDailyTravelData"
    _load_dataset_dictionary(inferred_dataset)

    print(f"\n[1/11] Loading and mapping columns ...")
    res, ora, col_map, meta_cols, n = load_and_map(results_path, oracle_path)
    log_info = load_log(log_path)
    print(f"       {n} rows  |  {len(col_map)} shared cols  |  "
          f"excluded metadata: {meta_cols}")

    print("[1b/11] Resolving code:label format in agent responses ...")
    res, ora = resolve_code_label_format(res, ora)

    print("[1c/11] Binarizing multi-select indicator columns ...")
    binarized = binarize_flag_cols(res, ora)
    if binarized:
        print(f"       Binarized {len(binarized)} flag columns")

    print("[1d/11] Identifying and removing NR/NA-heavy columns ...")
    nr_cols = identify_nr_columns(ora, NR_THRESHOLD)
    if nr_cols:
        print(f"       Dropping {len(nr_cols)} columns with >{NR_THRESHOLD:.0%} NR/NA")
        res.drop(columns=[c for c in nr_cols if c in res.columns], inplace=True)
        ora.drop(columns=[c for c in nr_cols if c in ora.columns], inplace=True)
    else:
        print(f"       No columns exceed {NR_THRESHOLD:.0%} NR/NA")

    print("[1e/11] Removing known mismatched (persona) columns ...")
    to_drop = [c for c in EXCLUDED_COLS if c in ora.columns]
    if to_drop:
        print(f"       Dropping {len(to_drop)} columns with mismatched content: {to_drop}")
        res.drop(columns=to_drop, inplace=True)
        ora.drop(columns=to_drop, inplace=True)
    else:
        print(f"       No mismatched columns found")

    print("[2/11] Auto-classifying columns ...")
    numeric_cols, categorical_cols, free_text_cols = classify_columns(res, ora)
    print(f"       Categorical : {categorical_cols}")
    print(f"       Numeric     : {numeric_cols}")
    print(f"       Free-text   : {free_text_cols}")

    print("[2b/11] Selecting columns for entropy/KL (2-9 oracle categories) ...")
    entropy_kl_cols = select_entropy_kl_columns(res, ora)
    print(f"       Entropy/KL cols : {entropy_kl_cols}")

    print("[3/11] Shannon Entropy ...")
    entropy_df = compute_entropy_table(res, ora, entropy_kl_cols)

    print("[4/11] KL Divergence ...")
    kl_df = compute_kl_table(res, ora, entropy_kl_cols)

    print("[5/11] Numeric comparison ...")
    num_df = compute_numeric_table(res, ora, numeric_cols)

    print("[6/11] BLEU ...")
    bleu = compute_bleu(res, ora, free_text_cols)
    print("[7/11] ROUGE ...")
    rouge = compute_rouge(res, ora, free_text_cols)
    print("[8/11] METEOR ...")
    meteor = compute_meteor(res, ora, free_text_cols)
    print("[9/11] BERTScore ...")
    bertscore = compute_bertscore(res, ora, free_text_cols)
    print(f"       {bertscore['note']}")

    print("[10/11] SR / TGC / Factual / Relevance ...")
    sr = compute_sr(res)
    tgc = compute_tgc(res, ora, categorical_cols, numeric_cols)
    factual = compute_factual(res, ora, categorical_cols)
    relevance = compute_relevance(res, ora, free_text_cols)

    print("[11/11] Assembling report ...")
    summary_df = build_summary(
        log_info, n, col_map,
        categorical_cols, numeric_cols, free_text_cols,
        entropy_df, kl_df, bleu, rouge, meteor, bertscore,
        sr, tgc, factual, relevance,
    )
    nlp_df = build_nlp_table(bleu, rouge, meteor, bertscore)
    factual_df = build_factual_table(factual)
    relevance_df = build_relevance_table(relevance)

    col_class_txt = (
        f"Categorical      ({len(categorical_cols)}): {categorical_cols}\n"
        f"Numeric          ({len(numeric_cols)}):     {numeric_cols}\n"
        f"Free-text        ({len(free_text_cols)}):   {free_text_cols}\n"
        f"Entropy/KL cols  ({len(entropy_kl_cols)}):  {entropy_kl_cols}\n"
    )

    header_txt = (
        f"Results : {Path(results_path).name}\n"
        f"Oracle  : {Path(oracle_path).name}\n"
        f"Log     : {Path(log_path).name if log_path else 'N/A'}\n"
        f"{SEP2}"
    )

    write_report(out_path, [
        ("HEADER", header_txt),
        ("TABLE 0 — GLOBAL SUMMARY", tbl(summary_df)),
        ("COLUMN CLASSIFICATION  (auto-inferred from data)", col_class_txt),
        ("TABLE 1 — NORMALIZED SHANNON ENTROPY", tbl(entropy_df)),
        ("TABLE 2 — KL DIVERGENCE  P(Oracle) || Q(Agent)", tbl(kl_df)),
        ("TABLE 3 — NUMERIC VARIABLE COMPARISON", tbl(num_df)),
        ("TABLE 4 — NLP METRICS PER FREE-TEXT FIELD\n"
         "           BLEU / ROUGE / METEOR / BERTScore", tbl(nlp_df)),
        ("TABLE 5 — SUCCESS RATE  (SR)", _kv(sr)),
        ("TABLE 6 — TASK GOAL COMPLETION  (TGC)", _kv(tgc)),
        ("TABLE 7 — FACTUAL CORRECTNESS  (exact norm. match)", tbl(factual_df)),
        ("TABLE 8 — RESPONSE RELEVANCE  (TF-IDF cosine)", tbl(relevance_df)),
    ])

    print(f"\n{SEP}")
    print("  GLOBAL SUMMARY")
    print(SEP)
    print(tbl(summary_df))

    if not nlp_df.empty:
        print(f"\n{SEP}")
        print("  NLP METRICS PER FREE-TEXT FIELD")
        print(SEP)
        print(tbl(nlp_df))

    print(f"\n{SEP}")
    print("  FACTUAL CORRECTNESS PER COLUMN")
    print(SEP)
    print(tbl(factual_df))
    print()


if __name__ == "__main__":
    main()
