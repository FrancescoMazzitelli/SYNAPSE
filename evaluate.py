"""
USAGE
-----
  python evaluate.py \\
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

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from scipy.stats import entropy as scipy_entropy
from tabulate import tabulate

warnings.filterwarnings("ignore")


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
# STEP 2 — AUTO-CLASSIFY COLUMNS
# ==============================================================================

# Thresholds (overridable via CLI)
CATEGORICAL_MAX_UNIQUE = 20    # <= N distinct values in oracle -> categorical
FREE_TEXT_MIN_UNIQUE   = 5     # >= N distinct values in oracle -> free-text


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

# ==============================================================================
# UTILITIES
# ==============================================================================

def norm_str(s) -> str:
    """Lowercase + accent fold + collapse whitespace."""
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    for src, dst in [("éèêë","e"),("àâä","a"),("îï","i"),("ôö","o"),("ùûü","u")]:
        for ch in src:
            s = s.replace(ch, dst)
    return re.sub(r"\s+", " ", s)

def tokenise(text) -> list:
    return norm_str(text).split()

def get_dist(series: pd.Series) -> dict:
    counts = series.dropna().value_counts()
    total  = counts.sum()
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


def compute_entropy_table(res, ora, categorical_cols) -> pd.DataFrame:
    rows = []
    for col in categorical_cols:
        d_res = get_dist(res[col])
        d_ora = get_dist(ora[col])
        h_res = entropy_norm(d_res)
        h_ora = entropy_norm(d_ora)
        rows.append({
            "Variable":         col,
            "# Categories":     len(d_ora),
            "Entropy (Oracle)": round(h_ora, 4),
            "Entropy (Agent)":  round(h_res, 4),
            "Delta Entropy":    round(h_res - h_ora, 4),
        })
    return pd.DataFrame(rows)

# ==============================================================================
# METRIC A — NORMALIZED SHANNON ENTROPY
# ==============================================================================

def entropy_norm(dist: dict) -> float:
    if len(dist) <= 1:
        return 0.0
    probs = np.array(list(dist.values()))
    probs = probs[probs > 0]
    H     = -np.sum(probs * np.log(probs))
    Hmax  = math.log(len(dist))
    return H / Hmax if Hmax > 0 else 0.0

def compute_entropy_table(res, ora, categorical_cols) -> pd.DataFrame:
    rows = []
    for col in categorical_cols:
        d_res = get_dist(res[col])
        d_ora = get_dist(ora[col])
        h_res = entropy_norm(d_res)
        h_ora = entropy_norm(d_ora)
        rows.append({
            "Variable":         col,
            "# Categories":     len(d_ora),
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

def compute_kl_table(res, ora, categorical_cols) -> pd.DataFrame:
    rows = []
    for col in categorical_cols:
        d_res = get_dist(res[col])
        d_ora = get_dist(ora[col])
        if not d_ora or not d_res:
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

def _char_ngram(text: str, n: int = 3) -> Counter:
    t = norm_str(text).replace(" ", "_")
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
            _, _, F_ = bert_score_fn(list(h_list), list(r_list),
                                     lang="fr", verbose=False)
            f_vals = F_.tolist()
        else:
            f_vals = [_cosine(_char_ngram(h), _char_ngram(r)) for h, r in pairs]

        per_col[col] = round(np.mean(f_vals), 4)
        all_F.extend(f_vals)

    note = ("Neural BERTScore" if USE_NEURAL
            else "BERTScore approx. via char-3gram cosine (pip install bert-score for neural)")

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
            rv = norm_str(row[col])
            ov = norm_str(ora_row[col])
            if ov:
                score += 1.0 if rv == ov else 0.0
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
    """Exact normalised-match accuracy for every categorical column."""
    per_col  = {}
    all_accs = []

    for col in categorical_cols:
        correct = 0
        total   = 0
        for rv, ov in zip(res[col], ora[col]):
            if not pd.isna(ov) and str(ov).strip() not in ("", "NAN"):
                total   += 1
                correct += int(norm_str(rv) == norm_str(ov))
        acc = correct / total if total else 0.0
        per_col[col] = {"Accuracy": round(acc, 4), "N_oracle": total}
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
    df = pd.DataFrame(rows)
    return df.sort_values("TF-IDF Cosine Sim.", ascending=False).reset_index(drop=True)


def write_report(out_path: str, sections: list):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for title, content in sections:
            f.write(f"\n{SEP}\n{title}\n{SEP}\n{content}\n")
    print(f"  -> Report saved: {out_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    global CATEGORICAL_MAX_UNIQUE, FREE_TEXT_MIN_UNIQUE
    ap = argparse.ArgumentParser(description="Dynamic LLM Agent Evaluation Suite")
    ap.add_argument("--results", default="/mnt/user-data/uploads/20260316_1143_results.csv")
    ap.add_argument("--oracle",  default="/mnt/user-data/uploads/sampled_oracle.csv")
    ap.add_argument("--log",     default="/mnt/user-data/uploads/log.txt")
    ap.add_argument("--out",     default="/mnt/user-data/outputs/evaluation_report.txt")
    ap.add_argument("--cat-max-unique", type=int, default=CATEGORICAL_MAX_UNIQUE,
                    help=f"Max distinct values to treat a column as categorical (default {CATEGORICAL_MAX_UNIQUE})")
    ap.add_argument("--free-text-min-unique", type=int, default=FREE_TEXT_MIN_UNIQUE,
                    help=f"Min distinct values needed to treat a column as free-text (default {FREE_TEXT_MIN_UNIQUE})")
    args = ap.parse_args()

    CATEGORICAL_MAX_UNIQUE = args.cat_max_unique
    FREE_TEXT_MIN_UNIQUE   = args.free_text_min_unique

    print(f"\n{SEP}")
    print("  LLM TRANSPORTATION AGENT — DYNAMIC EVALUATION SUITE")
    print(SEP)

    # 1. Load & map
    print("[1/11] Loading and mapping columns ...")
    res, ora, col_map, meta_cols, n = load_and_map(args.results, args.oracle)
    log_info = load_log(args.log)
    print(f"       {n} rows  |  {len(col_map)} shared cols  |  "
          f"excluded metadata: {meta_cols}")

    # 2. Classify
    print("[2/11] Auto-classifying columns ...")
    numeric_cols, categorical_cols, free_text_cols = classify_columns(res, ora)
    print(f"       Categorical : {categorical_cols}")
    print(f"       Numeric     : {numeric_cols}")
    print(f"       Free-text   : {free_text_cols}")

    # 3. Entropy
    print("[3/11] Shannon Entropy ...")
    entropy_df = compute_entropy_table(res, ora, categorical_cols)

    # 4. KL Divergence
    print("[4/11] KL Divergence ...")
    kl_df = compute_kl_table(res, ora, categorical_cols)

    # 5. Numeric
    print("[5/11] Numeric comparison ...")
    num_df = compute_numeric_table(res, ora, numeric_cols)

    # 6-9. NLP
    print("[6/11] BLEU ...")
    bleu = compute_bleu(res, ora, free_text_cols)
    print("[7/11] ROUGE ...")
    rouge = compute_rouge(res, ora, free_text_cols)
    print("[8/11] METEOR ...")
    meteor = compute_meteor(res, ora, free_text_cols)
    print("[9/11] BERTScore ...")
    bertscore = compute_bertscore(res, ora, free_text_cols)
    print(f"       {bertscore['note']}")

    # 10. Task / quality metrics
    print("[10/11] SR / TGC / Factual / Relevance ...")
    sr        = compute_sr(res)
    tgc       = compute_tgc(res, ora, categorical_cols, numeric_cols)
    factual   = compute_factual(res, ora, categorical_cols)
    relevance = compute_relevance(res, ora, free_text_cols)

    # 11. Report
    print("[11/11] Assembling report ...")
    summary_df   = build_summary(
        log_info, n, col_map,
        categorical_cols, numeric_cols, free_text_cols,
        entropy_df, kl_df, bleu, rouge, meteor, bertscore,
        sr, tgc, factual, relevance,
    )
    nlp_df       = build_nlp_table(bleu, rouge, meteor, bertscore)
    factual_df   = build_factual_table(factual)
    relevance_df = build_relevance_table(relevance)


    col_class_txt = (
        f"Categorical  ({len(categorical_cols)}): {categorical_cols}\n"
        f"Numeric      ({len(numeric_cols)}):     {numeric_cols}\n"
        f"Free-text    ({len(free_text_cols)}):   {free_text_cols}\n"
    )

    header_txt = (
        f"Results : {Path(args.results).name}\n"
        f"Oracle  : {Path(args.oracle).name}\n"
        f"Log     : {Path(args.log).name}\n"
        f"{SEP2}"
    )

    write_report(args.out, [
        ("HEADER",                                              header_txt),
        ("TABLE 0 — GLOBAL SUMMARY",                           tbl(summary_df)),
        ("COLUMN CLASSIFICATION  (auto-inferred from data)",   col_class_txt),
        ("TABLE 1 — NORMALIZED SHANNON ENTROPY",               tbl(entropy_df)),
        ("TABLE 2 — KL DIVERGENCE  P(Oracle) || Q(Agent)",     tbl(kl_df)),
        ("TABLE 3 — NUMERIC VARIABLE COMPARISON",              tbl(num_df)),
        ("TABLE 4 — NLP METRICS PER FREE-TEXT FIELD\n"
         "           BLEU / ROUGE / METEOR / BERTScore",        tbl(nlp_df)),
        ("TABLE 5 — SUCCESS RATE  (SR)",                       _kv(sr)),
        ("TABLE 6 — TASK GOAL COMPLETION  (TGC)",              _kv(tgc)),
        ("TABLE 7 — FACTUAL CORRECTNESS  (exact norm. match)", tbl(factual_df)),
        ("TABLE 8 — RESPONSE RELEVANCE  (TF-IDF cosine)",      tbl(relevance_df)),
    ])

    # Console summary
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