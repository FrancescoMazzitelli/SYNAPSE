import pandas as pd
from jinja2 import Environment, FileSystemLoader
import json
import re
from typing import Dict, Tuple, Set, Any, List
import json
from pathlib import Path


class SystemMessageGenerator:
    def __init__(self, config_folder: str, template_name: str = "SystemMessage.j2"):
        self.config_folder = Path(config_folder)
        template_path = self.config_folder / "templates"
        self.env = Environment(loader=FileSystemLoader(str(template_path)))
        self.template = self.env.get_template(template_name)

    def write_system_message(self, **kwargs) -> str:
        return self.template.render(**kwargs)


def generate_questions(config_folder: str) -> Dict[str, Dict]:
    config_path = Path(config_folder)
    with open(config_path / "config.json", "r") as f:
        config = json.load(f)
    return config.get("questions", {})


def _normalize_str(x: Any) -> str:
    return str(x).strip().lower()


def safe_csv_read(path):
    encodings = ["utf-8-sig", "utf-8", "latin1"]

    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                quotechar='"',
                encoding=enc,
            )
        except UnicodeDecodeError:
            continue

    raise ValueError("File impossible to decode")

def split_answers_keep_parentheses(s: str) -> list:
    """
    Split a string by commas, semicolons, or newlines, 
    but keep commas inside (), {}, [] intact.
    """
    result = []
    current = []
    stack = []
    
    pairs = {"(":")", "{":"}", "[":"]"}
    
    for char in s:
        if char in pairs.keys():
            stack.append(pairs[char])
            current.append(char)
        elif char in pairs.values():
            if stack and char == stack[-1]:
                stack.pop()
            current.append(char)
        elif char in [",", ";", "\n"] and not stack:
            piece = ''.join(current).strip().strip('"')
            if piece:
                result.append(piece)
            current = []
        else:
            current.append(char)
    
    # Add last piece
    piece = ''.join(current).strip().strip('"')
    if piece:
        result.append(piece)
    
    return result

def extract_answer_set(data_path: str,
                        name_column: str = "Name",
                        responses_column: str = "Responses") -> dict:

    if data_path.endswith(".csv"):
        df = safe_csv_read(data_path)
    else:
        df = pd.read_excel(data_path)

    return {
        _normalize_str(row[name_column]): (
            split_answers_keep_parentheses(str(row[responses_column]))
            if pd.notna(row[responses_column]) else []
        )
        for _, row in df.iterrows()
    }

def infer_dtype(header: str, answer_set: set, question: str, description: str) -> str:
    """Infer the expected response dtype for a survey variable.

    Returns one of: 'SINGLE', 'MULTIPLE', 'NUMERIC', 'TEXT'.
    """
    if answer_set:
        q_lower = (question or "").lower()
        multi_hints = ['select all', 'choose all', 'select all that apply', 'please select all']
        if any(hint in q_lower for hint in multi_hints):
            return "MULTIPLE"
        if len(answer_set) == 1:
            return "SINGLE"
        return "SINGLE"

    q_lower = (question or "").lower()
    d_lower = (description or "").lower()
    h_lower = header.lower()
    numeric_hints = [
        'how many', 'how much', 'age', 'year', 'salary', 'income',
        'miles', 'minutes', 'hours', 'cost', 'price', 'distance',
        'time', 'weight', 'height', 'number of', 'count', 'temperature',
        'how old', 'what is your age'
    ]
    for hint in numeric_hints:
        if hint in q_lower or hint in d_lower or hint in h_lower:
            return "NUMERIC"
    return "TEXT"


def map_generic_data_with_answers(
    data_path: str, 
    desc_path: str
) -> Dict[str, Tuple[str, str, Set[Any]]]:
    """
    Maps headers to (description, question, answer_set).
    Dtype is added later by LLM inference.

    Args:
        data_path: Path to the CSV or Excel data file
        desc_path: Path to the CSV, Excel, or JSON description file
        
    Returns:
        Dict mapping header -> (description, question, answer_set)
    """
    # Load data headers
    if data_path.endswith('.csv'):
        df = safe_csv_read(data_path)
        headers = df.columns.tolist()
    else:
        headers = pd.read_excel(data_path, nrows=0).columns.tolist()

    # Load descriptions
    if desc_path.endswith('.csv'):
        desc_df = safe_csv_read(desc_path)
    else:
        desc_df = pd.read_excel(desc_path)

    # Normalize column names
    desc_df.columns = [c.strip().upper() for c in desc_df.columns]

    flag = False
    
    # Find relevant columns with fallbacks
    var_col = next((c for c in desc_df.columns if 'VARIABLE' in c or 'NAME' in c or 'HEADER' in c), desc_df.columns[0])
    desc_col = next((c for c in desc_df.columns if 'DESC' in c or 'DESCRIPTION' in c), None)
    ques_col = next((c for c in desc_df.columns if 'QUEST' in c or 'QUESTION' in c), None)

    if desc_col is None:
        desc_col = desc_df.columns[1] if len(desc_df.columns) > 1 else var_col
    if ques_col is None:
        ques_col = desc_df.columns[2] if len(desc_df.columns) > 2 else desc_col

    print("Columns detected:", list(desc_df.columns))
    if "RESPONSES" in desc_df.columns and "NAME" in desc_df.columns:
        answer_dict = extract_answer_set(desc_path)
        flag = True
        
    mapping = {}
    for h in headers:
        match = desc_df[desc_df[var_col].astype(str).str.upper() == h.upper()]
        if match.empty:
            continue
        
        desc = match[desc_col].values[0] if desc_col in match.columns else h
        ques = match[ques_col].values[0] if ques_col in match.columns else h
        
        # Extract answer set from actual data - NO TOOL INFERENCE
        key = _normalize_str(h)
        answer_list = answer_dict.get(key, [])
        if answer_list:
            answer_list = filter_skip_codes(answer_list)
            answer_set = set(answer_list)
        else:
            answer_set = set()

        mapping[h] = (str(desc), str(ques), answer_set)
        print(f"{h}: {answer_set}")

    return mapping

def extract_personal_agent_info(desc_df: pd.DataFrame) -> List[str]:
    """
    Extract socio-demographic variable names from description dataframe.
    """
    col_map = {c.lower(): c for c in desc_df.columns}

    # Find theme column
    theme_col = next(
        (orig for low, orig in col_map.items() if "theme" in low),
        None
    )

    # Find variable/name column
    var_col = next(
        (orig for low, orig in col_map.items()
         if "name" in low or "variable" in low or "nom" in low),
        None
    )

    if theme_col is None or var_col is None:
        return []

    labels: List[str] = []

    for _, row in desc_df.iterrows():
        theme_val = str(row[theme_col]).lower()
        if "sociodem" in theme_val:
            labels.append(str(row[var_col]))

    return {"labels": labels}
    

def get_questions_generic_with_answers(
    key_mapping: Dict[str, Tuple]
) -> Dict[str, Dict]:
    """
    Convert mapping to questions dictionary with answer sets and dtype.
    
    Args:
        key_mapping: Dict mapping header -> (description, question, answer_set[, dtype])
        
    Returns:
        Dict in format {header: {"text": question, "description": desc, 
                                 "possible_answers": answer_list, "dtype": dtype}}
    """
    questions = {}
    for key, value in key_mapping.items():
        if isinstance(value, tuple) and len(value) >= 3:
            desc, question, answer_set = value[:3]
            dtype = value[3] if len(value) >= 4 else infer_dtype(key, answer_set, str(question), str(desc))
            
            # Convert set to sorted list for JSON serialization
            if answer_set:
                answer_list = sorted(list(answer_set), key=lambda x: (isinstance(x, str), x))
                answer_list = filter_skip_codes(answer_list)
            else:
                answer_list = []
            
            questions[key] = {
                "text": question,
                "description": desc,
                "possible_answers": answer_list,
                "answer_count": len(answer_list),
                "dtype": dtype
            }
        else:
            dtype = "TEXT"
            questions[key] = {
                "text": f"What is your {key}?",
                "description": str(value),
                "possible_answers": [],
                "answer_count": 0,
                "dtype": dtype
            }
    
    return questions


SKIP_CODE_PHRASES = [
    "appropriate skip", "prefer not to answer",
    "i prefer not to answer", "i don't know",
    "don't know", "not ascertained", "refused",
    "does not know", "prefer not to say", "dont know"
]

def _is_skip_code(value: str) -> bool:
    stripped = value.strip()
    lowered = stripped.lower()
    for phrase in SKIP_CODE_PHRASES:
        if phrase in lowered:
            return True
    return False


def filter_skip_codes(answer_list: List[Any]) -> List[Any]:
    return [a for a in answer_list if not (isinstance(a, str) and _is_skip_code(a))]


def format_answer_set_for_prompt(answer_list: List[Any], max_display: int = 60) -> str:
    """
    Format the answer set into a readable string for the agent prompt.
    
    Args:
        answer_list: List of possible answers
        max_display: Maximum number of answers to display before truncating
        
    Returns:
        Formatted string describing the possible answers
    """
    if not answer_list:
        return "Open-ended response (no predefined answers)"
    
    if len(answer_list) <= max_display:
        # All numeric
        if all(isinstance(a, (int, float)) for a in answer_list):
            return f"{answer_list}"
        # Mix or all strings
        else:
            formatted = ", ".join([f'"{a}"' if isinstance(a, str) else str(a) for a in answer_list])
            return f"{formatted}"
    else:
        # Truncate with indication
        sample = answer_list[:max_display]
        if all(isinstance(a, (int, float)) for a in answer_list):
            return f"Range approximately: {min(answer_list)} to {max(answer_list)} ({len(answer_list)} total options)"
        else:
            formatted = ", ".join([f'"{a}"' if isinstance(a, str) else str(a) for a in sample])
            return f"{formatted}... ({len(answer_list)} total options)"


# Backward compatibility - keep old function names as wrappers
def map_generic_data(data_path: str, desc_path: str) -> Dict[str, Tuple[str, str, Set[Any]]]:
    """Wrapper for backward compatibility."""
    return map_generic_data_with_answers(data_path, desc_path)


def get_questions_generic(key_desc_quest_mapping: Dict) -> Dict[str, Dict]:
    """Wrapper for backward compatibility."""
    return get_questions_generic_with_answers(key_desc_quest_mapping)