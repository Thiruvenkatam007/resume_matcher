import os
import re
import json
import yaml
import csv
import math
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import json

# LangChain (modern imports)
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import backoff
import numpy as np
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ========== IO helpers ==========

def load_yaml_job_description(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        jd = yaml.safe_load(f) or {}
    return jd


import logging

def _read_pdf_pdfminer_md(file_path: str) -> str:
    """
    Extract PDF text as Markdown using pdfminer.six.
    """
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(file_path) or ""
        if text.strip():
            # Convert to markdown-friendly (basic)
            return "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        return ""
    except Exception as e:  # pragma: no cover
        logging.warning(f"pdfminer failed on {file_path}: {e}")
        return ""


def _read_pdf_pypdf_md(file_path: str) -> str:
    """
    Extract PDF text as Markdown fallback using pypdf.
    """
    try:
        from pypdf import PdfReader
    except Exception:  # pragma: no cover
        return ""

    try:
        text = []
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        # Markdown-friendly formatting
                        text.append("\n".join([ln.strip() for ln in page_text.splitlines() if ln.strip()]))
                except Exception:
                    text.append("")
        return "\n".join(text)
    except Exception as e:
        logging.warning(f"pypdf failed on {file_path}: {e}")
        return ""


def load_resume_markdown(file_path: str) -> str:
    """
    Load resume and return Markdown text.
    Priority: pdfminer -> pypdf -> warn if fails.
    """
    file_path = str(file_path)
    if file_path.lower().endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.warning(f"Could not read TXT {file_path}: {e}")
            return ""
    elif file_path.lower().endswith('.pdf'):
        text = _read_pdf_pdfminer_md(file_path)
        if not text.strip():
            text = _read_pdf_pypdf_md(file_path)
        if not text.strip():
            logging.warning(f"Could not extract text from PDF {file_path}")
        return text
    else:
        logging.warning(f"Unsupported resume format: {file_path}")
        return ""



def safe_candidate_name_from_file(file_name: str) -> str:
    stem = Path(file_name).stem
    return re.sub(r"[^A-Za-z0-9_.-]", "_", stem)


# ========== Pydantic schema for structured output ==========

class MatchReport(BaseModel):
    matched_required_skills: List[str] = []
    missing_required_skills: List[str] = []
    matched_optional_skills: List[str] = []
    education_match: str
    experience_match: str
    keywords_matched: List[str] = []
    soft_skills_match: List[str] = []
    resume_summary: str
    match_score: float = Field(ge=0, le=1)
    city_tier_match: bool
    longest_tenure_months: int
    final_score: int = Field(ge=0, le=100)
    # --- extended, optional fields for evaluation criteria ---
    detected_city: Optional[str] = None
    detected_city_tier: Optional[int] = None  # 1/2/3 if the model can infer
    max_job_gap_months: Optional[int] = None
    stability_score: Optional[float] = Field(default=None, ge=0, le=1)

# ========== Prompt ==========

PROMPT = ChatPromptTemplate.from_messages([
    (
    "system",
    "You are an expert technical recruiter and data scientist. "
    "Your job is to read a job description (JD) and a resume , then return a STRICT JSON object matching the schema. "
    "Be precise, consistent, and terse. If the information is not present, return a sensible null/empty value rather than guessing. "
    "NEVER add commentary, markdown, or keys not in the schema."
),
(
    "user",
    "<OBJECTIVE>\n"
    "Evaluate the resume against the JD and produce high-quality, schema-valid JSON capturing skills, education, experience fit, city-tier & gaps, longest tenure, and a calibrated final_score.\n"
    "\n"
    "<INPUTS>\n"
    "Job Description (YAML): {job_description}\n"
    "Resume : {resume_text}\n"
    "\n"
     "<RUBRIC FOR final_score (100-point scale)>\n"
    "Weightage:\n"
    "- required skills coverage: 40%\n"
    "- optional skills coverage: 15%\n"
    "- experience fit (years/recency/scope): 15%\n"
    "- education fit: 10%\n"
    "- location fit: 5% (true if city_tier meets JD or is unspecified)\n"
    "- stability: 10% (longest_tenure_months; full credit at 48 months; scale proportionally)\n"
    "- diversity by city tier: 5% bonus (Tier-3 > Tier-2 > Tier-1; score 100 for T3, 60 for T2, 0 for T1)\n"

    "\n"
    "<SCHEMA AND CONSTRAINTS>\n"
    "You must return a single JSON object with the following keys and constraints: Strictly Don't add any extra key value pair:\n"
    "- matched_required_skills: string[] (subset of JD.required_skills that appear in the resume; normalize case and aliases like js->javascript, py->python, torch->pytorch)\n"
    "- missing_required_skills: string[] (skills from JD.required_skills not evidenced in the resume chunk)\n"
    "- matched_optional_skills: string[] (subset of JD.optional_skills found)\n"
    "- education_match: string (short justification or 'false' if not met; keep to <= 1 sentence)\n"
    "- experience_match: string (short justification or 'false' if not met; <= 1 sentence)\n"
    "- keywords_matched: string[] (notable JD keywords present in resume text)\n"
    "- soft_skills_match: string[] (soft skills evidenced in text; e.g., communication, leadership)\n"
    "- resume_summary: string (1–2 sentences summarizing the candidate relevant to the JD)\n"
    "- match_score: number in [0,1] (your calibrated similarity for THIS resume)\n"
    "- city_tier_match: boolean (true if city_tier meets or exceeds JD requirement if any; else false)\n"
    "- longest_tenure_months: integer >= 0 (The longest duration (in months) the candidate was employed in a **single company** based on their work history)\n"
    "- final_score: integer in [0,100] (An integer between 0 and 100 summarizing the overall resume match quality against the job description based on all criteria above; be conservative if context is incomplete)\n"
    "- detected_city: string|null\n"
    "- detected_city_tier: 1|2|3|null\n"
    "- city_tier_match: boolean (true if detected_city_tier meets the Job description requirement any; else false)\n"
    "- max_job_gap_months: integer|null (largest gap in months between consecutive jobs for this resume; compute as difference between next job start and previous job end)"
    "- max_job_gap_months_check: string|null (short justification if the candidate’s maximum job gap in months is <= Maximum_Job_Gap_Months specified in the Job Description; else null)"
    "\n"
    "<EVIDENCE RULES>\n"
    "- Consider common aliases: js↔javascript, ts↔typescript, py↔python, torch↔pytorch, tf↔tensorflow, np↔numpy, sk↔scikit-learn. Normalize to canonical names.\n"
    "- Date parsing: recognize ranges like 'Jan 2019 - Mar 2022', '2018–2021', '2020 to Present'. Compute tenure in months (approx). Present/current = current month. If ambiguous, be conservative.\n"
    "\n"
   
    "<ROBUSTNESS & STYLE>\n"
    "- Keep outputs concise; arrays deduplicated and normalized to lowercase where appropriate.\n"
    "- Never include markdown or commentary—only the JSON object.\n"
    "\n"
    "<OUTPUT> Return a **single valid JSON object** using this structure(without extra comments or explanations)"
)
])

# ========== Embeddings (pre-filter) ==========

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))

_embedder: Optional[SentenceTransformerEmbeddings] = None

def get_embedder() -> SentenceTransformerEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embedder

def embed_text(text: str) -> np.ndarray:
    emb = get_embedder().embed_query(text or "")
    return np.array(emb, dtype=np.float32)


def prefilter_resumes(jd: Dict[str, Any], resume_paths: List[Path], texts: List[str], topk: Optional[int] = None, topk_frac: float = 0.4) -> List[Tuple[Path, float]]:
    """Rank resumes by embedding similarity to the JD and return the top subset.
    If topk is None, select ceil(len(resumes) * topk_frac). Never fewer than 1.
    """
    jd_text = yaml.dump(jd, sort_keys=False)
    jd_vec = embed_text(jd_text)

    sims: List[Tuple[int, float]] = []
    for i, t in enumerate(texts):
        try:
            v = embed_text(t)
            sims.append((i, _cosine(jd_vec, v)))
        except Exception:
            sims.append((i, -1.0))
    sims.sort(key=lambda x: x[1], reverse=True)

    n = len(resume_paths)
    k = int(topk) if topk is not None else int(np.ceil(max(1, n) * float(topk_frac)))
    k = max(1, min(n, k))

    selected = [(resume_paths[i], score) for i, score in sims[:k]]
    return selected

# ========== LLM client ==========
import streamlit as st
api_key = st.secrets["api_keys"]["OPENROUTER_API_KEY"]
os.environ["OPENROUTER_API_KEY"] = api_key
base = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
# model = os.getenv("MODEL_NAME", "nvidia/nemotron-nano-9b-v2:free")
# model_name = "qwen/qwen2.5-vl-72b-instruct:free"

def make_llm(model: str = "gpt-4o-mini", temperature: float = 0.6):
    # print(model_name)
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key, base_url=base)
    # return llm.with_structured_output(MatchReport)
    return llm

# ========== Retry wrapper ==========

@backoff.on_exception(backoff.expo, Exception, max_time=90)
def call_llm_structured(structured_llm, jd_dict: Dict[str, Any], resume_text: str) -> MatchReport:
    msg = PROMPT.format(job_description=yaml.dump(jd_dict, sort_keys=False),
                        resume_text=resume_text)
    
    # IMPORTANT: for structured outputs, invoke with messages
    return structured_llm.invoke(msg)
# ========== Per-resume processing ==========


def clean_text_v2(text: str) -> str:
    """
    Extract the first valid JSON object/array from arbitrary text.

    - Handles Markdown fences like ```json ... ``` or plain ``` ... ```.
    - Ignores any prose before/after the JSON.
    - Returns the JSON substring (not a Python dict). If nothing parses,
      returns a best-effort substring starting at the first '{' or '['.
    """
    if not text:
        return ""

    s = text.strip().replace("\u00A0", " ")  # normalize non-breaking spaces

    # 1) Collect candidates from fenced code blocks (prefer these first).
    fence_re = re.compile(r"```(?:\w+)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    candidates = [m.group(1).strip() for m in fence_re.finditer(s)]

    # 2) Also consider the full text in case JSON isn't fenced.
    candidates.append(s)

    def balanced_json_substrings(src: str):
        """Yield substrings that are balanced JSON blocks starting at '{' or '['."""
        out = []
        i, n = 0, len(src)
        while i < n:
            ch = src[i]
            if ch in "{[":
                start = i
                stack = [ch]
                i += 1
                in_str = False
                esc = False
                while i < n:
                    c = src[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif c == "\\":
                            esc = True
                        elif c == '"':
                            in_str = False
                        i += 1
                        continue
                    else:
                        if c == '"':
                            in_str = True
                        elif c in "{[":
                            stack.append(c)
                        elif c in "}]":
                            if not stack:
                                break
                            opening = stack.pop()
                            if (opening == "{" and c != "}") or (opening == "[" and c != "]"):
                                break
                            if not stack:
                                # Found a balanced block
                                out.append(src[start:i+1].strip())
                                break
                        i += 1
                # Move forward to search for the next block
                i = start + 1
            else:
                i += 1
        return out

    # 3) Try to find a substring that actually parses as JSON.
    for cand in candidates:
        for sub in balanced_json_substrings(cand):
            try:
                json.loads(sub)
                return sub
            except Exception:
                continue

    # 4) Fallback: strip fences and return from the first '{' or '[' onward.
    def _strip_fences(m):  # keep inner content
        return (m.group(1) or "").strip()

    unfenced = fence_re.sub(_strip_fences, s).strip()
    m = re.search(r"[\{\[]", unfenced)
    return unfenced[m.start():].strip() if m else ""




