# === NEW / UPDATED SECTIONS BELOW ===
# Add these imports
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Dict, Any, Optional, Tuple
from utils import *
import traceback
# ---------- Graph State ----------
class ResumeState(TypedDict, total=False):
    jd: Dict[str, Any]
    resume_path: str
    resume_text: str
    llm: Any
    raw_llm_output: str
    parsed_json: Dict[str, Any]
    report: Dict[str, Any]
    errors: List[str]

# ---------- Graph Nodes ----------
def node_load_resume(state: ResumeState) -> ResumeState:
    try:
        text = load_resume_markdown(state["resume_path"])
        if not text.strip():
            raise ValueError("Empty text extracted from resume")
        state["resume_text"] = text
    except Exception as e:
        errs = state.get("errors", [])
        errs.append(f"load_resume: {e}")
        state["errors"] = errs
        # You could choose to raise to stop the graph early
        raise
    return state

# def node_call_llm(state: ResumeState) -> ResumeState:
#     try:
#         r = call_llm_structured(state["llm"], state["jd"], state["resume_text"])
#         print(r.content)
#         state["raw_llm_output"] = r.content
#         cleaned = clean_text_v2(r.content)
#         state["parsed_json"] = json.loads(cleaned)
#     except Exception as e:
#         errs = state.get("errors", [])
#         errs.append(f"call_llm: {e}")
#         state["errors"] = errs
#         raise
#     return state
from requests.exceptions import Timeout, RequestException
import random
def node_call_llm(state: ResumeState) -> ResumeState:
    max_retries = 3
    backoff_factor = 1  # seconds
    timeout = 10  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            # Attempt to call the LLM
            r = call_llm_structured(state["llm"], state["jd"], state["resume_text"])
            print(r.content)
            state["raw_llm_output"] = r.content

            # Process the response
            cleaned = clean_text_v2(r.content)
            state["parsed_json"] = json.loads(cleaned)
            return state

        except Timeout as e:
            # Handle timeout errors
            error_message = f"Attempt {attempt} failed due to timeout: {e}"
            print(error_message)
            if attempt == max_retries:
                state["errors"] = state.get("errors", []) + [error_message]
                raise

        except RequestException as e:
            # Handle other request-related errors
            error_message = f"Attempt {attempt} failed due to request error: {e}"
            print(error_message)
            if attempt == max_retries:
                state["errors"] = state.get("errors", []) + [error_message]
                raise

        except Exception as e:
            # Handle unexpected errors
            error_message = f"Attempt {attempt} failed due to unexpected error: {e}"
            print(error_message)
            if attempt == max_retries:
                state["errors"] = state.get("errors", []) + [error_message]
                raise

        # Calculate exponential backoff with jitter
        sleep_time = backoff_factor * (2 ** (attempt - 1)) + random.uniform(0, 1)
        print(f"Retrying in {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)

    # If all attempts fail, raise an exception
    raise Exception("All retry attempts failed.")

def node_build_report(state: ResumeState) -> ResumeState:
    try:
        resume_path_str = state["resume_path"]
        jd = state["jd"]
        parsed = state["parsed_json"]
        if isinstance(parsed, list):
            if len(parsed) == 1 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            else:
                state.setdefault("errors", []).append(
                    f"parsed_json is a list with length {len(parsed)}; expected single dict"
                )
                parsed = {}  # fallback to empty dict

        candidate = safe_candidate_name_from_file(Path(resume_path_str).name)
        report = {
            "candidate_name": candidate,
            "job_title": jd.get("Job_Title") or jd.get("job_title"),
            **parsed,
        }
        state["report"] = report
    except Exception as e:
        errs = state.get("errors", [])
        errs.append(f"build_report: {e}")
        state["errors"] = errs
        raise
    return state

def node_save_report(state: ResumeState) -> ResumeState:
    try:
        reports_dir = Path("reports"); reports_dir.mkdir(exist_ok=True)
        rep = state["report"]
        out_path = reports_dir / f"{rep['candidate_name']}_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
    except Exception as e:
        errs = state.get("errors", [])
        errs.append(f"save_report: {e}")
        state["errors"] = errs
        raise
    return state

# ---------- Graph Builder ----------
def build_resume_graph(structured_llm) -> Any:
    """
    Build a LangGraph pipeline for SINGLE resume processing:
      load_resume -> call_llm -> build_report -> save_report -> END
    The compiled graph reuses your existing helpers & prompt.
    """
    graph = StateGraph(ResumeState)
    graph.add_node("load_resume", node_load_resume)
    graph.add_node("call_llm", node_call_llm)
    graph.add_node("build_report", node_build_report)
    graph.add_node("save_report", node_save_report)

    graph.set_entry_point("load_resume")
    graph.add_edge("load_resume", "call_llm")
    graph.add_edge("call_llm", "build_report")
    graph.add_edge("build_report", "save_report")
    graph.add_edge("save_report", END)

    # In-memory checkpointing is handy for debugging / retries
    
    return graph.compile()

# ---------- Single-run Helper ----------
def run_single_resume_with_graph(jd: Dict[str, Any], resume_path: Path, structured_llm) -> Optional[Dict[str, Any]]:
    """
    Invoke the compiled LangGraph workflow for a single resume.
    Returns the final report dict or None on failure.
    """
    g = build_resume_graph(structured_llm)
    init: ResumeState = {
        "jd": jd,
        "resume_path": str(resume_path),
        "llm": structured_llm,
        "errors": [],
    }
    try:
        final_state = g.invoke(init)
        return final_state.get("report")
    except Exception as e:
        logging.error(f"Graph failed for {Path(resume_path).name}: {e}")
        logging.error(traceback.format_exc())  # <-- logs full traceback
        return None