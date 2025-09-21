# Import your process_all
from main_processor import process_all  # <- put your function in a file called process_module.py
import streamlit as st
import os
import time
import logging
import json
import zipfile
import io
from pathlib import Path
from typing import List, Dict, Any
from langraph_agent import *

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent  # automatically detect base dir
DEFAULT_JD = BASE_DIR / "jd.yaml"
DEFAULT_RESUMES_DIR = BASE_DIR / "resumes"
REPORTS_DIR = BASE_DIR / "reports"
UPLOADS_DIR = BASE_DIR / "uploads"

import shutil
from pathlib import Path
import logging

# ---------- HELPERS ----------
def clear_folder(folder: Path):
    """Delete all files and subfolders inside a folder."""
    if folder.exists():
        for f in folder.iterdir():
            try:
                if f.is_file() or f.is_symlink():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
            except Exception as e:
                logging.error(f"Could not delete {f}: {e}")
    else:
        folder.mkdir(parents=True, exist_ok=True)



def load_reports(reports_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON reports from reports directory."""
    reports = []
    for rf in reports_dir.glob("*.json"):
        try:
            with open(rf, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_report_file"] = rf.name  # track filename
                reports.append(data)
        except Exception as e:
            logging.error(f"Failed to load {rf}: {e}")
    return reports


# ---------- STREAMLIT APP ----------
def main():
    st.set_page_config(page_title="Resume Matcher", layout="wide")
    st.title("📄 AI Resume Matcher")

    st.markdown("Default JD and 5 resumes are already configured. "
                "You may upload new files, but they will be cleared after processing.")

    # --- Upload Job Description ---
    jd_file = st.file_uploader("Upload Job Description (YAML)", type=["yaml", "yml"])

    # --- Upload resumes ---
    resumes = st.file_uploader("Upload Resumes (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)

    # --- Parameters ---
    workers = st.slider("Number of parallel workers(Use in caution due to ratelimit for Open router API)", 1, 8, 2)
    topk = st.number_input("Top-K resumes to evaluate (leave 0 to use all)", min_value=0, value=5)
    topk_frac = st.slider("Fraction of resumes to consider (if Top-K not set)", 0.1, 1.0, 0.4)
    model = st.selectbox("Choose LLM model", ["qwen/qwen2.5-vl-72b-instruct:free", "nvidia/nemotron-nano-9b-v2:free"])

    if st.button("🚀 Run Resume Matching", type="primary"):
        # Clear reports before every run
        clear_folder(REPORTS_DIR)

        # Decide which JD to use
        jd_path = UPLOADS_DIR / "jd.yaml"
        resumes_folder = UPLOADS_DIR / "resumes"
        resumes_folder.mkdir(parents=True, exist_ok=True)

        # Clear any previous uploads
        clear_folder(UPLOADS_DIR)

        if jd_file:
            with open(jd_path, "wb") as f:
                f.write(jd_file.getbuffer())
        else:
            jd_path = DEFAULT_JD  # fallback to default

        if resumes:
            for r in resumes:
                save_path = resumes_folder / r.name
                with open(save_path, "wb") as f:
                    f.write(r.getbuffer())
        else:
            resumes_folder = DEFAULT_RESUMES_DIR  # fallback to default

        st.info("Processing resumes... please wait ⏳")

        # --- Run the pipeline ---
        t0 = time.time()
        process_all(
            job_description_file=str(jd_path),
            resumes_folder=str(resumes_folder),
            workers=workers,
            model=model,
            topk=None if topk == 0 else topk,
            topk_frac=topk_frac,
        )
        duration = time.time() - t0

        # --- Load reports ---
        reports = load_reports(REPORTS_DIR)

        if not reports:
            st.error("⚠️ No reports found. Please check logs.")
            return

        # Sort by final_score if available
        if "final_score" in reports[0]:
            reports.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        st.success(f"✅ Done! Processed resumes in {duration:.1f}s")
        st.markdown("### 📊 Candidate Rankings")

        # Display as table
        table_data = [
            {
                "Rank": i + 1,
                "Candidate": r.get("candidate_name", r["_report_file"]),
                "Final Score": r.get("final_score", "N/A"),
                "Report File": r["_report_file"],
            }
            for i, r in enumerate(reports)
        ]
        st.dataframe(table_data, use_container_width=True)

        # Allow downloading each report
        st.markdown("### 📥 Download Reports")
        for r in reports:
            report_str = json.dumps(r, indent=2)
            st.download_button(
                label=f"Download {r.get('candidate_name', r['_report_file'])}.json",
                data=report_str,
                file_name=r["_report_file"],
                mime="application/json",
            )

        # Allow downloading all as ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for r in REPORTS_DIR.glob("*.json"):
                zipf.write(r, arcname=r.name)
        zip_buffer.seek(0)

        st.download_button(
            label="⬇️ Download All Reports (ZIP)",
            data=zip_buffer,
            file_name="all_candidate_reports.zip",
            mime="application/zip",
        )

        # --- Cleanup uploads after processing ---
        clear_folder(UPLOADS_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

# def clear_reports(reports_dir: Path):
#     """Remove all old reports before a new run."""
#     if reports_dir.exists():
#         for f in reports_dir.glob("*"):
#             try:
#                 f.unlink()
#             except Exception as e:
#                 logging.error(f"Could not delete {f}: {e}")
#     else:
#         reports_dir.mkdir(parents=True, exist_ok=True)

# import streamlit as st
# import os
# import time
# import logging
# import json
# import zipfile
# import io
# from pathlib import Path
# from typing import Optional, List, Dict, Any
# from langraph_agent import *
# # from process_module import process_all  # <-- keep your function here


# def load_reports(reports_dir: Path) -> List[Dict[str, Any]]:
#     """Load all JSON reports from reports directory."""
#     reports = []
#     for rf in reports_dir.glob("*.json"):
#         try:
#             with open(rf, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#                 data["_report_file"] = rf.name  # track filename
#                 reports.append(data)
#         except Exception as e:
#             logging.error(f"Failed to load {rf}: {e}")
#     return reports


# def main():
#     st.set_page_config(page_title="Resume Matcher", layout="wide")
#     st.title("📄 AI Resume Matcher")

#     st.markdown("Upload a Job Description YAML and Resumes (PDF/TXT) to run the LangGraph pipeline.")

#     # --- Upload Job Description ---
#     jd_file = st.file_uploader("Upload Job Description (YAML)", type=["yaml", "yml"])

#     # --- Upload resumes ---
#     resumes = st.file_uploader("Upload Resumes (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)

#     # --- Parameters ---
#     workers = st.slider("Number of parallel workers", 1, 8, 2)
#     topk = st.number_input("Top-K resumes to evaluate (leave 0 to use all)", min_value=0, value=5)
#     topk_frac = st.slider("Fraction of resumes to consider (if Top-K not set)", 0.1, 1.0, 0.4)
#     model = st.selectbox("Choose LLM model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-3.5-turbo"])

#     if st.button("🚀 Run Resume Matching", type="primary"):
#         if not jd_file or not resumes:
#             st.error("Please upload both Job Description and Resumes first.")
#             return

#         # Create temporary folders
#         jd_path = Path("uploaded_jd.yaml")
#         resumes_folder = Path("uploaded_resumes")
#         resumes_folder.mkdir(exist_ok=True)

#         # Save JD
#         with open(jd_path, "wb") as f:
#             f.write(jd_file.getbuffer())

#         # Save resumes
#         for r in resumes:
#             save_path = resumes_folder / r.name
#             with open(save_path, "wb") as f:
#                 f.write(r.getbuffer())

#         st.info("Processing resumes... please wait ⏳")

#         t0 = time.time()
#         process_all(
#             job_description_file=str(jd_path),
#             resumes_folder=str(resumes_folder),
#             workers=workers,
#             model=model,
#             topk=None if topk == 0 else topk,
#             topk_frac=topk_frac,
#         )
#         duration = time.time() - t0

#         reports_dir = Path("reports")
#         reports = load_reports(reports_dir)

#         if not reports:
#             st.error("⚠️ No reports found. Please check logs.")
#             return

#         # Sort by final_score if available
#         if "final_score" in reports[0]:
#             reports.sort(key=lambda x: x.get("final_score", 0), reverse=True)

#         st.success(f"✅ Done! Processed {len(resumes)} resumes in {duration:.1f}s")
#         st.markdown("### 📊 Candidate Rankings")

#         # Display as table
#         table_data = [
#             {
#                 "Rank": i + 1,
#                 "Candidate": r.get("candidate_name", r["_report_file"]),
#                 "Final Score": r.get("final_score", "N/A"),
#                 "Report File": r["_report_file"],
#             }
#             for i, r in enumerate(reports)
#         ]
#         st.dataframe(table_data, use_container_width=True)

#         # Allow downloading each report
#         st.markdown("### 📥 Download Reports")
#         for r in reports:
#             report_str = json.dumps(r, indent=2)
#             st.download_button(
#                 label=f"Download {r.get('candidate_name', r['_report_file'])}.json",
#                 data=report_str,
#                 file_name=r["_report_file"],
#                 mime="application/json",
#             )

#         # Allow downloading all as ZIP
#         zip_buffer = io.BytesIO()
#         with zipfile.ZipFile(zip_buffer, "w") as zipf:
#             for r in reports_dir.glob("*.json"):
#                 zipf.write(r, arcname=r.name)
#         zip_buffer.seek(0)

#         st.download_button(
#             label="⬇️ Download All Reports (ZIP)",
#             data=zip_buffer,
#             file_name="all_candidate_reports.zip",
#             mime="application/zip",
#         )


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     main()
