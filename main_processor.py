
from langraph_agent import *
# ---------- UPDATED process_all ----------
def process_all(job_description_file: str, resumes_folder: str, workers: int = 2, model: str = "gpt-4o-mini", topk: Optional[int] = None, topk_frac: float = 0.4) -> None:
    reports_dir = Path("reports"); reports_dir.mkdir(exist_ok=True)

    jd = load_yaml_job_description(job_description_file)

    # Gather candidate files
    resume_files = [Path(resumes_folder) / fn for fn in os.listdir(resumes_folder)
                    if fn.lower().endswith((".pdf", ".txt"))]

    # Pre-load resume texts once for embedding prefilter
    resume_texts = [load_resume_markdown(str(p)) for p in resume_files]

    # Pre-filter via embeddings
    ranked = prefilter_resumes(jd, resume_files, resume_texts, topk=topk, topk_frac=topk_frac)
    selected_files = [p for p, _ in ranked]
    logging.info(f"Pre-filter selected {len(selected_files)}/{len(resume_files)} resumes via embeddings")

    reports: List[Dict[str, Any]] = []

    def _worker(resume_path: Path) -> Optional[Dict[str, Any]]:
        # Safer to create an LLM client per thread
        structured_llm = make_llm(model=model)
        return run_single_resume_with_graph(jd, resume_path, structured_llm)

    # Fan out across a thread pool, each calling the LangGraph workflow
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {ex.submit(_worker, p): p for p in selected_files}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                rep = fut.result()
                if rep:
                    reports.append(rep)
                    # (Already saved inside the graph's save node)
            except Exception as e:
                logging.error(f"Failed {p.name}: {e}")

    logging.info(f"Wrote {len(reports)} reports to {reports_dir.resolve()}")
    
if __name__ == "__main__":
    job_description_file = r"C:\Users\Lenovo\resume_matcher\jd.yaml"
    resumes_folder = r"C:\Users\Lenovo\resume_matcher\resumes"
    topk = 5
    Path("reports").mkdir(exist_ok=True)

    t0 = time.time()
    reports = process_all(job_description_file=job_description_file, resumes_folder=resumes_folder, topk=topk)
    logging.info(f"Done processing resumes in {time.time() - t0:.1f}s")
