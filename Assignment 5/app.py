"""Streamlit UI for the Assignment 5 resume matcher."""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

st.set_page_config(
    page_title="Resume Match | Assignment 5",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.8rem; padding-bottom: 2.5rem; max-width: 1180px; }
      .hero { padding: 1.3rem 1.5rem; border: 1px solid rgba(128,128,128,.25);
              border-radius: 16px; margin-bottom: 1.2rem; }
      .muted { opacity: .75; }
      div[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.2);
          border-radius: 12px; padding: .8rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0">📄 Resume Match</h1>
      <p style="margin:.5rem 0 0 0">Find resumes with evidence for <b>at least one</b> requested skill.
      Results include a short, readable explanation and evidence from the resume.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Search settings")
    max_results = st.slider("Maximum resumes to show", min_value=1, max_value=100, value=30)
    filter_experience = st.checkbox("Filter by minimum experience", value=False)
    min_experience = st.number_input(
        "Minimum years of experience", min_value=0.0, max_value=50.0,
        value=1.0, step=0.5, disabled=not filter_experience,
    )
    st.divider()
    st.subheader("Resume index")
    st.caption("Use this after adding or changing files in the resumes folder.")
    refresh_index = st.button("Index / refresh resumes", use_container_width=True)

if refresh_index:
    if not os.getenv("COHERE_API_KEY"):
        st.error("COHERE_API_KEY was not found. Add it to the .env file in this project folder, then restart the app.")
    else:
        try:
            import resume_rag
            with st.spinner("Reading resumes and updating the index… this may use Cohere API credits."):
                resume_rag.index_resumes()
            st.success("Resume indexing finished. Check the terminal for any per-file warnings.")
        except Exception as exc:
            st.error(f"Could not index resumes: {exc}")

left, right = st.columns([1.55, 1])
with left:
    job_description = st.text_area(
        "Job description",
        height=250,
        placeholder="Paste the job description here. Include responsibilities and required skills.",
        help="Skills mentioned in the description are detected automatically where supported.",
    )
with right:
    extra_skills_text = st.text_area(
        "Skills to check (optional)",
        height=150,
        placeholder="Java, Spring Boot, SQL, REST API",
        help="Comma-separated skills. These are evaluated individually; by default, a resume appears if at least one is matched.",
    )
    st.info("Default behavior: show a resume when **any one** requested skill has verified matched evidence. Skills marked partial or missing do not count as matched.")

search_clicked = st.button("Find matching resumes", type="primary", use_container_width=True)


def _parse_skills(raw: str) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in re.split(r"[,;\n]+", raw or "") if item.strip()))


def _shorten(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _short_description(candidate: dict) -> str:
    matched = candidate.get("matched_skills", [])
    evidence_map = candidate.get("skill_evidence", {}) or {}
    snippets = []
    for skill in matched:
        item = evidence_map.get(skill, {}) or {}
        evidence = _shorten(item.get("evidence", ""), 180)
        if evidence and evidence not in snippets:
            snippets.append(evidence)
        if len(snippets) >= 2:
            break
    if snippets:
        return "Resume evidence: " + " | ".join(snippets)
    return "The resume contains verified evidence for " + ", ".join(matched) + ". Review the evidence details below for context."


def _render_candidate(candidate: dict, index: int) -> None:
    name = candidate.get("candidate_name") or "Unnamed candidate"
    matched = candidate.get("matched_skills") or []
    experience = candidate.get("experience_years")
    path_value = candidate.get("resume_path") or ""

    with st.container(border=True):
        heading, score_col = st.columns([4, 1])
        with heading:
            st.subheader(name)
            st.caption(_short_description(candidate))
        with score_col:
            st.metric("Match score", f"{candidate.get('score', 0):.0f}%")

        metric_cols = st.columns(3)
        metric_cols[0].metric("Matched skills", len(matched))
        metric_cols[1].metric("Experience", f"{experience:g} yrs" if isinstance(experience, (int, float)) else "Not stated")
        metric_cols[2].metric("Semantic relevance", f"{candidate.get('semantic_similarity', 0):.0f}%")
        st.markdown("**Matched skills**")
        st.write(" · ".join(matched))

        with st.expander("View skill evidence and resume details"):
            evidence_map = candidate.get("skill_evidence", {}) or {}
            for skill, detail in evidence_map.items():
                status = detail.get("status", "unknown")
                label = {"matched": "✅ Matched", "partial": "🟡 Partial", "missing": "⚪ Missing"}.get(status, status.title())
                st.markdown(f"**{skill} — {label}**")
                evidence = detail.get("evidence", "")
                reason = detail.get("reason", "")
                if evidence:
                    st.write(f"> {evidence}")
                if reason:
                    st.caption(reason)
            if path_value:
                st.caption(f"Resume path: {path_value}")
                try:
                    resume_path = Path(path_value)
                    if resume_path.is_file():
                        st.download_button(
                            "Download resume",
                            data=resume_path.read_bytes(),
                            file_name=resume_path.name,
                            key=f"download_{index}_{name}",
                            use_container_width=False,
                        )
                except OSError:
                    pass


if search_clicked:
    if not job_description.strip():
        st.warning("Please paste a job description first.")
    elif not os.getenv("COHERE_API_KEY"):
        st.error("COHERE_API_KEY was not found. Add it to the .env file in this project folder, then restart the app.")
    else:
        try:
            import job_matcher

            extra_skills = _parse_skills(extra_skills_text)
            with st.spinner("Searching the resume index and checking skill evidence…"):
                # Do not pass the user-entered skills as mandatory requirements here.
                # This is intentional: a resume with any matched skill should be visible.
                results = job_matcher.match_candidates(
                    job_description=job_description,
                    mandatory_skills=[],
                    min_experience=None,
                    top_n=10000,
                    skills_to_evaluate=extra_skills,
                )

            if results.get("message"):
                st.warning(results["message"])
            else:
                all_results = results.get("matches", [])
                any_skill_results = [item for item in all_results if item.get("matched_skills")]
                if filter_experience:
                    any_skill_results = [
                        item for item in any_skill_results
                        if isinstance(item.get("experience_years"), (int, float))
                        and item["experience_years"] >= min_experience
                    ]
                any_skill_results.sort(key=lambda item: item.get("score", 0), reverse=True)
                displayed = any_skill_results[:max_results]

                st.session_state["resume_match_results"] = {
                    "results": displayed,
                    "indexed": results.get("total_candidates_indexed", len(all_results)),
                    "eligible": len(any_skill_results),
                    "filter_experience": filter_experience,
                    "min_experience": min_experience,
                }
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            st.caption("Check your .env settings, Cohere API access, and whether resumes have been indexed by resume_rag.py.")

saved = st.session_state.get("resume_match_results")
if saved:
    st.divider()
    st.header("Matching resumes")
    c1, c2, c3 = st.columns(3)
    c1.metric("Resumes indexed", saved.get("indexed", 0))
    c2.metric("Resumes with ≥1 matched skill", saved.get("eligible", 0))
    c3.metric("Shown", len(saved.get("results", [])))

    if saved.get("filter_experience"):
        st.caption(f"Experience filter applied: at least {saved.get('min_experience', 0):g} years. Candidates with unknown experience are excluded by this optional filter.")

    if not saved.get("results"):
        st.warning("No resumes had at least one verified matched skill under the current settings. Try checking the skill list or updating the resume index.")
    else:
        st.caption("Results are ranked by the match score. A partial skill does not count as a matched skill.")
        for idx, candidate in enumerate(saved["results"]):
            _render_candidate(candidate, idx)

st.divider()
st.caption("Assignment 5 · Resume matching with Cohere embeddings, evidence-based skill evaluation, ChromaDB, and Streamlit")
