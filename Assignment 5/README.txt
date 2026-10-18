ASSIGNMENT 5 - RESUME MATCHER (STREAMLIT UI)

FILES
- app.py: Streamlit interface. Shows a resume when at least one requested skill is verified as matched.
- job_matcher.py: Resume search, skill evidence evaluation, scoring, and ranking.
- resume_rag.py: Resume indexing and text cleanup.
- fs_tools.py: Existing PDF/TXT/DOCX text extraction utilities.
- run_app.bat: Windows launcher.
- requirements.txt: Python dependencies.
- .env.example: Example environment settings (do not put a real API key in a shared ZIP).

SETUP
1. Keep all files in the project folder with the existing resumes folder.
2. If you already have a working .env, keep it. Otherwise copy .env.example to .env and edit COHERE_API_KEY.
3. Double-click run_app.bat. The first launch creates .venv and installs dependencies.
4. The browser opens to the Streamlit UI. Keep the command window open while using it.
5. Use "Index / refresh resumes" after adding or changing resumes. Indexing calls Cohere and may use API credits.
6. Paste a job description, optionally enter comma-separated skills, then click "Find matching resumes".

MATCHING BEHAVIOR
- Default: a resume is shown if at least one requested skill has status "matched".
- Partial or missing skills do not count as a matched skill.
- Optional experience filter is off by default. If enabled, candidates with unknown experience are excluded.
- Each result has a short summary, match score, evidence excerpts, and a resume download button when the original file path is available.

IMPORTANT
- Keep your existing .env and chroma_db folder; do not share API keys.
- Run resume_rag.py or use the UI's indexing button after updating resumes.
- Scanned/image-only PDFs may need OCR, which is not included.
