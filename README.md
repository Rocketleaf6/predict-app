# Numerology Hiring Scorer (Streamlit)

Python Streamlit web app for macOS with numerology scoring, role details, resume upload, and optional OpenAI analysis.

## Features
- Candidate + Leader DOB numerology scoring (Destiny/Birth/Month)
- Full rule-engine matrices and composite scoring
- Role inputs:
  - Role Name
  - Role Description
  - Role Type (Execution Focused / Strategy Focused)
- Resume upload:
  - `.pdf`
  - `.txt`
- Optional OpenAI integration:
  - Resume summary
  - Fit assessment vs role
  - Interview focus points
  - Final recommendation

## Run on macOS
```bash
cd /Users/rahilnuwal/Documents/CODEX/numerology_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## OpenAI setup
- Enter API key in sidebar field `OpenAI API Key (optional)`.
- App works without API key for numerology-only output.

## Notes
- Development specification is in `/Users/rahilnuwal/Documents/CODEX/numerology_app/DEVELOPMENT_DOCUMENT.md`.
- Verdict bands:
  - `>= 80`: Strong Hire
  - `65-79`: Hire w/ Guardrails
  - `< 65`: Weak / Risk
