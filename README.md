# Manifold LLM Betting

Automation scripts to analyze Manifold Markets and optionally place bets using LLMs.

## Overview
- `modular_manifold_bettor.py`: Uses OpenRouter (e.g., `google/gemini-2.5-pro:online`) to analyze markets and place bets on Manifold.
- `manifold_gemini_autobet.py`: Uses Google Gemini 2.5 Pro via the Google GenAI SDK to analyze markets and place bets on Manifold.

Both scripts:
- Read API keys from environment variables (no secrets in repo).
- Show rich console output and listen for `q` to gracefully stop.
- Can place live trades; use with care and small stakes.

## Requirements
- Python 3.9+
- Packages:
  - Core: `requests`, `rich`, `keyboard`
  - For Gemini script: `google-genai` (new SDK)

Install:
```
pip install requests rich keyboard google-genai
```

Note: If using an older Gemini sample or different imports, you might need `google-generativeai` instead. This repo’s `manifold_gemini_autobet.py` uses the new `google-genai` SDK (`from google import genai`).

## Environment Variables
Copy `.env.example` to your secrets manager or export these in your shell:
- `MANIFOLD_API_KEY`: Manifold Markets API key (with trade permissions if betting).
- `OPENROUTER_API_KEY`: OpenRouter API key (for `modular_manifold_bettor.py`).
- `GEMINI_API_KEY`: Google Gemini API key (for `manifold_gemini_autobet.py`).

Windows PowerShell examples:
```
$env:MANIFOLD_API_KEY="your_key_here"
$env:OPENROUTER_API_KEY="your_key_here"
$env:GEMINI_API_KEY="your_key_here"
```

Command Prompt examples:
```
set MANIFOLD_API_KEY=your_key_here
set OPENROUTER_API_KEY=your_key_here
set GEMINI_API_KEY=your_key_here
```

## Usage
Run OpenRouter-based bettor:
```
python modular_manifold_bettor.py
```

Run Gemini-based bettor:
```
python manifold_gemini_autobet.py
```

Press `q` to exit gracefully after the current market analysis.

## Configuration Notes
- Betting controls (e.g., `KELLY_FRACTION`, `MIN_EDGE`, market search limits) are constants near the top of each script.
- `modular_manifold_bettor.py` uses an OpenRouter model ID string like `google/gemini-2.5-pro:online`. Ensure your key has access to the chosen model.
- Network calls are limited to Manifold API and the respective LLM provider; no filesystem writes beyond normal Python runtime.

## Caution
These scripts can place real bets on your Manifold account. Start with small stakes, confirm API permissions, and monitor output. You are responsible for all trades executed by these tools.

