# Manifold LLM Betting

Automation scripts to analyze Manifold Markets and optionally place bets using LLMs.

## Overview

This project contains two Python scripts for automated betting on Manifold Markets:

-   `modular_manifold_bettor.py`: Uses a model from OpenRouter to analyze and bet on markets.
-   `manifold_gemini_autobet.py`: Uses Google Gemini 2.5 Pro to analyze and bet on markets.

Both scripts share a common set of features, including:

-   Reading API keys from environment variables.
-   Displaying rich console output.
-   Gracefully stopping on user input.
-   Placing live trades.

## Features

-   **Modular Design:** The project is designed to be modular, with common functionality shared between the two scripts.
-   **Model Selection:** The `modular_manifold_bettor.py` script allows the user to select a model from a list of available models.
-   **Configurable Betting Parameters:** The betting parameters can be configured from the command line.
-   **Bet Logging:** The scripts log all bets to a CSV file.
-   **Dry-Run Mode:** The scripts can be run in a dry-run mode, where they only simulate bets without actually placing them.

## Requirements

-   Python 3.9+
-   The following Python packages:
    -   `requests`
    -   `rich`
    -   `keyboard`
    -   `google-genai`

To install the required packages, run the following command:

```
pip install requests rich keyboard google-genai
```

## Environment Variables

Before running the scripts, you need to set the following environment variables:

-   `MANIFOLD_API_KEY`: Your Manifold Markets API key.
-   `OPENROUTER_API_KEY`: Your OpenRouter API key (for the `modular_manifold_bettor.py` script).
-   `GEMINI_API_KEY`: Your Google Gemini API key (for the `manifold_gemini_autobet.py` script).

## Usage

To run the scripts, use the following commands:

```
python modular_manifold_bettor.py
```

```
python manifold_gemini_autobet.py
```

The scripts accept the following command-line arguments:

-   `--kelly-fraction`: The fraction of the Kelly criterion to use for betting.
-   `--resolution-months-limit`: The maximum number of months in the future to consider for markets.
-   `--min-confidence`: The minimum confidence level required to place a bet.
-   `--dry-run`: If set, the script will not place any bets.

## Caution

These scripts can place real bets on your Manifold account. Use them with care and at your own risk.