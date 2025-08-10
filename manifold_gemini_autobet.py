import json
import time
from datetime import datetime, timedelta
import threading
import os

try:
    import requests
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import track
    from rich.live import Live
    from google import genai
    from google.genai import types
    import keyboard
except ImportError:
    print("This script requires several libraries.")
    print("Please install them using: pip install requests rich google-generativeai keyboard")
    exit()

# --- Unified Configuration ---
# IMPORTANT: Your API keys have been removed from this file for security.
# To run this script, you must set the following environment variables:
#
# 1. MANIFOLD_API_KEY: Your API key from Manifold Markets.
# 2. GEMINI_API_KEY: Your API key for the Gemini API.
#
# For example, in PowerShell:
# $env:MANIFOLD_API_KEY="your_key_here"
# $env:GEMINI_API_KEY="your_key_here"
#
# Or in Command Prompt:
# set MANIFOLD_API_KEY="your_key_here"
# set GEMINI_API_KEY="your_key_here"
#
# To make them permanent, you can set them in your system's environment variable settings.
MANIFOLD_API_KEY = os.environ.get("MANIFOLD_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

MARKET_LIMIT = 50
KELLY_FRACTION = 0.25 # Bet a fraction of the Kelly criterion suggestion to reduce risk. 1.0 is full Kelly.
RESOLUTION_MONTHS_LIMIT = 1
MINIMUM_CONFIDENCE_TO_BET = ["Medium", "High"] # Only bet on predictions with this confidence level.

# --- API Configuration ---
console = Console()
try:
    if not MANIFOLD_API_KEY or not GEMINI_API_KEY:
        console.print("[bold red]Error: MANIFOLD_API_KEY and GEMINI_API_KEY environment variables must be set.[/bold red]")
        exit()

    client = genai.Client(api_key=GEMINI_API_KEY)
    search_tool = types.Tool(google_search=types.GoogleSearch())
    gen_cfg = types.GenerateContentConfig(tools=[search_tool])
    GEMINI_MODEL = "gemini-2.5-pro"
except Exception as e:
    console.print(f"[bold red]Failed to configure Gemini API: {e}[/bold red]")
    exit()

# --- Graceful Exit ---
exit_flag = False
def graceful_exit_listener():
    global exit_flag
    keyboard.wait('q')
    exit_flag = True
    console.print("\n[bold yellow]Exit signal received. Terminating after the current market analysis...[/bold yellow]")

# --- Unified Functions ---

def get_headers(api_key=MANIFOLD_API_KEY):
    """Returns the headers for the API request, including the API key."""
    return {
        'Authorization': f'Key {api_key}',
        'Content-Type': 'application/json'
    }

def get_user_details():
    """Fetches the user's details from Manifold."""
    api_url = "https://api.manifold.markets/v0/me"
    try:
        response = requests.get(api_url, headers=get_headers())
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error fetching user details:[/bold red] {e}")
        return None

def search_manifold_markets(search_term, limit):
    """Searches for markets on Manifold Markets."""
    api_url = "https://api.manifold.markets/v0/search-markets"
    params = {'term': search_term, 'limit': limit}
    try:
        response = requests.get(api_url, params=params, headers=get_headers())
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error fetching data from Manifold API:[/bold red] {e}")
        return None
    except json.JSONDecodeError:
        console.print("[bold red]Error: Failed to decode JSON response.[/bold red]")
        return None

def get_market_by_slug(slug):
    """Fetches the full details of a single market by its slug."""
    api_url = f"https://api.manifold.markets/v0/slug/{slug}"
    try:
        response = requests.get(api_url, headers=get_headers())
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None
    except json.JSONDecodeError:
        return None

def place_bet(market_id, amount, outcome):
    """Places a bet on a given market."""
    global exit_flag
    if exit_flag:
        return False, 0
    api_url = "https://api.manifold.markets/v0/bet"
    payload = {"amount": amount, "contractId": market_id, "outcome": outcome}
    console.print(f"\n[bold green]BETTING:[/bold green] Placing M${amount:.2f} on '{outcome}' for market {market_id}...")
    try:
        response = requests.post(api_url, headers=get_headers(MANIFOLD_API_KEY), json=payload)
        response.raise_for_status()
        console.print("[bold green]✔ BET PLACED SUCCESSFULLY.[/bold green]")
        return True, amount
    except requests.exceptions.RequestException as e:
        error_message = e.response.json().get('message', str(e)) if e.response else str(e)
        if e.response and e.response.status_code == 403:
            error_message += "\n[bold yellow]This is a 403 Forbidden error. Please ensure your MANIFOLD_API_KEY has 'trade' permissions.[/bold yellow]"
        console.print(f"[bold red]✖ FAILED TO PLACE BET:[/bold red] {error_message}")
        return False, 0

def parse_description(description):
    """Parses the description object to extract plain text."""
    if not description:
        return "Not specified."
    if isinstance(description, str):
        return description
    if isinstance(description, dict) and 'content' in description:
        text_parts = []
        for item in description.get('content', []):
            if 'content' in item:
                for sub_item in item['content']:
                    if sub_item.get('type') == 'text' and 'text' in sub_item:
                        text_parts.append(sub_item['text'])
        full_text = " ".join(text_parts).strip()
        return full_text if full_text else "Description not parsable."
    return "Not specified."

def format_timestamp(ts):
    """
    Formats a millisecond timestamp into a human-readable string.
    Includes error handling for out-of-range timestamps.
    """
    if not ts:
        return "N/A"
    try:
        # Convert milliseconds to seconds for fromtimestamp
        return datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
    except (OSError, ValueError):
        # This catches errors from timestamps that are too large (far future) or invalid.
        return "Date out of range"

def _build_market_panel(full_market, gemini_prob_str, gemini_reason_str):
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column(style="bold blue", width=20)
    table.add_column()
    table.add_row("Question:", Text(full_market.get('question', 'N/A'), style="bold white"))
    market_url = f"https://manifold.markets/market/{full_market.get('slug')}"
    table.add_row("URL:", f"[link={market_url}]{market_url}[/link]")
    table.add_row("Market Creator:", f"[cyan]@{full_market.get('creatorUsername', 'N/A')}[/cyan]")
    table.add_row("Resolution Date:", f"[yellow]{format_timestamp(full_market.get('closeTime'))}[/yellow]")
    table.add_row("Total Volume:", f"[green]M${int(full_market.get('volume', 0)):,}[/green]")
    table.add_row("Unique Bettors:", f"{full_market.get('uniqueBettorCount', 0)}")
    outcome_type = full_market.get('outcomeType')
    table.add_row("Market Type:", outcome_type)
    if outcome_type == 'BINARY':
        table.add_row("Market Probability:", f"[bold magenta]{full_market.get('probability', 0):.2%}[/bold magenta]")
    table.add_row("Resolution Criteria:", Text(parse_description(full_market.get('description')), style="italic dim"))
    table.add_row("---", "---")
    table.add_row("Gemini 2.5 Pro Prob:", gemini_prob_str)
    table.add_row("Gemini Reasoning:", Text(gemini_reason_str, style="italic"))
    return Panel(table, border_style="blue", expand=False, title=f"Market Details: {full_market.get('slug')}", title_align="left")

def stream_gemini_analysis(full_market):
    global exit_flag
    prompt = f'''
**[Persona]**
You are a committee of three world-class prediction market analysts and domain experts, assembled to analyze a prediction market.
- **Analyst A (The Bull):** You are an expert in the field and tend to be optimistic. Your role is to build the strongest possible case for a "YES" outcome.
- **Analyst B (The Bear):** You are a skeptical, data-driven analyst who excels at finding risks and counter-arguments. Your role is to build the strongest possible case for a "NO" outcome.
- **Analyst C (The Moderator):** You are a seasoned superforecaster. Your role is to facilitate the debate, weigh the arguments from both sides, and guide the committee to a final, precise probability.

**[Goal]**
Your collective goal is to conduct a rigorous, unbiased analysis and produce the most accurate probability for the following market. You must collaborate and follow the structured process below.

**[Market Information]**
- **Question:** {full_market.get('question', 'N/A')}
- **Resolution Criteria:** {parse_description(full_market.get('description'))}
- **Resolution Date:** {format_timestamp(full_market.get('closeTime'))}

**[Deep Research Protocol]**
Before beginning your analysis, you must conduct a deep and comprehensive research sweep. Your goal is to gather a wide spectrum of information, from official reports to public sentiment. Your research must include, but is not limited to:
- **Official Sources:** Press releases, company statements, scientific papers, and official documentation.
- **News & Media:** Recent news articles from reputable sources, investigative journalism reports, and expert analysis in established publications.
- **Social & Public Sentiment:** Scour social media platforms (like X/Twitter, Reddit), forums, and message boards to gauge the "cultural temperature," public opinion, and identify any grassroots movements or narratives.
- **Blogs & Expert Opinions:** Seek out blog posts and articles from credible domain experts, industry insiders, and respected commentators.
- **Historical Context:** Look for information on similar past events to provide historical context and identify patterns.

Analysts A and B must explicitly use this deep research protocol to build their cases.

**[Structured Analysis Process]**

**Step 1: Independent Analysis & Research (Analysts A & B)**
- **Analyst A (Bull Case):**
  1.  Following the Deep Research Protocol, use the web search tool to find all supporting evidence for a "YES" outcome.
  2.  Present your findings as a numbered list of arguments.
- **Analyst B (Bear Case):**
  1.  Following the Deep Research Protocol, use the web search tool to find all supporting evidence for a "NO" outcome.
  2.  Present your findings as a numbered list of arguments.

**Step 2: Debate and Synthesis (Analyst C)**
- **Moderator's Summary:**
  1.  Briefly summarize the strongest points from both the Bull and Bear cases.
  2.  Identify the key areas of disagreement and uncertainty.
  3.  Weigh the arguments against each other. Which case is stronger and why?

**Step 3: Red Teaming & Final Conclusion (Analyst C)**
- **Devil's Advocate:**
  1.  Challenge the stronger case. What are its biggest weaknesses? What assumptions is it making? What could go wrong?
- **Final Probability and Confidence:**
  1.  Based on the entire analysis, state the final, precise probability.
  2.  Provide a confidence score for this prediction (Low, Medium, or High).
  3.  Briefly justify the confidence level.

**[Output Format]**
Stream your entire analysis as plain text. After you have explained your thinking, write the token `[END_OF_REASONING]` on a new line. Finally, provide a JSON object with two keys: "probability" and "confidence".

Example JSON output:
```json
{{
  "probability": 0.72,
  "confidence": "Medium"
}}
```
    '''
    
    reasoning_text = ""
    full_response_text = ""
    gemini_prob = None
    gemini_confidence = None

    with Live(console=console, refresh_per_second=10) as live:
        live.update(_build_market_panel(full_market, "Researching...", "…"))
        
        try:
            full_response_text = ""
            for chunk in client.models.generate_content_stream(model=GEMINI_MODEL, contents=prompt, config=gen_cfg):
                if exit_flag:
                    break
                if getattr(chunk, "text", None):
                    full_response_text += chunk.text
                    reasoning_text = full_response_text.split("[END_OF_REASONING]")[0]
                    live.update(_build_market_panel(full_market, "Thinking...", reasoning_text + "…"))
        except Exception as e:
            live.update(_build_market_panel(full_market, "[red]Error[/red]", f"API Error: {e}"))
            return None, None
        
        if exit_flag:
            return None, None

        try:
            parts = full_response_text.split("[END_OF_REASONING]")
            final_reasoning = parts[0].strip()
            json_part = parts[1].strip().replace("```json", "").replace("```", "").strip()
            gemini_data = json.loads(json_part)
            gemini_prob = float(gemini_data["probability"])
            gemini_confidence = gemini_data["confidence"]
            final_prob_str = f"[bold green]{gemini_prob:.2%}[/bold green] (Confidence: {gemini_confidence})"
        except (json.JSONDecodeError, IndexError, ValueError, KeyError):
            final_prob_str = "[red]Error[/red]"
            final_reasoning = "[red]Failed to parse model output.[/red]"
        live.update(_build_market_panel(full_market, final_prob_str, final_reasoning))

    return gemini_prob, gemini_confidence

def main_gemini_autobet(search_query):
    """Main function to run Gemini-powered auto-betting."""
    global exit_flag
    console.print(Panel(f"Searching for markets related to: [bold cyan]'{search_query}'[/bold cyan]",
                        title="Manifold + Gemini 2.5 Pro AUTOBET", border_style="red"))

    user_details = get_user_details()
    if user_details is None:
        return
    
    balance = user_details.get('balance', 0)
    total_deposits = user_details.get('totalDeposits', 0)
    profit_cached = user_details.get('profitCached', {}).get('allTime', 0)
    net_worth = balance + total_deposits

    stats_table = Table(title="Your Manifold Stats", show_header=False, box=None, padding=(0, 1))
    stats_table.add_column(style="bold blue", width=20)
    stats_table.add_column(style="bold green")
    stats_table.add_row("Balance:", f"M${balance:,.2f}")
    stats_table.add_row("Net Worth (Balance + Deposits):", f"M${net_worth:,.2f}")
    stats_table.add_row("All-Time Profit:", f"M${profit_cached:,.2f}")
    console.print(Panel(stats_table, border_style="green"))

    with console.status("[bold green]Searching for markets...[/bold green]"):
        markets = search_manifold_markets(search_query, limit=MARKET_LIMIT)
    
    if not markets:
        console.print("[bold red]Could not retrieve markets or no markets found.[/bold red]")
        return

    now = datetime.now()
    cutoff_date = now + timedelta(days=RESOLUTION_MONTHS_LIMIT * 30)
    
    recent_open_markets = []
    for m in markets:
        if not m.get('isResolved') and m.get('closeTime'):
            close_time = datetime.fromtimestamp(m['closeTime'] / 1000)
            if now < close_time < cutoff_date:
                recent_open_markets.append(m)

    console.print(f"\nFound {len(recent_open_markets)} open markets resolving in the next {RESOLUTION_MONTHS_LIMIT} month(s). Analyzing...\n")

    for market_summary in recent_open_markets:
        if exit_flag:
            console.print("[bold yellow]Exiting gracefully...[/bold yellow]")
            break
        slug = market_summary.get('slug')
        if not slug: continue
        
        full_market = get_market_by_slug(slug)
        if not full_market:
            console.print(f"[yellow]Could not fetch details for market slug: {slug}[/yellow]")
            continue
            
        if full_market.get('outcomeType') != 'BINARY':
            console.print(f"[dim yellow]Skipping non-binary market: {slug}[/dim yellow]")
            continue

        console.print(Panel(f"Analyzing market: [bold cyan]{full_market.get('question')}[/bold cyan]", border_style="blue"))
        gemini_prob, gemini_confidence = stream_gemini_analysis(full_market)

        if gemini_prob is not None and gemini_confidence in MINIMUM_CONFIDENCE_TO_BET:
            market_prob = full_market.get('probability', 0)
            edge = gemini_prob - market_prob

            if abs(edge) > 0.01: # Minimum edge to consider a bet
                if edge > 0: # Bet on YES
                    p_win = gemini_prob
                    odds = 1 / market_prob
                    outcome = "YES"
                else: # Bet on NO
                    p_win = 1 - gemini_prob
                    odds = 1 / (1 - market_prob)
                    outcome = "NO"

                if odds <= 1: # Avoid division by zero or negative odds
                    continue

                kelly_percentage = (p_win * odds - 1) / (odds - 1)
                bet_amount = balance * kelly_percentage * KELLY_FRACTION

                if bet_amount >= 1:
                    if bet_amount > balance:
                        bet_amount = balance
                    
                    bet_placed, amount_bet = place_bet(full_market['id'], bet_amount, outcome)
                    if bet_placed:
                        balance -= amount_bet
                        console.print(f"[bold blue]New balance after bet:[/bold blue] M${balance:,.2f}")
                else:
                    console.print(f"\n[yellow]ANALYSIS:[/yellow] Kelly bet amount is less than M$1. No bet placed.")
            else:
                console.print(f"\n[yellow]ANALYSIS:[/yellow] No significant edge found. Gemini: {gemini_prob:.1%}, Market: {market_prob:.1%}. Holding.")
        elif gemini_prob is not None:
            console.print(f"\n[yellow]ANALYSIS:[/yellow] Confidence level '{gemini_confidence}' is below the minimum required to bet. Holding.")

        time.sleep(1)

if __name__ == "__main__":
    exit_thread = threading.Thread(target=graceful_exit_listener, daemon=True)
    exit_thread.start()

    while not exit_flag:
        console.print(Panel("Welcome to the Manifold + Gemini 2.5 Pro AUTOBET Script!", title="Main Menu", border_style="green"))
        console.print("Press 'q' at any time to gracefully exit after the current market analysis.")
        search_query = input("Enter the topic of markets to bet on (or type 'exit' to quit): ")
        if search_query.lower() == 'exit' or exit_flag:
            break
        main_gemini_autobet(search_query)
    
    console.print("[bold green]Script finished.[/bold green]")
