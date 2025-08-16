import json
import time
from datetime import datetime, timedelta
import threading
from rich.live import Live
import google.generativeai as genai
from rich.panel import Panel
from rich.table import Table

from common import (
    console,
    exit_flag,
    graceful_exit_listener,
    get_user_details,
    search_manifold_markets,
    get_market_by_slug,
    place_bet,
    parse_description,
    format_timestamp,
    build_market_panel,
    GEMINI_API_KEY,
    MANIFOLD_API_KEY,
    parse_args
)

# --- Unified Configuration ---
args = parse_args()
MARKET_LIMIT = 50
KELLY_FRACTION = args.kelly_fraction
RESOLUTION_MONTHS_LIMIT = args.resolution_months_limit
MINIMUM_CONFIDENCE_TO_BET = [args.min_confidence, "High"]

# --- API Configuration ---
try:
    if not MANIFOLD_API_KEY or not GEMINI_API_KEY:
        console.print("[bold red]Error: MANIFOLD_API_KEY and GEMINI_API_KEY environment variables must be set.[/bold red]")
        exit()

    genai.configure(api_key=GEMINI_API_KEY)  # type: ignore[attr-defined]
    GEMINI_MODEL = "gemini-2.5-pro"
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)  # type: ignore[attr-defined]
except Exception as e:
    console.print(f"[bold red]Failed to configure Gemini API: {e}[/bold red]")
    exit()

def _build_market_panel(full_market, gemini_prob_str, gemini_reason_str):
    return build_market_panel(full_market, gemini_prob_str, gemini_reason_str, "Gemini 2.5 Pro")

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
            # Stream model output incrementally
            response = gemini_model.generate_content(prompt, stream=True)
            for chunk in response:
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
                # Guard against division-by-zero and invalid odds
                EPS = 1e-9
                if edge > 0: # Bet on YES
                    if market_prob <= EPS:
                        console.print("[dim yellow]Skipping: market probability too low for stable odds (YES).[/dim yellow]")
                        continue
                    p_win = gemini_prob
                    odds = 1 / market_prob
                    outcome = "YES"
                else: # Bet on NO
                    denom = 1 - market_prob
                    if denom <= EPS:
                        console.print("[dim yellow]Skipping: market probability too high for stable odds (NO).[/dim yellow]")
                        continue
                    p_win = 1 - gemini_prob
                    odds = 1 / denom
                    outcome = "NO"

                if odds <= 1: # Avoid degenerate or unprofitable odds
                    continue

                kelly_percentage = (p_win * odds - 1) / (odds - 1)
                if kelly_percentage <= 0:
                    console.print("[dim]Kelly fraction <= 0; no bet.[/dim]")
                    continue
                bet_amount = balance * kelly_percentage * KELLY_FRACTION

                if bet_amount >= 1:
                    if bet_amount > balance:
                        bet_amount = balance
                    
                    bet_placed, amount_bet = place_bet(full_market['id'], bet_amount, outcome, full_market, gemini_prob, gemini_confidence, GEMINI_MODEL, args.dry_run)
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
