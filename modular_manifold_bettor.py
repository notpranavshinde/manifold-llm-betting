import json
import re
import time
from datetime import datetime, timedelta
import threading
import requests
from rich.live import Live
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
    MANIFOLD_API_KEY,
    OPENROUTER_API_KEY,
    parse_args
)

# --- Model Configuration ---
args = parse_args()
MODEL_NAME = "google/gemini-2.5-pro:online"

# --- Betting Configuration ---
MARKET_LIMIT = 50
KELLY_FRACTION = args.kelly_fraction
RESOLUTION_MONTHS_LIMIT = args.resolution_months_limit
MINIMUM_CONFIDENCE_TO_BET = [args.min_confidence, "High"]
MIN_EDGE = 0.01

# --- API Configuration ---
if not MANIFOLD_API_KEY or not OPENROUTER_API_KEY:
    console.print("[bold red]Error: MANIFOLD_API_KEY and OPENROUTER_API_KEY environment variables must be set.[/bold red]")
    exit()

HTTP_TIMEOUT = 60

def _build_market_panel(full_market, model_prob_str, model_reason_str, model_name):
    return build_market_panel(full_market, model_prob_str, model_reason_str, model_name)

def format_request_error(e: requests.exceptions.RequestException) -> str:
    """Extract a helpful message from a requests exception."""
    msg = str(e)
    resp = getattr(e, "response", None)
    if resp is None:
        return msg
    # Try JSON envelope: { "error": { "message": "...", "code": "..." } }
    try:
        ej = resp.json()
        if isinstance(ej, dict):
            if "error" in ej and isinstance(ej["error"], dict):
                code = ej["error"].get("code")
                m = ej["error"].get("message") or msg
                return f"{m} (code: {code}, http {resp.status_code})"
            # OpenAI-style: {"error": {"message": "..."}}
            if "message" in ej:
                return f"{ej.get('message')} (http {resp.status_code})"
    except Exception:
        pass
    # Fallback to text
    try:
        t = resp.text
        if t:
            return f"HTTP {resp.status_code}: {t[:1000]}"
    except Exception:
        pass
    return msg

def parse_model_output_to_prob_conf(text: str):
    """
    Expect model to output ... [END_OF_REASONING] then a JSON object.
    Be robust: try to split on token; if missing, try to grab the last {...} JSON blob.
    """
    final_reasoning = ""
    model_prob = None
    model_conf = None

    if "[END_OF_REASONING]" in text:
        parts = text.split("[END_OF_REASONING]")
        final_reasoning = parts[0].strip()
        json_part = parts[1]
    else:
        # Try to find the last JSON object in the text
        m = list(re.finditer(r"\{(?:[^{}]|(?R))*\}\s*$", text, re.DOTALL))
        if m:
            final_reasoning = text[:m[-1].start()].strip()
            json_part = text[m[-1].start():m[-1].end()]
        else:
            raise ValueError("Could not locate JSON in model output.")

    # Strip fences if any
    json_part = json_part.replace("```json", "").replace("```", "").strip()
    data = json.loads(json_part)

    # Pull fields
    model_prob = float(data["probability"])
    model_conf = data["confidence"]
    return final_reasoning, model_prob, model_conf

def get_available_models():
    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        models = response.json().get('data', [])
        return [model['id'] for model in models]
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error fetching available models:[/bold red] {format_request_error(e)}")
        return []

def get_model_analysis(full_market, model_name):
    """
    Call OpenRouter, get model reasoning + {probability, confidence}.
    """
    global exit_flag

    prompt = f'''
**[Persona]**
You are a committee of three world-class prediction market analysts and domain experts, assembled to analyze a prediction market.
- **Analyst A (The Bull):** Build the strongest case for "YES".
- **Analyst B (The Bear):** Build the strongest case for "NO".
- **Analyst C (The Moderator):** Weigh both sides and give a precise probability.

**[Market Information]**
- **Question:** {full_market.get('question', 'N/A')}
- **Resolution Criteria:** {parse_description(full_market.get('description'))}
- **Resolution Date:** {format_timestamp(full_market.get('closeTime'))}

**[Deep Research Protocol]**
Use official sources, news, blogs/experts, social sentiment, and historical context.

**[Output Format]**
Stream your reasoning. After you have explained your thinking, write the token `[END_OF_REASONING]` on a new line.
Then provide a JSON object with keys "probability" (0..1) and "confidence" ("Low"|"Medium"|"High").
'''

    model_prob = None
    model_confidence = None

    with Live(console=console, refresh_per_second=10) as live:
        live.update(_build_market_panel(full_market, "Querying model...", "", model_name))

        # OpenRouter call (fixed)
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            # Optional but recommended by OpenRouter to identify your app:
            "HTTP-Referer": "http://localhost",
            "X-Title": "Manifold AutoBet",
        }

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            # If supported by your chosen model, you can try to force JSON:
            # "response_format": {"type": "json_object"},
        }

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=HTTP_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            full_response_text = data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            error_message = format_request_error(e)
            live.update(_build_market_panel(full_market, "[red]Error[/red]", f"API Error: {error_message}", model_name))
            return None, None

        if exit_flag:
            return None, None

        try:
            final_reasoning, model_prob, model_confidence = parse_model_output_to_prob_conf(full_response_text)
            final_prob_str = f"[bold green]{model_prob:.2%}[/bold green] (Confidence: {model_confidence})"
        except Exception as e:
            final_prob_str = "[red]Error[/red]"
            final_reasoning = f"[red]Failed to parse model output.[/red] ({e})"

        live.update(_build_market_panel(full_market, final_prob_str, final_reasoning, model_name))

    return model_prob, model_confidence

def main_modular_autobet(search_query, model_name):
    """Main function."""
    global exit_flag
    console.print(Panel(f"Searching for markets related to: [bold cyan]'{search_query}'[/bold cyan]",
                        title="Manifold + Model AUTOBET", border_style="red"))

    user_details = get_user_details()
    if user_details is None:
        return

    balance = float(user_details.get('balance', 0))
    total_deposits = float(user_details.get('totalDeposits', 0))
    profit_cached = float(user_details.get('profitCached', {}).get('allTime', 0))
    net_worth = balance + total_deposits

    stats_table = Table(title="Your Manifold Stats", show_header=False, box=None, padding=(0, 1))
    stats_table.add_column(style="bold blue", width=28)
    stats_table.add_column(style="bold green")
    stats_table.add_row("Balance:", f"M${balance:,.2f}")
    stats_table.add_row("Net Worth (Balance + Deposits):", f"M${net_worth:,.2f}")
    stats_table.add_row("All-Time Profit:", f"M${profit_cached:,.2f}")
    stats_table.add_row("Model in Use:", f"[bold cyan]{model_name}[/bold cyan]")
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
        try:
            if not m.get('isResolved') and m.get('closeTime'):
                close_time = datetime.fromtimestamp(m['closeTime'] / 1000)
                if now < close_time < cutoff_date:
                    recent_open_markets.append(m)
        except Exception:
            continue

    console.print(f"\nFound {len(recent_open_markets)} open markets resolving in the next {RESOLUTION_MONTHS_LIMIT} month(s). Analyzing...\n")

    for market_summary in recent_open_markets:
        if exit_flag:
            console.print("[bold yellow]Exiting gracefully...[/bold yellow]")
            break

        slug = market_summary.get('slug')
        if not slug:
            continue

        full_market = get_market_by_slug(slug)
        if not full_market:
            console.print(f"[yellow]Could not fetch details for market slug: {slug}[/yellow]")
            continue

        if full_market.get('outcomeType') != 'BINARY':
            console.print(f"[dim yellow]Skipping non-binary market: {slug}[/dim yellow]")
            continue

        console.print(Panel(f"Analyzing market: [bold cyan]{full_market.get('question')}[/bold cyan]", border_style="blue"))
        model_prob, model_confidence = get_model_analysis(full_market, model_name)

        if model_prob is not None and model_confidence in MINIMUM_CONFIDENCE_TO_BET:
            try:
                market_prob = float(full_market.get('probability', 0))
            except Exception:
                market_prob = 0.0

            edge = model_prob - market_prob

            if abs(edge) > MIN_EDGE:
                if edge > 0:  # Bet YES
                    p_win = model_prob
                    if market_prob <= 0:
                        console.print("[yellow]Market prob is 0; skipping to avoid div-by-zero odds.[/yellow]")
                        continue
                    odds = 1.0 / market_prob
                    outcome = "YES"
                else:  # Bet NO
                    p_win = 1.0 - model_prob
                    if market_prob >= 1:
                        console.print("[yellow]Market prob is 1; skipping to avoid div-by-zero odds.[/yellow]")
                        continue
                    odds = 1.0 / (1.0 - market_prob)
                    outcome = "NO"

                if odds <= 1:  # sanity
                    console.print("[yellow]Odds <= 1; skipping.[/yellow]")
                    continue

                kelly_percentage = (p_win * odds - 1.0) / (odds - 1.0)
                bet_amount = balance * kelly_percentage * KELLY_FRACTION

                if bet_amount >= 1:
                    if bet_amount > balance:
                        bet_amount = balance

                    bet_placed, amount_bet = place_bet(full_market['id'], bet_amount, outcome, full_market, model_prob, model_confidence, model_name, args.dry_run)
                    if bet_placed:
                        balance -= amount_bet
                        console.print(f"[bold blue]New balance after bet:[/bold blue] M${balance:,.2f}")
                else:
                    console.print(f"\n[yellow]ANALYSIS:[/yellow] Kelly bet amount < M$1. No bet placed.")
            else:
                console.print(f"\n[yellow]ANALYSIS:[/yellow] No significant edge. Model: {model_prob:.1%}, Market: {market_prob:.1%}. Holding.")
        elif model_prob is not None:
            console.print(f"\n[yellow]ANALYSIS:[/yellow] Confidence '{model_confidence}' below threshold. Holding.")

        time.sleep(1)

'''if __name__ == "__main__":
    exit_thread = threading.Thread(target=graceful_exit_listener, daemon=True)
    exit_thread.start()

    models = get_available_models()
    if not models:
        console.print("[bold red]Could not fetch available models. Exiting.[/bold red]")
        exit()

    while not exit_flag:
        console.print(Panel("Welcome to the Manifold + Model AUTOBET Script!", title="Main Menu", border_style="green"))
        console.print("Press 'q' at any time to gracefully exit after the current market analysis.")
        
        model_name = None
        while not model_name:
            try:
                search_term = input("Enter a search term to filter models (or press Enter to list all): ")
                filtered_models = [m for m in models if search_term.lower() in m.lower()]

                if not filtered_models:
                    console.print("[bold red]No models found matching your search term. Please try again.[/bold red]")
                    continue
                
                if len(filtered_models) == 1:
                    model_name = filtered_models[0]
                    console.print(f"Found one model: [bold cyan]{model_name}[/bold cyan]")
                else:
                    console.print("
Available Models:")
                    for i, model in enumerate(filtered_models):
                        console.print(f"[{i+1}] {model}")
                    
                    selection = input("Select a model to use (enter the number): ")
                    selected_index = int(selection) - 1
                    if 0 <= selected_index < len(filtered_models):
                        model_name = filtered_models[selected_index]
                    else:
                        console.print("[bold red]Invalid selection. Please try again.[/bold red]")
                        continue
            except (ValueError, IndexError):
                console.print("[bold red]Invalid input. Please enter a number from the list.[/bold red]")
                continue
            except (EOFError, KeyboardInterrupt):
                break
        
        if not model_name:
            break

        try:
            search_query = input("Enter the topic of markets to bet on (or type 'exit' to quit): ")
        except (EOFError, KeyboardInterrupt):
            break
        if search_query.lower() == 'exit' or exit_flag:
            break
        main_modular_autobet(search_query, model_name)

    console.print("[bold green]Script finished.[/bold green]")''

