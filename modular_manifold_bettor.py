import json
import re
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
    import keyboard
except ImportError:
    print("This script requires several libraries.")
    print("Please install them using: pip install requests rich keyboard")
    exit()

# --- Unified Configuration ---
# Set your API keys as environment variables.
# PowerShell example:
#   $env:MANIFOLD_API_KEY="your_key_here"
#   $env:OPENROUTER_API_KEY="your_key_here"
MANIFOLD_API_KEY = os.environ.get("MANIFOLD_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# --- Model Configuration ---
# Pick a model you actually have access to on OpenRouter.
# See https://openrouter.ai/models for valid IDs tied to your key.
MODEL_NAME = "google/gemini-2.5-pro:online"

# --- Betting Configuration ---
MARKET_LIMIT = 50
KELLY_FRACTION = 0.25
RESOLUTION_MONTHS_LIMIT = 1
MINIMUM_CONFIDENCE_TO_BET = ["Medium", "High"]
MIN_EDGE = 0.01  # minimum edge to bet

# --- API Configuration ---
console = Console()
if not MANIFOLD_API_KEY or not OPENROUTER_API_KEY:
    console.print("[bold red]Error: MANIFOLD_API_KEY and OPENROUTER_API_KEY environment variables must be set.[/bold red]")
    exit()

HTTP_TIMEOUT = 60  # seconds

# --- Graceful Exit ---
exit_flag = False
def graceful_exit_listener():
    global exit_flag
    keyboard.wait('q')
    exit_flag = True
    console.print("\n[bold yellow]Exit signal received. Terminating after the current market analysis...[/bold yellow]")

# --- Helpers ---

def get_headers(api_key=MANIFOLD_API_KEY):
    """Headers for Manifold requests."""
    return {
        'Authorization': f'Key {api_key}',
        'Content-Type': 'application/json'
    }

def get_user_details():
    """Fetch Manifold user details."""
    api_url = "https://api.manifold.markets/v0/me"
    try:
        response = requests.get(api_url, headers=get_headers(), timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error fetching user details:[/bold red] {format_request_error(e)}")
        return None

def search_manifold_markets(search_term, limit):
    """Search Manifold markets."""
    api_url = "https://api.manifold.markets/v0/search-markets"
    params = {'term': search_term, 'limit': limit}
    try:
        response = requests.get(api_url, params=params, headers=get_headers(), timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error fetching data from Manifold API:[/bold red] {format_request_error(e)}")
        return None
    except json.JSONDecodeError:
        console.print("[bold red]Error: Failed to decode JSON response.[/bold red]")
        return None

def get_market_by_slug(slug):
    """Fetch a single market by slug."""
    api_url = f"https://api.manifold.markets/v0/slug/{slug}"
    try:
        response = requests.get(api_url, headers=get_headers(), timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None
    except json.JSONDecodeError:
        return None

def place_bet(market_id, amount, outcome):
    """Place a bet on Manifold."""
    global exit_flag
    if exit_flag:
        return False, 0

    api_url = "https://api.manifold.markets/v0/bet"

    # Manifold accepts numbers, but integers are safest for "mana" amounts.
    bet_amount_int = max(1, int(round(amount)))

    payload = {
        "amount": bet_amount_int,
        "contractId": market_id,
        "outcome": outcome,
        # Optional: set a limit probability to avoid bad slippage, example:
        # "limitProb": 0.70 if outcome == "YES" else 0.30,
    }

    console.print(f"\n[bold green]BETTING:[/bold green] Placing M${bet_amount_int} on '{outcome}' for market {market_id}...")
    try:
        response = requests.post(api_url, headers=get_headers(MANIFOLD_API_KEY), json=payload, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        console.print("[bold green]✔ BET PLACED SUCCESSFULLY.[/bold green]")
        return True, bet_amount_int
    except requests.exceptions.RequestException as e:
        error_message = format_request_error(e)
        # Hint for common permission issue
        if getattr(e, "response", None) is not None and e.response.status_code == 403:
            error_message += "\n[bold yellow]403 Forbidden: ensure your MANIFOLD_API_KEY has 'trade' permissions.[/bold yellow]"
        console.print(f"[bold red]✖ FAILED TO PLACE BET:[/bold red] {error_message}")
        return False, 0

def parse_description(description):
    """Extract plain text from the Manifold 'description' field."""
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
    """Format ms timestamp to readable string, with range guard."""
    if not ts:
        return "N/A"
    try:
        return datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
    except (OSError, ValueError):
        return "Date out of range"

def _build_market_panel(full_market, model_prob_str, model_reason_str):
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
        try:
            table.add_row("Market Probability:", f"[bold magenta]{float(full_market.get('probability', 0)):.2%}[/bold magenta]")
        except Exception:
            table.add_row("Market Probability:", "N/A")
    table.add_row("Resolution Criteria:", Text(parse_description(full_market.get('description')), style="italic dim"))
    table.add_row("---", "---")
    table.add_row("Model Prob:", model_prob_str)
    table.add_row("Model Reasoning:", Text(model_reason_str, style="italic"))
    return Panel(table, border_style="blue", expand=False, title=f"Market Details: {full_market.get('slug')}", title_align="left")

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

def get_model_analysis(full_market):
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
        live.update(_build_market_panel(full_market, "Querying model...", ""))

        # OpenRouter call (fixed)
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            # Optional but recommended by OpenRouter to identify your app:
            "HTTP-Referer": "http://localhost",
            "X-Title": "Manifold AutoBet",
        }

        payload = {
            "model": MODEL_NAME,
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
            live.update(_build_market_panel(full_market, "[red]Error[/red]", f"API Error: {error_message}"))
            return None, None

        if exit_flag:
            return None, None

        try:
            final_reasoning, model_prob, model_confidence = parse_model_output_to_prob_conf(full_response_text)
            final_prob_str = f"[bold green]{model_prob:.2%}[/bold green] (Confidence: {model_confidence})"
        except Exception as e:
            final_prob_str = "[red]Error[/red]"
            final_reasoning = f"[red]Failed to parse model output.[/red] ({e})"

        live.update(_build_market_panel(full_market, final_prob_str, final_reasoning))

    return model_prob, model_confidence

def main_modular_autobet(search_query):
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
    stats_table.add_row("Model in Use:", f"[bold cyan]{MODEL_NAME}[/bold cyan]")
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
        model_prob, model_confidence = get_model_analysis(full_market)

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

                    bet_placed, amount_bet = place_bet(full_market['id'], bet_amount, outcome)
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

if __name__ == "__main__":
    exit_thread = threading.Thread(target=graceful_exit_listener, daemon=True)
    exit_thread.start()

    while not exit_flag:
        console.print(Panel("Welcome to the Manifold + Model AUTOBET Script!", title="Main Menu", border_style="green"))
        console.print("Press 'q' at any time to gracefully exit after the current market analysis.")
        try:
            search_query = input("Enter the topic of markets to bet on (or type 'exit' to quit): ")
        except (EOFError, KeyboardInterrupt):
            break
        if search_query.lower() == 'exit' or exit_flag:
            break
        main_modular_autobet(search_query)

    console.print("[bold green]Script finished.[/bold green]")

