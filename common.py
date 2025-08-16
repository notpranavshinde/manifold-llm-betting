import json
import os
import time
from datetime import datetime, timedelta
import keyboard
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import argparse
import csv

# --- Environment Variable Loading ---
MANIFOLD_API_KEY = os.environ.get("MANIFOLD_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- Console ---
console = Console()

# --- Graceful Exit ---
exit_flag = False
def graceful_exit_listener():
    global exit_flag
    keyboard.wait('q')
    exit_flag = True
    console.print("\n[bold yellow]Exit signal received. Terminating after the current market analysis...[/bold yellow]")

# --- Manifold API Functions ---
def get_headers(api_key):
    return {'Authorization': f'Key {api_key}', 'Content-Type': 'application/json'}

def get_user_details():
    api_url = "https://api.manifold.markets/v0/me"
    try:
        response = requests.get(api_url, headers=get_headers(MANIFOLD_API_KEY))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error fetching user details:[/bold red] {e}")
        return None

def search_manifold_markets(search_term, limit):
    api_url = "https://api.manifold.markets/v0/search-markets"
    params = {'term': search_term, 'limit': limit}
    try:
        response = requests.get(api_url, params=params, headers=get_headers(MANIFOLD_API_KEY))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error fetching data from Manifold API:[/bold red] {e}")
        return None
    except json.JSONDecodeError:
        console.print("[bold red]Error: Failed to decode JSON response.[/bold red]")
        return None

def get_market_by_slug(slug):
    api_url = f"https://api.manifold.markets/v0/slug/{slug}"
    try:
        response = requests.get(api_url, headers=get_headers(MANIFOLD_API_KEY))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None
    except json.JSONDecodeError:
        return None

def place_bet(market_id, amount, outcome, full_market, model_prob, model_confidence, model_name, dry_run):
    global exit_flag
    if exit_flag:
        return False, 0
    
    if dry_run:
        console.print(f"\n[bold yellow]DRY RUN:[/bold yellow] Would have placed a bet of M${amount:.2f} on '{outcome}' for market {market_id}.")
        log_bet(market_id, full_market.get('question'), outcome, amount, model_name, model_prob, model_confidence, full_market.get('probability'), dry_run)
        return True, amount

    api_url = "https://api.manifold.markets/v0/bet"
    payload = {"amount": amount, "contractId": market_id, "outcome": outcome}
    console.print(f"\n[bold green]BETTING:[/bold green] Placing M${amount:.2f} on '{outcome}' for market {market_id}...")
    try:
        response = requests.post(api_url, headers=get_headers(MANIFOLD_API_KEY), json=payload)
        response.raise_for_status()
        console.print("[bold green]✔ BET PLACED SUCCESSFULLY.[/bold green]")
        log_bet(market_id, full_market.get('question'), outcome, amount, model_name, model_prob, model_confidence, full_market.get('probability'), dry_run)
        return True, amount
    except requests.exceptions.RequestException as e:
        error_message = e.response.json().get('message', str(e)) if e.response else str(e)
        if e.response and e.response.status_code == 403:
            error_message += "\n[bold yellow]This is a 403 Forbidden error. Please ensure your MANIFOLD_API_KEY has 'trade' permissions.[/bold yellow]"
        console.print(f"[bold red]✖ FAILED TO PLACE BET:[/bold red] {error_message}")
        return False, 0


# --- Logging ---
def log_bet(market_id, question, outcome, amount, model_name, model_prob, model_confidence, market_prob, dry_run):
    log_file = "bet_log.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'market_id', 'question', 'outcome', 'amount', 'model_name', 'model_prob', 'model_confidence', 'market_prob', 'edge', 'dry_run']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market_id': market_id,
            'question': question,
            'outcome': outcome,
            'amount': amount,
            'model_name': model_name,
            'model_prob': model_prob,
            'model_confidence': model_confidence,
            'market_prob': market_prob,
            'edge': model_prob - market_prob if outcome == "YES" else (1 - model_prob) - (1 - market_prob),
            'dry_run': dry_run
        })

# --- Formatting and Parsing ---
def parse_description(description):
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
    if not ts:
        return "N/A"
    try:
        return datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
    except (OSError, ValueError):
        return "Date out of range"

def build_market_panel(full_market, prob_str, reason_str, model_name):
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
    table.add_row(f"{model_name} Prob:", prob_str)
    table.add_row(f"{model_name} Reasoning:", Text(reason_str, style="italic"))
    return Panel(table, border_style="blue", expand=False, title=f"Market Details: {full_market.get('slug')}", title_align="left")

def parse_args():
    parser = argparse.ArgumentParser(description="Manifold LLM Autobet")
    parser.add_argument("--kelly-fraction", type=float, default=0.25, help="The fraction of the Kelly criterion to use for betting.")
    parser.add_argument("--resolution-months-limit", type=int, default=1, help="The maximum number of months in the future to consider for markets.")
    parser.add_argument("--min-confidence", type=str, default="Medium", choices=["Low", "Medium", "High"], help="The minimum confidence level required to place a bet.")
    parser.add_argument("--dry-run", action="store_true", help="If set, the script will not place any bets.")
    return parser.parse_args()
