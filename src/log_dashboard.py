import sqlite3
import time
from rich.console import Console
from rich.table import Table
from rich.live import Live

DB_PATH = "ameva_universal_logs.db"

def get_latest_logs(limit=25):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM universal_logs ORDER BY log_id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return list(reversed(rows)) # Return oldest first for the tailing effect
    except sqlite3.OperationalError:
        # DB might not exist yet
        return []

def generate_table(logs):
    table = Table(show_header=True, header_style="bold magenta", expand=True, title="🚀 AMEVA 1MB-class Log Dashboard (Live Tail)")
    table.add_column("ID", style="dim", width=10)
    table.add_column("Time", width=25)
    table.add_column("Source", style="cyan", width=20)
    table.add_column("Level", width=12)
    table.add_column("Payload")

    for row in logs:
        level_style = "green"
        if row["level"] == "ERROR":
            level_style = "bold red"
        elif row["level"] == "WARN":
            level_style = "bold yellow"
        elif row["level"] == "BOTTLENECK":
            level_style = "bold magenta"
            
        payload = row["payload_json"]
        if len(payload) > 80:
            payload = payload[:77] + "..."
            
        table.add_row(
            str(row["log_id"]),
            row["timestamp"],
            row["source"],
            f"[{level_style}]{row['level']}[/{level_style}]",
            payload
        )
    return table

if __name__ == "__main__":
    console = Console()
    console.print("[dim]Connecting to stream...[/dim]")
    
    last_id = -1
    
    with Live(generate_table([]), refresh_per_second=4) as live:
        while True:
            logs = get_latest_logs(25)
            if logs and logs[-1]["log_id"] != last_id:
                last_id = logs[-1]["log_id"]
                live.update(generate_table(logs))
            time.sleep(0.25)
