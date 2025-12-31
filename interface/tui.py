from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table
import time

class NanoTUI:
    def __init__(self):
        self.console = Console()
        self.layout = Layout()

        # Split: Main Chat (Left/Top) and Telemetry (Right/Bottom)
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="input", size=3)
        )
        self.layout["body"].split_row(
            Layout(name="chat", ratio=3),
            Layout(name="stats", ratio=1)
        )

        # State
        self.history = Text()
        self.stats = {
            "Status": "IDLE",
            "Speed": "0.00 T/s",
            "Context": "0 / 2048",
            "Cache Usage": "0 MB"
        }

    def render_layout(self):
        # 1. Header
        header = Panel(
            Text("NANOINFER-CPU // v0.1 // QUANTIZED INT8", justify="center", style="bold green"),
            style="green"
        )

        # 2. Chat Area
        chat_panel = Panel(
            self.history,
            title="Chat Buffer",
            border_style="blue",
            padding=(1, 1)
        )

        # 3. Stats Area
        stat_table = Table.grid(padding=1)
        stat_table.add_column(style="bold yellow")
        stat_table.add_column()

        for k, v in self.stats.items():
            stat_table.add_row(f"{k}:", v)

        stats_panel = Panel(
            stat_table,
            title="Telemetry",
            border_style="yellow"
        )

        # 4. Input Placeholder
        input_panel = Panel(
            Text("> _", style="blink white"),
            title="Input",
            border_style="white"
        )

        self.layout["header"].update(header)
        self.layout["chat"].update(chat_panel)
        self.layout["stats"].update(stats_panel)
        self.layout["input"].update(input_panel)

        return self.layout

    def stream_token(self, token_str):
        """Appends a new token to the chat window live."""
        self.history.append(token_str)

    def update_stats(self, tps, context_len):
        self.stats["Speed"] = f"{tps:.2f} T/s"
        self.stats["Context"] = f"{context_len} / 2048"
        self.stats["Status"] = "GENERATING"

    def reset_input(self):
        self.stats["Status"] = "IDLE"
