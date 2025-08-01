from rich.theme import Theme
from rich.console import Console

custom_theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "bold red",
    "header": "bold magenta",
    "prompt": "bold blue",
    "combo": "bold green",
    "highlight": "bold white on dark_green",
    "text": "bold yellow"
})

console = Console(theme=custom_theme)

ASCII_ART = """[highlight]
 ██████╗ ███████╗ ██████╗██╗  ██╗    ██████╗ ██████╗  ██████╗ ██████╗  █████╗ ██████╗ ██╗██╗     ██╗████████╗██╗   ██╗     ██████╗ █████╗ ██╗      ██████╗██╗   ██╗██╗      █████╗ ████████╗ ██████╗ ██████╗  
  ██╔══██╗██╔════╝██╔════╝██║ ██╔╝    ██╔══██╗██╔══██╗██╔═══██╗██╔══██╗██╔══██╗██╔══██╗██║██║     ██║╚══██╔══╝╚██╗ ██╔╝    ██╔════╝██╔══██╗██║     ██╔════╝██║   ██║██║     ██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
  ██║  ██║█████╗  ██║     █████╔╝     ██████╔╝██████╔╝██║   ██║██████╔╝███████║██████╔╝██║██║     ██║   ██║    ╚████╔╝     ██║     ███████║██║     ██║     ██║   ██║██║     ███████║   ██║   ██║   ██║██████╔╝
  ██║  ██║██╔══╝  ██║     ██╔═██╗     ██╔═══╝ ██╔══██╗██║   ██║██╔══██╗██╔══██║██╔══██╗██║██║     ██║   ██║     ╚██╔╝      ██║     ██╔══██║██║     ██║     ██║   ██║██║     ██╔══██║   ██║   ██║   ██║██╔══██╗
  ██████╔╝███████╗╚██████╗██║  ██╗    ██║     ██║  ██║╚██████╔╝██████╔╝██║  ██║██████╔╝██║███████╗██║   ██║      ██║       ╚██████╗██║  ██║███████╗╚██████╗╚██████╔╝███████╗██║  ██║   ██║   ╚██████╔╝██║  ██║
  ╚═════╝ ╚══════╝ ╚═════╝╚═╝  ╚═╝    ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝╚══════╝╚═╝   ╚═╝      ╚═╝        ╚═════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
                                                                                                                                                                                                                                                                                                                                                                                                                                           
[/highlight]"""

ASCII_ART_SMALL = """[highlight]
▗▄▄▄ ▗▄▄▄▖ ▗▄▄▖▗▖ ▗▖    ▗▄▄▖ ▗▄▄▖  ▗▄▖ ▗▄▄▖  ▗▄▖ ▗▄▄▖ ▗▄▄▄▖▗▖   ▗▄▄▄▖▗▄▄▄▖▗▖  ▗▖     ▗▄▄▖ ▗▄▖ ▗▖    ▗▄▄▖▗▖ ▗▖▗▖    ▗▄▖▗▄▄▄▖▗▄▖ ▗▄▄▖ 
▐▌  █▐▌   ▐▌   ▐▌▗▞▘    ▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌  █  ▐▌     █    █   ▝▚▞▘     ▐▌   ▐▌ ▐▌▐▌   ▐▌   ▐▌ ▐▌▐▌   ▐▌ ▐▌ █ ▐▌ ▐▌▐▌ ▐▌
▐▌  █▐▛▀▀▘▐▌   ▐▛▚▖     ▐▛▀▘ ▐▛▀▚▖▐▌ ▐▌▐▛▀▚▖▐▛▀▜▌▐▛▀▚▖  █  ▐▌     █    █    ▐▌      ▐▌   ▐▛▀▜▌▐▌   ▐▌   ▐▌ ▐▌▐▌   ▐▛▀▜▌ █ ▐▌ ▐▌▐▛▀▚▖
▐▙▄▄▀▐▙▄▄▖▝▚▄▄▖▐▌ ▐▌    ▐▌   ▐▌ ▐▌▝▚▄▞▘▐▙▄▞▘▐▌ ▐▌▐▙▄▞▘▗▄█▄▖▐▙▄▄▖▗▄█▄▖  █    ▐▌      ▝▚▄▄▖▐▌ ▐▌▐▙▄▄▖▝▚▄▄▖▝▚▄▞▘▐▙▄▄▖▐▌ ▐▌ █ ▝▚▄▞▘▐▌ ▐▌
                
[/highlight]                                                                                                                           
"""

ASCII_ART_EXTRA_SMALL = """[highlight]
 +-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+
 |D|E|C|K| |P|r|o|b|a|b|i|l|i|t|y| |C|a|l|c|u|l|a|t|o|r|
 +-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+
[/highlight]                                                                                                                           
"""

state = {
    "deck_size": None,
    "hand_size": None,
    "cards": {},
    "card_types": {},
    "combos": {},
    "combo_sets": {},
    "card_type_cache": {},
}

y_axis_options = {
    "1": "P(X = k)",
    "2": "P(X ≠ k)",
    "3": "P(X ≤ k)",
    "4": "P(X ≥ k)",
    "5": "P(k_1 ≤ X ≤ k_2)",
    "6": "Expected Number of Cards"
}

mulligan_types = ["Traditional", "No Penalty", "London", "Partial",  "Partial No Penalty"]

mulligan_descriptions = [
    "Redraw hand with 1 card less",
    "Redraw hand without any penalty",
    "Redraw hand, then place back cards eqaual to the amount of times mulliganed",
    "Choose certian cards in hand to shuffle back and redraw with 1 card less",
    "Choose certian cards in hand to shuffle back and redraw with no penalty",
]
