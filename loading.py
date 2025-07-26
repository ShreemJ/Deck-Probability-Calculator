from base64 import b64decode
from requests import get
from collections import Counter
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from os import listdir
from os.path import isfile, join
from json import load, dump

from config import console, state
from utility import *

def import_ydk_file(path):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        ids = []
        in_main = False
        for line in lines:
            line = line.strip()
            match line:
                case "#main":
                    in_main = True
                    continue
                case _ if line.startswith("#"):
                    in_main = False
                case _ if in_main and line.isdigit():
                    ids.append(int(line))
        console.print(f"[info]Importing {len(ids)} cards...[/info]")

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), transient=True) as progress:
            task = progress.add_task("Fetching card names...", total=len(ids))
            result = fetch_card_names(ids)

        state["cards"] = result
        state["deck_size"] = sum(result.values())
        update_card_type_cache()
        state["hand_size"] = state.get("hand_size", 5)
        console.print(f"[success]Deck size set to {state['deck_size']}[/success]")
    except Exception as e:
        console.print(f"[error]Error: {e}[/error]")
        pause()
        import_deck_prompt()
    pause()

def import_ydke_link():
    link = console.input("[prompt]Paste ydke:// link: [/prompt]")
    if not link.startswith("ydke://"):
        console.print("[error]Invalid YDKE link format.[/error]")
        pause()
        import_deck_prompt()
        return
    try:
        encoded_parts = link[7:].split("!")
        ids = []
        if encoded_parts and encoded_parts[0]:
            decoded = b64decode(encoded_parts[0])
            ids = [int.from_bytes(decoded[i:i+4], "little") for i in range(0, len(decoded), 4)]
        state["cards"] = fetch_card_names(ids)
        state["deck_size"] = sum(state["cards"].values())
        update_card_type_cache()
        state["hand_size"] = 5
        console.print(f"[success]Imported {len(ids)} main deck cards from YDKE link.[/success]")
        console.print(f"[success]Deck size set to {state['deck_size']}.[/success]")
    except Exception as e:
        console.print(f"[error]Failed to decode YDKE link: {e}[/error]")
        pause()
        import_deck_prompt()
    pause()

def fetch_card_names(ids):
    counts = Counter(ids)
    result = {}

    with Progress() as progress:
        task = progress.add_task("Fetching cards...", total=len(counts))
        
        for card_id, count in counts.items():
            try:
                r = get(f"https://db.ygoprodeck.com/api/v7/cardinfo.php?id={card_id}")
                data = r.json()
                name = data['data'][0]['name']
            except Exception:
                name = f"Unknown Card ({card_id})"
            result[name] = count
            
            progress.update(task, advance=1)

    return result

def import_deck_ygo():
    clear_screen()
    console.print("[header]Yu-Gi-Oh Deck Import[/header]")
    console.print("[info]1. Import from .ydk file[/info]")
    console.print("[info]2. Import from ydke:// link[/info]")
    choice = console.input("[prompt]> [/prompt]")
    match choice:
        case "1":
            file_path = console.input("[prompt]Input the file location of the YDK file: [/prompt]")
            import_ydk_file(file_path)
        case "2":
            import_ydke_link()
        case _:
            console.print("[error]Invalid option.[/error]")
            pause()
            import_deck_prompt()

def import_deck_prompt(from_first_screen=True):
    clear_screen()
    console.print("[header]Deck Import[/header]")
    console.print("[info]1. Import Yu-Gi-Oh Deck[/info]")
    console.print("[info]2. Import .json file[/info]")
    console.print("[info]3. Continue[/info]")
    choice = console.input("[prompt]> [/prompt]")
    match choice:
        case "1":
            import_deck_ygo()
        case "2":
            page_load(from_first_screen=from_first_screen)
        case "3" | "":
            return
        case _:
            console.print("[error]Invalid option.[/error]")
            pause()
            import_deck_prompt()

def page_load(from_first_screen=False):
    clear_screen()
    if not from_first_screen:
        console.print("[header][9] Load Deck State[/header]")
    else:
        console.print("[header]Load Deck State[/header]")
    files = []
    files += [f for f in listdir() if f.endswith(".json") and isfile(f)]
    subdirs = [d for d in listdir() if not isfile(d)]
    for subdir in subdirs:
        sub_files = [
            join(subdir, f)
            for f in listdir(subdir)
            if f.endswith(".json") and isfile(join(subdir, f))
        ]
        files.extend(sub_files)

    if files:
        console.print("[info]Available .json files:[/info]")
        for i, fname in enumerate(files):
            console.print(f"[option]{i + 1}[/option]: {fname}")
    else:
        console.print("[info]No .json files found in current directory or subfolders.[/info]")

    console.print("[prompt]Enter the number of the file to load, or enter a path to a json:[/prompt]")
    while True:
        user_input = console.input("[prompt]> [/prompt]").strip()
        if user_input == "":
            return
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(files):
                path = files[idx]
                break
            else:
                console.print("[error]Invalid selection number.[/error]")
        else:
            path = user_input
            if isfile(path) and path.endswith(".json"):
                break
            else:
                console.print("[error]File not found or not a .json file. Try again.[/error]")
    try:
        with open(path, "r") as f:
            loaded = load(f)
        required = {"deck_size", "hand_size", "cards", "card_types", "combos", "combo_sets"}
        if not required.issubset(loaded.keys()):
            console.print("[error]File missing required fields.[/error]")
        else:
            state.clear()
            state.update(loaded)
            update_card_type_cache()
            console.print(f"[success]State loaded from {path} successfully.[/success]")
    except Exception as e:
        console.print(f"[error]Error loading file: {e}[/error]")
        if from_first_screen:
            pause()
            import_deck_prompt()
    pause()

def page_save():
    clear_screen()
    console.print("[header][8] Save Deck State[/header]")
    path = console.input("[prompt]Enter filename to save (e.g., deck.json): [/prompt]")
    try:
        with open(path, "w") as f:
            dump(state, f, indent=4)
        console.print(f"[success]State saved to {path}.[/success]")
    except Exception as e:
        console.print(f"[error]Error saving file: {e}[/error]")
    pause()
