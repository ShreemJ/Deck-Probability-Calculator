from config import *
from shutil import get_terminal_size
from collections import defaultdict

def clear_screen():
    console.clear()
    columns, _ = get_terminal_size(fallback=(80, 24))
    if columns >= 205:
        console.print(ASCII_ART, justify="center")
    elif columns >= 135:
        console.print(ASCII_ART_SMALL, justify="center")
    else:
        console.print(ASCII_ART_EXTRA_SMALL, justify="center")
    

def pause():
    console.print("[prompt]Press Enter to continue...[/prompt]", end="")
    console.input()

def parse_expression(tokens):
    precedence = {'OR': 1, 'XOR': 2, 'AND': 3}
    output, stack = [], []

    for token in tokens:
        if token == '(':
            stack.append(token)
        elif token == ')':
            while stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        elif token in precedence:
            while stack and stack[-1] in precedence and precedence[stack[-1]] >= precedence[token]:
                output.append(stack.pop())
            stack.append(token)
        else:
            output.append(token)
    output += reversed(stack)

    stack.clear()
    for token in output:
        if token in precedence:
            if len(stack) < 2:
                return console.print(f"[error]Invalid expression: {tokens}[/error]") or None
            b, a = stack.pop(), stack.pop()
            stack.append((token, a, b))
        else:
            stack.append(token)

    if len(stack) != 1:
        return console.print(f"[error]Malformed expression: {tokens}[/error]") or None
    return stack[0]


def update_card_type_cache():
    state["card_type_cache"].clear()
    card_to_types = defaultdict(set)
    for card_type, cards in state["card_types"].items():
        for card in cards:
            card_to_types[card].add(card_type)
    for card_type in state["card_types"]:
        state["card_type_cache"][card_type] = sum(state["cards"].get(card, 0) for card in state["card_types"][card_type])

def get_int_input(prompt, minimum=0, default=None):
    while True:
        val = console.input(f"[prompt]{prompt} > [/prompt]").strip()
        if val == "" and default is not None:
            return default
        try:
            num = int(val)
            if num >= minimum:
                return num
            console.print(f"[error]Value must be at least {minimum}.[/error]")
        except ValueError:
            console.print("[error]Please enter a valid integer.[/error]")

def get_multiple_ints(prompt, minimum=1):
    while True:
        vals = console.input(f"[prompt]{prompt} > [/prompt]").strip()
        try:
            vals = [int(v) for v in vals.replace(',', ' ').split() if v]
            if all(v >= minimum for v in vals):
                return vals
            console.print(f"[error]All values must be at least {minimum}.[/error]")
        except ValueError:
            console.print("[error]Please enter valid integers (space or comma-separated).[/error]")
