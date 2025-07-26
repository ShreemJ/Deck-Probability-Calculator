from json import load,dump
from base64 import b64decode
from requests import get
from collections import Counter, defaultdict, deque
from math import comb
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.theme import Theme
from os import listdir
from os.path import isfile
from sys import exit
from shutil import get_terminal_size
from pyautogui import size
import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Set, Tuple, Any, Union, Optional

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

def hand_satisfies_combo(
    hand: List[str],
    constraints_and_settings: Dict[str, Union[Tuple[int, int], Any]],
    card_types: Dict[str, List[str]]
) -> bool:
    constraints = {
        k: v if isinstance(v, tuple) else tuple(v)
        for k, v in constraints_and_settings.items()
        if isinstance(v, (list, tuple)) and len(v) == 2
    }

    roles: List[str] = []
    for T, (min_required, _) in constraints.items():
        roles += [T] * min_required

    cards = list(hand)

    card_to_types: Dict[str, set] = defaultdict(set)
    for T, card_list in card_types.items():
        for c in card_list:
            card_to_types[c].add(T)
    for card in hand:
        card_to_types[card].add(card)

    n_roles = len(roles)
    n_cards = len(cards)

    adj: Dict[int, List[int]] = defaultdict(list)
    for role_idx, role in enumerate(roles):
        for card_idx, card in enumerate(cards):
            if role in card_to_types[card]:
                adj[role_idx].append(card_idx)

    pair_U: Dict[int, Optional[int]] = {u: None for u in range(n_roles)}
    pair_V: Dict[int, Optional[int]] = {v: None for v in range(n_cards)}
    dist: Dict[int, int] = {}

    def bfs() -> bool:
        queue = deque()
        for u in range(n_roles):
            if pair_U[u] is None:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = float('inf')
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                matched_u = pair_V[v]
                if matched_u is None:
                    found = True
                elif dist[matched_u] == float('inf'):
                    dist[matched_u] = dist[u] + 1
                    queue.append(matched_u)
        return found

    def dfs(u: int) -> bool:
        for v in adj[u]:
            matched_u = pair_V[v]
            if matched_u is None or (dist[matched_u] == dist[u] + 1 and dfs(matched_u)):
                pair_U[u] = v
                pair_V[v] = u
                return True
        dist[u] = float('inf')
        return False

    matching_size = 0
    while bfs():
        for u in range(n_roles):
            if pair_U[u] is None and dfs(u):
                matching_size += 1

    if matching_size < n_roles:
        return False

    hand_counts = Counter(hand)
    for T, (min_required, max_allowed) in constraints.items():
        cards_of_type = card_types.get(T, [T])
        count = sum(hand_counts[c] for c in cards_of_type)
        if not (min_required <= count <= max_allowed):
            return False

    return True


def evaluate_combo_expression(hand: List[str], combo_expr: Union[str, Tuple[str, Any, Any]], combos: Dict[str, Dict[str, Any]], card_types: Dict[str, List[str]]) -> bool:
    if isinstance(combo_expr, str):
        if combo_expr in combos:
            return hand_satisfies_combo(hand, combos[combo_expr], card_types)
        else:
            return hand.count(combo_expr) >= 1
    op: str = combo_expr[0]
    args: Tuple[Any, ...] = combo_expr[1:]

    if op == "AND":
        return all(evaluate_combo_expression(hand, arg, combos, card_types) for arg in args)
    elif op == "OR":
        return any(evaluate_combo_expression(hand, arg, combos, card_types) for arg in args)
    elif op == "XOR":
        return sum(evaluate_combo_expression(hand, arg, combos, card_types) for arg in args) == 1
    else:
        raise ValueError(f"Unknown logical operator: {op}")

def combo_type_probability(deck_size: int, hand_size: int, card_count: Dict[str, int], card_types: Dict[str, List[str]], combos: Dict[str, Dict[str, Any]], debug: bool = False, extra_info: bool = False) -> Dict[str, Union[float, Dict[str, float]]]:
    card_counts: Dict[str, int] = card_count
    total_card_count: int = sum(card_counts.values())
    if total_card_count < deck_size:
        missing: int = deck_size - total_card_count
        card_counts = dict(card_counts)
        card_counts["Unspecified"] = missing
        if debug:
            console.print(f"[info]Added {missing} Unspecified cards to reach deck size {deck_size}[/info]")
    elif total_card_count > deck_size:
        raise ValueError(f"Sum of card counts ({total_card_count}) exceeds deck size ({deck_size})")
    results: Dict[str, Union[float, Dict[str, float]]] = {}

    def generate_counts(cards: List[str], hand_size: int, current_counts: Dict[str, int], index: int, constraints: Dict[str, Union[Tuple[int, int], Any]], result: List[float], card_counts: Dict[str, int]) -> None:
        if index == len(cards):
            if sum(current_counts.values()) == hand_size:
                hand: List[str] = []
                for card, count in current_counts.items():
                    hand.extend([card] * count)
                if not hand_satisfies_combo(hand, constraints, card_types):
                    return
                prob: float = 1.0
                for card, count in current_counts.items():
                    prob *= comb(card_counts[card], count)
                result[0] += prob
            return

        card: str = cards[index]
        max_count: int = min(card_counts[card], hand_size - sum(current_counts.values()))
        for count in range(max_count + 1):
            current_counts[card] = count
            generate_counts(cards, hand_size, current_counts, index + 1, constraints, result, card_counts)
            current_counts[card] = 0

    def generate_counts_with_replacements(cards: List[str], hand_size: int, current_counts: Dict[str, int], index: int, constraints: Dict[str, Union[Tuple[int, int], Any]], replacements: int, deck_counts: Dict[str, int], result: List[float]) -> None:
        if index == len(cards):
            if sum(current_counts.values()) == hand_size:
                hand: List[str] = []
                for card, count in current_counts.items():
                    hand.extend([card] * count)
                if hand_satisfies_combo(hand, constraints, card_types):
                    prob: float = 1.0
                    for card, count in current_counts.items():
                        prob *= comb(deck_counts[card], count)
                    result[0] += prob
            return

        card: str = cards[index]
        max_count: int = min(deck_counts[card], hand_size - sum(current_counts.values()))
        for count in range(max_count + 1):
            current_counts[card] = count
            generate_counts_with_replacements(cards, hand_size, current_counts, index + 1, constraints, replacements, deck_counts, result)
            current_counts[card] = 0

    def prob_with_mulligans(constraints: Dict[str, Any], mulligans: bool, mulligan_count: Optional[int], free_first_mulligan: bool, h_size: int, mulligan_type: str) -> Union[float, Dict[str, float]]:
        max_mulls: int = mulligan_count if mulligans and mulligan_count is not None else h_size - 1
        if max_mulls + (1 if free_first_mulligan else 0) <= 0:
            max_mulls = 0
        if mulligan_type == "traditional":
            attempts: List[int] = [h_size]
            if free_first_mulligan:
                attempts.append(h_size)
            for i in range(max_mulls):
                next_size: int = h_size - (i + 1)
                if next_size <= 0:
                    break
                attempts.append(next_size)

            p_hit: List[float] = []
            for h in attempts:
                result: List[float] = [0.0]
                generate_counts(list(all_cards_this_combo.keys()), h, defaultdict(int), 0, constraints, result, all_cards_this_combo)
                p_hit.append(result[0] / comb(deck_size, h))

        elif mulligan_type == "london":
            attempts: List[Tuple[int, int]] = []
            if free_first_mulligan:
                attempts.append((h_size, 0))  
                attempts.append((h_size, 1)) 
            else:
                attempts.append((h_size, 0))
            for i in range(max_mulls):
                cards_to_return: int = i + (1 if not free_first_mulligan else 2)
                final_hand_size: int = h_size - cards_to_return
                if final_hand_size <= 0:
                    break
                attempts.append((h_size, cards_to_return))

            p_hit: List[float] = []
            for h, cards_to_return in attempts:
                final_hand_size: int = h - cards_to_return
                if final_hand_size <= 0:
                    break
                result: List[float] = [0.0]
                generate_counts(list(all_cards_this_combo.keys()), h, defaultdict(int), 0, constraints, result, all_cards_this_combo)
                prob: float = result[0] / comb(deck_size, h)
                p_hit.append(prob)

        elif mulligan_type == "partial":
            p_hit: List[float] = []
            result: List[float] = [0.0]
            generate_counts(list(all_cards_this_combo.keys()), h_size, defaultdict(int), 0, constraints, result, all_cards_this_combo)
            p_initial: float = result[0] / comb(deck_size, h_size)
            p_hit.append(p_initial)
            min_required: int = sum(min_count for card_type, (min_count, max_count) in constraints.items()
                              if card_type not in ("mulligans", "mulligan_count", "free_first_mulligan", "hand_size", "deck_size", "mulligan_type"))

            remaining_mulls: int = max_mulls + (1 if free_first_mulligan else 0)
            for mull in range(remaining_mulls):
                result = [0.0]
                initial_counts: Dict[str, int] = defaultdict(int)
                def gen_initial(idx: int, curr: Dict[str, int], hand_size: int) -> None:
                    if idx == len(all_cards_this_combo):
                        if sum(curr.values()) != hand_size:
                            return
                        hand: List[str] = []
                        for c, n in curr.items():
                            hand += [c] * n
                        if hand_satisfies_combo(hand, constraints, card_types):
                            prob: float = 1.0
                            for c, n in curr.items():
                                prob *= comb(all_cards_this_combo[c], n)
                            result[0] += prob
                            return

                        non_contributing: int = sum(n for c, n in curr.items() if c == "Unspecified" or not any(c in card_types.get(t, [t]) for t in constraints))
                        cards_to_replace: int = min(non_contributing, h_size - min_required)
                        if cards_to_replace <= 0:
                            return

                        new_deck: Dict[str, int] = dict(all_cards_this_combo)
                        for c, n in curr.items():
                            new_deck[c] -= n
                            if new_deck[c] < 0:
                                return  
                        new_deck["Unspecified"] = new_deck.get("Unspecified", 0) + cards_to_replace
                        new_result: List[float] = [0.0]
                        generate_counts_with_replacements(list(new_deck.keys()), cards_to_replace, defaultdict(int), 0, constraints, cards_to_replace, new_deck, new_result)
                        prob = 1.0
                        for c, n in curr.items():
                            prob *= comb(all_cards_this_combo[c], n)
                        result[0] += prob * (new_result[0] / comb(deck_size - hand_size, cards_to_replace))
                    c: str = list(all_cards_this_combo.keys())[idx]
                    max_n: int = min(all_cards_this_combo[c], hand_size - sum(curr.values()))
                    for take in range(max_n + 1):
                        curr[c] = take
                        gen_initial(idx + 1, curr, hand_size)
                        curr[c] = 0

                gen_initial(0, defaultdict(int), h_size)
                p_hit.append(result[0] / comb(deck_size, h_size))

        if not extra_info or (not mulligans):
            p_fail_all: float = 1.0
            for p in p_hit:
                p_fail_all *= (1 - p)
            return 1 - p_fail_all

        info: Dict[str, float] = {}
        p_fail_so_far: float = 1.0
        for i, p in enumerate(p_hit):
            attempt_name: str = "No Mulligan" if i == 0 else f"Mulligan {i}"
            p_reach_attempt: float = p_fail_so_far
            p_success_this_attempt: float = p * p_reach_attempt
            info[attempt_name] = p_success_this_attempt
            p_fail_so_far *= (1 - p)

        info["Total"] = 1 - p_fail_so_far
        return info

    for combo_name, data in combos.items():
        if "requirements" in data:
            constraints: Dict[str, Any] = data["requirements"]
        else:
            constraints = {k: v for k, v in data.items() if k not in ("mulligans", "mulligan_count", "free_first_mulligan", "hand_size", "deck_size", "mulligan_type")}

        mulligans_config: Union[bool, Dict[str, Any]] = data.get("mulligans", False)
        if isinstance(mulligans_config, dict):
            mulligans: bool = mulligans_config.get("enabled", False)
            mulligan_count: Optional[int] = mulligans_config.get("count", None)
            free_first_mulligan: bool = mulligans_config.get("first_free", False)
            mulligan_type: str = mulligans_config.get("type", "traditional")
        else:
            mulligans = mulligans_config
            mulligan_count = None
            free_first_mulligan = False
            mulligan_type = "traditional"

        h_size: int = data.get("hand_size", hand_size)
        d_size: int = data.get("deck_size", deck_size)

        combo_cards: Set[str] = set()
        for item in constraints:
            if item in card_types:
                combo_cards.update(card_types.get(item, [item]))
            else:
                combo_cards.add(item)

        all_cards_this_combo: Dict[str, int] = {}
        total_combo_cards: int = 0
        for card in combo_cards:
            count: int = card_counts.get(card, 0)
            all_cards_this_combo[card] = count
            total_combo_cards += count

        unspecified_count: int = d_size - total_combo_cards
        if unspecified_count > 0:
            all_cards_this_combo["Unspecified"] = unspecified_count

        if mulligans:
            results[combo_name] = prob_with_mulligans(constraints, mulligans, mulligan_count, free_first_mulligan, h_size, mulligan_type)
        else:
            result: List[float] = [0.0]
            generate_counts(list(all_cards_this_combo.keys()), h_size, defaultdict(int), 0, constraints, result, all_cards_this_combo)
            prob: float = result[0] / comb(d_size, h_size)
            if extra_info:
                results[combo_name] = {"No Mulligan": prob, "Total": prob}
            else:
                results[combo_name] = prob

    return results

def combo_set_probability(set_config: Dict[str, Any], card_counts: Dict[str, int], card_types: Dict[str, List[str]], combos_constraints: Dict[str, Dict[str, Any]], extra_info: bool = False) -> Union[float, Dict[str, float]]:
    d: int = set_config["deck_size"]
    h0: int = set_config["hand_size"]
    m: Dict[str, Any] = set_config["mulligans"]
    def collect_combo_cards(expression: Union[str, Tuple[str, Any, Any]]) -> Set[str]:
        combo_cards: Set[str] = set()
        if isinstance(expression, str):
            if expression in combos_constraints:
                for item in combos_constraints[expression]:
                    if item not in ("mulligans", "mulligan_count", "free_first_mulligan", "hand_size", "deck_size", "mulligan_type"):
                        if item in card_types:
                            combo_cards.update(card_types.get(item, [item]))
                        else:
                            combo_cards.add(item)
            else:
                combo_cards.add(expression)
        elif isinstance(expression, tuple):
            _, left, right = expression
            combo_cards.update(collect_combo_cards(left))
            combo_cards.update(collect_combo_cards(right))
        return combo_cards

    combo_cards: Set[str] = collect_combo_cards(set_config["expression"])
    all_cards_this_combo: Dict[str, int] = {}
    total_combo_cards: int = 0
    for card in combo_cards:
        count: int = card_counts.get(card, 0)
        all_cards_this_combo[card] = count
        total_combo_cards += count
    unspecified_count: int = d - total_combo_cards
    if unspecified_count > 0:
        all_cards_this_combo["Unspecified"] = unspecified_count

    def generate_counts_with_replacements(cards: List[str], hand_size: int, current_counts: Dict[str, int], index: int, expression: Union[str, Tuple[str, Any, Any]], replacements: int, deck_counts: Dict[str, int], result: List[float]) -> None:
        if index == len(cards):
            if sum(current_counts.values()) == hand_size:
                hand: List[str] = []
                for card, count in current_counts.items():
                    hand.extend([card] * count)
                if evaluate_combo_expression(hand, expression, combos_constraints, card_types):
                    prob: float = 1.0
                    for card, count in current_counts.items():
                        prob *= comb(deck_counts[card], count)
                    result[0] += prob
            return

        card: str = cards[index]
        max_count: int = min(deck_counts[card], hand_size - sum(current_counts.values()))
        for count in range(max_count + 1):
            current_counts[card] = count
            generate_counts_with_replacements(cards, hand_size, current_counts, index + 1, expression, replacements, deck_counts, result)
            current_counts[card] = 0

    def single_hit(h: int, cards_to_return: int = 0) -> float:
        cards: List[str] = list(all_cards_this_combo.keys()) 
        result: List[float] = [0.0]

        def gen(idx: int, curr: Dict[str, int]) -> None:
            if idx == len(cards):
                if sum(curr.values()) != h:
                    return
                hand: List[str] = []
                for c, n in curr.items():
                    hand += [c] * n
                if evaluate_combo_expression(hand, set_config["expression"], combos_constraints, card_types):
                    ways: float = 1.0
                    for c, n in curr.items():
                        ways *= comb(all_cards_this_combo[c], n)
                    result[0] += ways
                return

            c: str = cards[idx]
            max_n: int = min(all_cards_this_combo[c], h - sum(curr.values()))
            for take in range(max_n + 1):
                curr[c] = take
                gen(idx + 1, curr)
            curr.pop(c, None)

        gen(0, {})
        final_hand_size: int = h - cards_to_return
        if final_hand_size <= 0:
            return 0.0
        prob: float = result[0] / comb(d, h)
        return prob

    mulligan_type: str = m.get("type", "traditional")
    if mulligan_type == "traditional":
        attempts: List[int] = [h0]
        if m["first_free"]:
            attempts.append(h0)
        for i in range(m["count"] if m["enabled"] else 0):
            nh: int = h0 - (i + 1)
            if nh <= 0:
                break
            attempts.append(nh)
        p_hits: List[float] = [single_hit(h) for h in attempts]

    elif mulligan_type == "london":
        attempts: List[Tuple[int, int]] = []
        if m["first_free"]:
            attempts.append((h0, 0))
            attempts.append((h0, 1))
        else:
            attempts.append((h0, 0))
        for i in range(m["count"] if m["enabled"] else 0):
            nh: int = h0
            cards_to_return: int = i + (1 if not m["first_free"] else 2)
            if nh - cards_to_return <= 0:
                break
            attempts.append((nh, cards_to_return))
        p_hits: List[float] = [single_hit(h, cards_to_return) for h, cards_to_return in attempts]

    elif mulligan_type == "partial":
        p_hits: List[float] = []
        p_initial: float = single_hit(h0)
        p_hits.append(p_initial)
        def get_min_required(expression: Union[str, Tuple[str, Any, Any]]) -> int:
            if isinstance(expression, str):
                if expression in combos_constraints:
                    return sum(min_count for card_type, (min_count, max_count) in combos_constraints[expression].items()
                               if card_type not in ("mulligans", "mulligan_count", "free_first_mulligan", "hand_size", "deck_size", "mulligan_type"))
                else:
                    return 1
            elif isinstance(expression, tuple):
                op, left, right = expression
                if op == "AND":
                    return get_min_required(left) + get_min_required(right)
                elif op == "OR":
                    return min(get_min_required(left), get_min_required(right))
                elif op == "XOR":
                    return min(get_min_required(left), get_min_required(right))
            return 0

        min_required: int = get_min_required(set_config["expression"])
        max_mulls: int = m["count"] if m["enabled"] else 0
        if m["first_free"]:
            max_mulls += 1

        for mull in range(max_mulls):
            result: List[float] = [0.0]
            initial_counts: Dict[str, int] = defaultdict(int)
            def gen_initial(idx: int, curr: Dict[str, int], hand_size: int) -> None:
                if idx == len(all_cards_this_combo):
                    if sum(curr.values()) != hand_size:
                        return
                    hand: List[str] = []
                    for c, n in curr.items():
                        hand += [c] * n
                    if evaluate_combo_expression(hand, set_config["expression"], combos_constraints, card_types):
                        prob: float = 1.0
                        for c, n in curr.items():
                            prob *= comb(all_cards_this_combo[c], n)
                        result[0] += prob
                        return
                    non_contributing: int = sum(n for c, n in curr.items() if c == "Unspecified" or not any(c in card_types.get(t, [t]) for t in combos_constraints))
                    cards_to_replace: int = min(non_contributing, hand_size - min_required)
                    if cards_to_replace <= 0:
                        return
                    new_deck: Dict[str, int] = dict(all_cards_this_combo)
                    for c, n in curr.items():
                        new_deck[c] -= n
                        if new_deck[c] < 0:
                            return 
                    new_deck["Unspecified"] = new_deck.get("Unspecified", 0) + cards_to_replace
                    new_result: List[float] = [0.0]
                    generate_counts_with_replacements(list(new_deck.keys()), cards_to_replace, defaultdict(int), 0, set_config["expression"], cards_to_replace, new_deck, new_result)
                    prob = 1.0
                    for c, n in curr.items():
                        prob *= comb(all_cards_this_combo[c], n)
                    result[0] += prob * (new_result[0] / comb(d - hand_size, cards_to_replace))
                c: str = list(all_cards_this_combo.keys())[idx]
                max_n: int = min(all_cards_this_combo[c], hand_size - sum(curr.values()))
                for take in range(max_n + 1):
                    curr[c] = take
                    gen_initial(idx + 1, curr, hand_size)
                    curr[c] = 0

            gen_initial(0, defaultdict(int), h0)
            p_hits.append(result[0] / comb(d, h0))

    if not extra_info or not m["enabled"]:
        prod: float = 1.0
        for p in p_hits:
            prod *= (1 - p)
        return 1 - prod

    info: Dict[str, float] = {}
    p_fail: float = 1.0
    for i, p in enumerate(p_hits):
        label: str = "No Mulligan" if i == 0 else f"Mulligan {i}"
        succ: float = p * p_fail
        info[label] = succ
        p_fail *= (1 - p)
    info["Total"] = 1 - p_fail
    return info


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

def update_card_type_cache() -> None:
    state["card_type_cache"].clear()
    for card_type in state["card_types"]:
        state["card_type_cache"][card_type] = sum(state["cards"].get(card, 0) for card in state["card_types"].get(card_type, []))

def get_card_type_quantity(card_type_name: str) -> int:
    if card_type_name in state["card_type_cache"]:
        return state["card_type_cache"][card_type_name]
    qty: int = sum(state["cards"].get(card, 0) for card in state["card_types"].get(card_type_name, []))
    state["card_type_cache"][card_type_name] = qty
    return qty

def clear_screen() -> None:
    console.clear()
    columns, _ = get_terminal_size(fallback=(80, 24))
    if columns >= 205:
        console.print(ASCII_ART, justify="center")
    elif columns >= 135:
        console.print(ASCII_ART_SMALL, justify="center")
    else:
        console.print(ASCII_ART_EXTRA_SMALL, justify="center")
    

def pause() -> None:
    console.print("[prompt]Press Enter to continue...[/prompt]", end="")
    console.input()


def import_ydk_file(path: str) -> None:
    try:
        with open(path, "r") as f:
            lines: List[str] = f.readlines()
        ids: List[int] = []
        in_main: bool = False
        for line in lines:
            line = line.strip()
            if line == "#main": in_main = True; continue
            if line.startswith("#"): in_main = False
            if in_main and line.isdigit(): ids.append(int(line))
        console.print(f"[info]Importing {len(ids)} cards...[/info]")

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), transient=True) as progress:
            task = progress.add_task("Fetching card names...", total=len(ids))
            result: Dict[str, int] = fetch_card_names(ids)

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

def import_ydke_link() -> None:
    link: str = console.input("[prompt]Paste ydke:// link: [/prompt]")
    if not link.startswith("ydke://"):
        console.print("[error]Invalid YDKE link format.[/error]")
        pause()
        import_deck_prompt()
        return
    try:
        encoded_parts: List[str] = link[7:].split("!")
        ids: List[int] = []
        if encoded_parts and encoded_parts[0]:
            decoded: bytes = b64decode(encoded_parts[0])
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

def fetch_card_names(ids: List[int]) -> Dict[str, int]:
    counts: Counter[int] = Counter(ids)
    result: Dict[str, int] = {}

    with Progress() as progress:
        task = progress.add_task("Fetching cards...", total=len(counts))
        
        for card_id, count in counts.items():
            try:
                r = get(f"https://db.ygoprodeck.com/api/v7/cardinfo.php?id={card_id}")
                data: Dict[str, Any] = r.json()
                name: str = data['data'][0]['name']
            except Exception:
                name = f"Unknown Card ({card_id})"
            result[name] = count
            
            progress.update(task, advance=1)

    return result

def import_deck_ygo() -> None:
    clear_screen()
    console.print("[header]Yu-Gi-Oh Deck Import[/header]")
    console.print("[info]1. Import from .ydk file[/info]")
    console.print("[info]2. Import from ydke:// link[/info]")
    choice: str = console.input("[prompt]> [/prompt]")
    if choice == "1":
        file_path: str = console.input("[prompt]Input the file location of the YDK file: [/prompt]")
        import_ydk_file(file_path)
    elif choice == "2":
        import_ydke_link()
    else:
        console.print("[error]Invalid option.[/error]")
        pause()
        import_deck_prompt()

def import_deck_prompt(from_first_screen: bool = True) -> None:
    clear_screen()
    console.print("[header]Deck Import[/header]")
    console.print("[info]1. Import Yu-Gi-Oh Deck[/info]")
    console.print("[info]2. Import .json file[/info]")
    console.print("[info]3. Continue[/info]")
    choice: str = console.input("[prompt]> [/prompt]")
    if choice == "1":
        import_deck_ygo()
    elif choice == "2":
        page_load(from_first_screen=from_first_screen)
    elif choice == "3" or choice == "":
        return
    else:
        console.print("[error]Invalid option.[/error]")
        pause()
        import_deck_prompt()

state: Dict[str, Any] = {
    "deck_size": None,
    "hand_size": None,
    "cards": {},
    "card_types": {},
    "combos": {},
    "combo_sets": {},
    "card_type_cache": {},
}

def page_deck_hand_size() -> None:
    while True:
        clear_screen()
        console.print("[header][1] Deck and Hand Size[/header]\n")
        console.print(f"[info]Current deck size: {state['deck_size']}[/info]")
        console.print(f"[info]Current hand size: {state['hand_size']}[/info]\n")
        console.print("[info]1. Set Deck Size[/info]")
        console.print("[info]2. Set Hand Size[/info]")
        console.print("[info]3. Back[/info]")
        choice: str = console.input("[prompt]> [/prompt]")
        if choice == "1":
            val: str = console.input("[prompt]Enter new deck size: [/prompt]")
            if val.isdigit() and int(val) > 0:
                old_deck_size: Optional[int] = state['deck_size']
                state['deck_size'] = int(val)
                for combo in state['combos'].values():
                    if combo["deck_size"] == old_deck_size:
                        combo["deck_size"] = int(val)
                for combo_set in state['combo_sets'].values():
                    if combo_set["deck_size"] == old_deck_size:
                        combo_set["deck_size"] = int(val)
            else:
                console.print("[error]Invalid deck size.[/error]")
                pause()
        elif choice == "2":
            old_hand_size: Optional[int] = state['hand_size']
            val = console.input("[prompt]Enter new hand size: [/prompt]")
            if val.isdigit() and int(val) > 0:
                state['hand_size'] = int(val)
                for combo in state['combos'].values():
                    if combo["hand_size"] == old_hand_size:
                        combo["hand_size"] = int(val)
                for combo_set in state['combo_sets'].values():
                    if combo_set["hand_size"] == old_hand_size:
                        combo_set["hand_size"] = int(val)
            else:
                console.print("[error]Invalid hand size.[/error]")
                pause()
        elif choice == "3" or choice == "":
            return
        else:
            console.print("[error]Invalid option.[/error]")
            pause()

def page_cards() -> None:
    if state['deck_size'] is None:
        console.print("[error]Deck size must be set first.[/error]")
        pause()
        return
    while True:
        clear_screen()
        console.print("[header][2] Cards and Quantities[/header]\n")
        total: int = sum(state['cards'].values())
        for i, (card, qty) in enumerate(state['cards'].items()):
            console.print(f"[info][{i}] {card}: {qty}[/info]")
        console.print(f"[info]Unspecified cards: {state['deck_size'] - total}[/info]\n")
        console.print("[info]1. Add a Card[/info]")
        console.print("[info]2. Edit/Remove a Card[/info]")
        console.print("[info]3. Back[/info]")
        choice: str = console.input("[prompt]> [/prompt]")
        if choice == "1":
            name: str = console.input("[prompt]Card name: [/prompt]")
            qty_str: str = console.input("[prompt]Quantity: [/prompt]")
            if not qty_str.isdigit() or int(qty_str) < 0:
                console.print("[error]Invalid quantity.[/error]")
                pause()
                continue
            qty: int = int(qty_str)
            if sum(state['cards'].values()) + qty > state['deck_size']:
                console.print("[error]Card quantities exceed deck size. Check your input.[/error]")
                pause()
                continue
            state['cards'][name] = qty
            update_card_type_cache()
        elif choice == "2":
            index_str: str = console.input("[prompt]Enter card index to edit/remove: [/prompt]")
            if not index_str.isdigit() or int(index_str) >= len(state['cards']):
                console.print("[error]Invalid index.[/error]")
                pause()
                continue
            card: str = list(state['cards'].keys())[int(index_str)]
            console.print(f"[info]Selected: {card} (current qty: {state['cards'][card]})[/info]")
            new_val: str = console.input("[prompt]New quantity (or empty to remove): [/prompt]")
            if new_val == "":
                del state['cards'][card]
                update_card_type_cache()
            elif new_val.isdigit() and int(new_val) >= 0:
                qty = int(new_val)
                if sum(state['cards'].values()) - state['cards'][card] + qty > state['deck_size']:
                    console.print("[error]Card quantities exceed deck size.[/error]")
                    pause()
                    continue
                state['cards'][card] = qty
                update_card_type_cache()
            else:
                console.print("[error]Invalid quantity.[/error]")
                pause()
        elif choice == "3" or choice == "":
            return
        else:
            console.print("[error]Invalid option.[/error]")
            pause()

def page_card_types() -> None:
    if not state['cards']:
        console.print("[error]Cards must be added first.[/error]")
        pause()
        return
    while True:
        clear_screen()
        console.print("[header][3] Card Types[/header]\n")
        for i, (ctype, cards_list) in enumerate(state['card_types'].items()):
            console.print(f"[info][{i}] {ctype}: {cards_list}[/info]")
        console.print("\n[info]1. Add Card Type[/info]")
        console.print("[info]2. Edit/Remove Card Type[/info]")
        console.print("[info]3. Back[/info]")
        choice: str = console.input("[prompt]> [/prompt]")
        if choice == "1":
            name: str = console.input("[prompt]Card Type name: [/prompt]")
            if name in state['card_types']:
                console.print("[error]Card Type already exists.[/error]")
                pause()
                continue
            available_cards: List[str] = list(state['cards'].keys())
            console.print("[info]Available Cards:[/info]")
            for i, card in enumerate(available_cards):
                console.print(f"[info][{i}] {card}[/info]")
            indexes: str = console.input("[prompt]Enter indices of cards to add (comma-separated): [/prompt]")
            try:
                indices: List[int] = list(map(int, indexes.split(",")))
                state['card_types'][name] = list({available_cards[i] for i in indices if 0 <= i < len(available_cards)})
                update_card_type_cache()
            except:
                console.print("[error]Invalid input.[/error]")
                pause()
        elif choice == "2":
            idx_str: str = console.input("[prompt]Enter index of Card Type to edit/remove: [/prompt]")
            if not idx_str.isdigit() or int(idx_str) >= len(state['card_types']):
                console.print("[error]Invalid index.[/error]")
                pause()
                continue
            ctype_name: str = list(state['card_types'].keys())[int(idx_str)]
            console.print(f"[info]Selected: {ctype_name}[/info]")
            console.print("[info]1. Edit[/info]")
            console.print("[info]2. Remove[/info]")
            sub: str = console.input("[prompt]> [/prompt]")
            if sub == "1":
                available_cards = list(state['cards'].keys())
                console.print("[info]Available Cards:[/info]")
                for i, card in enumerate(available_cards):
                    console.print(f"[info][{i}] {card}[/info]")
                indexes = console.input("[prompt]Enter indices of cards to set (comma-separated): [/prompt]")
                try:
                    indices = list(map(int, indexes.split(",")))
                    state['card_types'][ctype_name] = list({available_cards[i] for i in indices if 0 <= i < len(available_cards)})
                    update_card_type_cache()
                except:
                    console.print("[error]Invalid input.[/error]")
                    pause()
            elif sub == "2":
                del state['card_types'][ctype_name]
                update_card_type_cache()
            else:
                console.print("[error]Invalid option.[/error]")
                pause()
        elif choice == "3" or choice == "":
            return
        else:
            console.print("[error]Invalid option.[/error]")
            pause()

def page_combos() -> None:
    clear_screen()
    console.print("[header][4] Combos[/header]\n")

    if not state["card_types"] and not state["cards"]:
        console.print("[error]You must define card types or add cards before adding combos.[/error]")
        pause()
        return

    console.print("[info]Defined combos:[/info]")
    for i, name in enumerate(state["combos"]):
        console.print(f"[info]{i}. {name}[/info]")

    console.print("\n[info]1. Add Combo[/info]")
    console.print("[info]2. Edit/Remove Combo[/info]")
    console.print("[info]3. Back[/info]")
    choice: str = console.input("[prompt]> [/prompt]")

    if choice == "1":
        name: str = console.input("[prompt]Enter combo name: [/prompt]").strip()
        if not name:
            console.print("[error]Invalid name.[/error]")
            pause()
            return

        console.print("[info]Available card types:[/info]")
        card_type_keys: List[str] = list(state["card_types"].keys())
        for i, ct in enumerate(card_type_keys):
            console.print(f"[info]T{i}. {ct}[/info]")
        console.print("[info]Available cards:[/info]")
        card_keys: List[str] = list(state["cards"].keys())
        for i, card in enumerate(card_keys):
            console.print(f"[info]C{i}. {card}[/info]")
        selected: str = console.input("[prompt]Enter indices of card types (T#) or cards (C#) to include (comma-separated): [/prompt]")
        try:
            selected_indices: List[str] = [i.strip() for i in selected.split(",")]
        except:
            console.print("[error]Invalid input.[/error]")
            pause()
            return

        combo_requirements: Dict[str, Tuple[int, int]] = {}
        for idx in selected_indices:
            if not (idx.startswith("T") or idx.startswith("C")):
                console.print(f"[error]Invalid index format: {idx}[/error]")
                pause()
                return
            is_card_type: bool = idx.startswith("T")
            idx_num_str: str = idx[1:]
            if not idx_num_str.isdigit():
                console.print(f"[error]Invalid index number: {idx}[/error]")
                pause()
                return
            idx_num: int = int(idx_num_str)
            if is_card_type:
                if idx_num >= len(card_type_keys):
                    console.print(f"[error]Invalid card type index: {idx}[/error]")
                    pause()
                    return
                item_name: str = card_type_keys[idx_num]
                max_available: int = get_card_type_quantity(item_name)
            else:
                if idx_num >= len(card_keys):
                    console.print(f"[error]Invalid card index: {idx}[/error]")
                    pause()
                    return
                item_name = card_keys[idx_num]
                max_available = state["cards"].get(item_name, 0)

            console.print(f"\n[info]{'Card Type' if is_card_type else 'Card'}: {item_name} (Total quantity in deck: {max_available})[/info]")
            try:
                min_ct_str: str = console.input("[prompt]Minimum required (default 1): [/prompt]").strip()
                max_ct_str: str = console.input(f"[prompt]Maximum allowed (default {max_available}): [/prompt]").strip()
                min_ct: int = int(min_ct_str) if min_ct_str else 1
                max_ct: int = int(max_ct_str) if max_ct_str else max_available
                if max_ct > max_available:
                    console.print(f"[error]Max cannot exceed total available ({max_available}).[/error]")
                    pause()
                    return
                if min_ct > max_ct:
                    console.print("[error]Minimum cannot exceed maximum.[/error]")
                    pause()
                    return
                combo_requirements[item_name] = (min_ct, max_ct)
            except:
                console.print("[error]Invalid input for min/max.[/error]")
                pause()
                return

        console.print("\n[prompt]Use custom deck/hand size? (y/N): [/prompt]", end="")
        custom: bool = console.input().lower().strip() == "y"
        deck_size: Optional[int] = state["deck_size"]
        hand_size: Optional[int] = state["hand_size"]
        if custom:
            try:
                deck_size = int(console.input(f"[prompt]Custom deck size (default {deck_size}): [/prompt]") or deck_size)
                hand_size = int(console.input(f"[prompt]Custom hand size (default {hand_size}): [/prompt]") or hand_size)
                if deck_size <= 0 or hand_size <= 0:
                    console.print("[error]Deck and hand sizes must be positive.[/error]")
                    pause()
                    return
            except:
                console.print("[error]Invalid custom size.[/error]")
                pause()
                return

        console.print("\n[prompt]Include mulligans? (y/N): [/prompt]", end="")
        include_mulligans: bool = console.input().lower().strip() == "y"
        mulligans: Dict[str, Any] = {
            "enabled": include_mulligans,
            "count": 0,
            "first_free": False,
            "type": "traditional"
        }
        if include_mulligans:
            console.print("\n[info]Select mulligan type:[/info]")
            console.print("[info]1. Traditional (redraw smaller hand)[/info]")
            console.print("[info]2. London (draw full hand, return cards)[/info]")
            console.print("[info]3. Partial (replace specific cards)[/info]")
            mulligan_choice: str = console.input("[prompt]> [/prompt]").strip()
            if mulligan_choice == "1":
                mulligans["type"] = "traditional"
            elif mulligan_choice == "2":
                mulligans["type"] = "london"
            elif mulligan_choice == "3":
                mulligans["type"] = "partial"
            else:
                console.print("[error]Invalid mulligan type. Defaulting to Traditional.[/error]")
                mulligans["type"] = "traditional"

            default_mull_count: int = hand_size - sum([v[0] for v in combo_requirements.values()]) if hand_size is not None else 0
            try:
                count_str: str = console.input(f"[prompt]Number of mulligans (default {default_mull_count}): [/prompt]").strip()
                mulligans["count"] = int(count_str) if count_str else default_mull_count
                if mulligans["count"] < 0:
                    console.print("[error]Mulligan count cannot be negative.[/error]")
                    pause()
                    return
            except:
                console.print("[error]Invalid mulligan count.[/error]")
                pause()
                return
            first: str = console.input("[prompt]Is the first mulligan free? (y/N): [/prompt]").strip().lower()
            mulligans["first_free"] = first == "y"

        state["combos"][name] = {
            "requirements": combo_requirements,
            "deck_size": deck_size,
            "hand_size": hand_size,
            "mulligans": mulligans
        }
        console.print(f"[success]Combo '{name}' added.[/success]")
        pause()

    elif choice == "2":
        if not state["combos"]:
            console.print("[error]No combos to edit.[/error]")
            pause()
            return
        try:
            idx: int = int(console.input("[prompt]Enter index of combo to edit: [/prompt]"))
            name = list(state["combos"].keys())[idx]
        except:
            console.print("[error]Invalid index.[/error]")
            pause()
            return
        console.print("[info]1. Edit[/info]")
        console.print("[info]2. Remove[/info]")
        sub: str = console.input("[prompt]> [/prompt]")
        if sub == "1":
            combo: Dict[str, Any] = state["combos"][name]
            console.print(f"\n[info]Editing combo '{name}':[/info]")

            change_ct: str = console.input("[prompt]Do you want to change the card types/cards in the combo? (y/N): [/prompt]").strip().lower()
            if change_ct == "y":
                combo_requirements = {}
                card_type_keys = list(state["card_types"].keys())
                card_keys = list(state["cards"].keys())
                console.print("[info]Available card types:[/info]")
                for i, ct in enumerate(card_type_keys):
                    console.print(f"[info]T{i}. {ct}[/info]")
                console.print("[info]Available cards:[/info]")
                for i, card in enumerate(card_keys):
                    console.print(f"[info]C{i}. {card}[/info]")
                selected = console.input("[prompt]Enter indices of card types (T#) or cards (C#) to include (comma-separated): [/prompt]")
                try:
                    selected_indices = [i.strip() for i in selected.split(",")]
                except:
                    console.print("[error]Invalid input.[/error]")
                    pause()
                    return

                for idx_item in selected_indices:
                    if not (idx_item.startswith("T") or idx_item.startswith("C")):
                        console.print(f"[error]Invalid index format: {idx_item}[/error]")
                        pause()
                        return
                    is_card_type = idx_item.startswith("T")
                    idx_num_str = idx_item[1:]
                    if not idx_num_str.isdigit():
                        console.print(f"[error]Invalid index number: {idx_item}[/error]")
                        pause()
                        return
                    idx_num = int(idx_num_str)
                    if is_card_type:
                        if idx_num >= len(card_type_keys):
                            console.print(f"[error]Invalid card type index: {idx_item}[/error]")
                            pause()
                            return
                        item_name = card_type_keys[idx_num]
                        max_available = get_card_type_quantity(item_name)
                    else:
                        if idx_num >= len(card_keys):
                            console.print(f"[error]Invalid card index: {idx_item}[/error]")
                            pause()
                            return
                        item_name = card_keys[idx_num]
                        max_available = state["cards"].get(item_name, 0)

                    console.print(f"\n[info]{'Card Type' if is_card_type else 'Card'}: {item_name} (Total quantity in deck: {max_available})[/info]")
                    try:
                        min_ct_str = console.input("[prompt]Minimum required (default 1): [/prompt]").strip()
                        max_ct_str = console.input(f"[prompt]Maximum allowed (default {max_available}): [/prompt]").strip()
                        min_ct = int(min_ct_str) if min_ct_str else 1
                        max_ct = int(max_ct_str) if max_ct_str else max_available
                        if max_ct > max_available:
                            console.print(f"[error]Max cannot exceed total available ({max_available}).[/error]")
                            pause()
                            return
                        if min_ct > max_ct:
                            console.print("[error]Minimum cannot exceed maximum.[/error]")
                            pause()
                            return
                        combo_requirements[item_name] = (min_ct, max_ct)
                    except:
                        console.print("[error]Invalid input for min/max.[/error]")
                        pause()
                        return
                combo["requirements"] = combo_requirements

            change_mull: str = console.input("[prompt]Do you want to change the mulligan settings? (y/N): [/prompt]").strip().lower()
            if change_mull == "y":
                include_mulligans = console.input("[prompt]Include mulligans? (y/N): [/prompt]").strip().lower() == "y"
                mulligans = {
                    "enabled": include_mulligans,
                    "count": 0,
                    "first_free": False,
                    "type": combo["mulligans"].get("type", "traditional")
                }
                if include_mulligans:
                    console.print("\n[info]Select mulligan type:[/info]")
                    console.print("[info]1. Traditional (redraw smaller hand)[/info]")
                    console.print("[info]2. London (draw full hand, return cards)[/info]")
                    console.print("[info]3. Partial (replace specific cards)[/info]")
                    mulligan_choice = console.input("[prompt]> [/prompt]").strip()
                    if mulligan_choice == "1":
                        mulligans["type"] = "traditional"
                    elif mulligan_choice == "2":
                        mulligans["type"] = "london"
                    elif mulligan_choice == "3":
                        mulligans["type"] = "partial"
                    else:
                        console.print("[error]Invalid mulligan type. Defaulting to Traditional.[/error]")
                        mulligans["type"] = "traditional"

                    default_mull_count = combo["hand_size"] - sum([v[0] for v in combo["requirements"].values()])
                    try:
                        count_str = console.input(f"[prompt]Number of mulligans (default {default_mull_count}): [/prompt]").strip()
                        mulligans["count"] = int(count_str) if count_str else default_mull_count
                        if mulligans["count"] < 0:
                            console.print("[error]Mulligan count cannot be negative.[/error]")
                            pause()
                            return
                    except:
                        console.print("[error]Invalid mulligan count.[/error]")
                        pause()
                        return
                    first = console.input("[prompt]Is the first mulligan free? (y/N): [/prompt]").strip().lower()
                    mulligans["first_free"] = first == "y"
                combo["mulligans"] = mulligans

            change_sizes: str = console.input("[prompt]Do you want to change the deck size and hand size for this combo? (y/N): [/prompt]").strip().lower()
            if change_sizes == "y":
                try:
                    deck_size_str: str = console.input(f"[prompt]Custom deck size (default {combo.get('deck_size', state['deck_size'])}): [/prompt]").strip()
                    hand_size_str: str = console.input(f"[prompt]Custom hand size (default {combo.get('hand_size', state['hand_size'])}): [/prompt]").strip()
                    if deck_size_str:
                        combo["deck_size"] = int(deck_size_str)
                    if hand_size_str:
                        combo["hand_size"] = int(hand_size_str)
                    if combo["deck_size"] <= 0 or combo["hand_size"] <= 0:
                        console.print("[error]Deck and hand sizes must be positive.[/error]")
                        pause()
                        return
                except:
                    console.print("[error]Invalid size input.[/error]")
                    pause()
                    return
            state["combos"][name] = combo
            console.print(f"[success]Combo '{name}' updated.[/success]")
            pause()
        elif sub == "2":
            del state["combos"][name]
        else:
            console.print("[error]Invalid option.[/error]")
            pause()

    elif choice == "3" or choice == "":
        return
    else:
        console.print("[error]Invalid option.[/error]")
        pause()

def validate_combo_set(expr: List[str]) -> bool:
    stack: int = 0
    last: str = "operator"
    for token in expr:
        if token == "(":
            stack += 1
            last = "open"
        elif token == ")":
            stack -= 1
            if stack < 0:
                return False
            last = "close"
        elif token in {"AND", "OR", "XOR"}:
            if last == "operator":
                return False
            last = "operator"
        else:
            if token not in state['combos'] and token not in state['cards']:
                return False
            last = "combo"
    return stack == 0 and last in {"combo", "close"}

def page_combo_sets() -> None:
    if not state['combos'] and not state['cards']:
        console.print("[error]Define combos or add cards first.[/error]")
        pause()
        return

    while True:
        clear_screen()
        console.print("[header][5] Combo Sets[/header]\n")
        for i, (name, data) in enumerate(state['combo_sets'].items()):
            expr_str: str = " ".join(data["expression"])
            m: Dict[str, Any] = data["mulligans"]
            console.print(f"[info][{i}] {name}: {expr_str}  (D{data['deck_size']} H{data['hand_size']} Mull:{m['enabled']}, Type:{m['type']})[/info]")
        console.print("\n[info]1. Add Combo Set[/info]")
        console.print("[info]2. Edit/Remove Combo Set[/info]")
        console.print("[info]3. Back[/info]")
        choice: str = console.input("[prompt]> [/prompt]")

        if choice == "1":
            name: str = console.input("[prompt]Combo Set name: [/prompt]").strip()
            expr: List[str] = []
            combo_keys: List[str] = list(state['combos'])
            card_keys: List[str] = list(state['cards'])
            while True:
                clear_screen()
                console.print(f"[info]Expression so far: {' '.join(expr)}[/info]")
                console.print("[info]Combos:[/info]")
                for idx, c in enumerate(combo_keys):
                    console.print(f"[info]  [C{idx}] {c}[/info]")
                console.print("[info]Cards:[/info]")
                for idx, c in enumerate(card_keys):
                    console.print(f"[info]  [K{idx}] {c}[/info]")
                console.print("[info]Operators: [A]ND [O]R [X]OR, [B]racket, [D]one[/info]")
                tok: str = console.input("[prompt]> [/prompt]").upper()
                if tok == "D":
                    break
                if tok in {"A","O","X"}:
                    expr.append({"A":"AND","O":"OR","X":"XOR"}[tok])
                elif tok == "B":
                    expr.append(console.input("[prompt]Enter '(' or ')': [/prompt]").strip())
                elif tok.startswith("C") and tok[1:].isdigit() and int(tok[1:]) < len(combo_keys):
                    expr.append(combo_keys[int(tok[1:])])
                elif tok.startswith("K") and tok[1:].isdigit() and int(tok[1:]) < len(card_keys):
                    expr.append(card_keys[int(tok[1:])])
                else:
                    console.print("[error]Invalid input.[/error]")
                    pause()
                    continue
            if not validate_combo_set(expr):
                console.print("[error]Invalid expression.[/error]")
                pause()
                continue
            d_size: Optional[int] = state['deck_size']
            h_size: Optional[int] = state['hand_size']
            if console.input("[prompt]Custom deck/hand size? (y/N): [/prompt]").lower() == "y":
                d_size = int(console.input(f"[prompt]Deck size [{d_size}]: [/prompt]") or d_size)
                h_size = int(console.input(f"[prompt]Hand size [{h_size}]: [/prompt]") or h_size)
            mull: Dict[str, Any] = {"enabled": False, "count": 0, "first_free": False, "type": "traditional"}
            if console.input("[prompt]Enable mulligans? (y/N): [/prompt]").lower() == "y":
                mull["enabled"] = True
                console.print("\n[info]Select mulligan type:[/info]")
                console.print("[info]1. Traditional (redraw smaller hand)[/info]")
                console.print("[info]2. London (draw full hand, return cards)[/info]")
                console.print("[info]3. Partial (replace specific cards)[/info]")
                mulligan_choice = console.input("[prompt]> [/prompt]").strip()
                if mulligan_choice == "1":
                    mull["type"] = "traditional"
                elif mulligan_choice == "2":
                    mull["type"] = "london"
                elif mulligan_choice == "3":
                    mull["type"] = "partial"
                else:
                    console.print("[error]Invalid mulligan type. Defaulting to Traditional.[/error]")
                    mull["type"] = "traditional"

                total_min_required: int = 0
                for c in expr:
                    if c in state['combos']:
                        requirements: Dict[str, Any] = state['combos'][c]["requirements"]
                        total_min_required += sum(min_count for min_count, max_count in requirements.values())
                    elif c in state['cards']:
                        total_min_required += 1
                default: int = max(0, (h_size if h_size is not None else 0) - total_min_required)
                try:
                    mull_count_str: str = console.input(f"[prompt]Count [{default}]: [/prompt]") or str(default)
                    mull["count"] = int(mull_count_str)
                    if mull["count"] < 0:
                        console.print("[error]Mulligan count cannot be negative.[/error]")
                        pause()
                        continue
                except ValueError:
                    console.print("[error]Invalid mulligan count. Using default.[/error]")
                    mull["count"] = default
                mull["first_free"] = (console.input("[prompt]First free? (y/N): [/prompt]").lower() == "y")

            state['combo_sets'][name] = {
                "expression": expr,
                "deck_size": d_size,
                "hand_size": h_size,
                "mulligans": mull
            }
            console.print(f"[success]Combo Set '{name}' added.[/success]")
            pause()

        elif choice == "2":
            if not state['combo_sets']:
                console.print("[error]No combo sets to edit.[/error]")
                pause()
                continue
            try:
                idx = int(console.input("[prompt]Index to edit/remove: [/prompt]"))
                key: str = list(state['combo_sets'])[idx]
            except (ValueError, IndexError):
                console.print("[error]Invalid index.[/error]")
                pause()
                continue
            console.print(f"[prompt][1] Edit '{key}'  [2] Remove: [/prompt]")
            sub: str = console.input(f"[prompt]>[/prompt]")
            if sub == "2":
                del state['combo_sets'][key]
                console.print(f"[success]Removed '{key}'.[/success]")
                pause()
                continue
            data: Dict[str, Any] = state['combo_sets'][key]
            if console.input("[prompt]Change expression? (y/N): [/prompt]").lower() == "y":
                expr = []
                combo_keys = list(state['combos'])
                card_keys = list(state['cards'])
                while True:
                    clear_screen()
                    console.print(f"[info]Expression so far: {' '.join(expr)}[/info]")
                    console.print("[info]Combos:[/info]")
                    for idx2, c in enumerate(combo_keys):
                        console.print(f"[info]  [C{idx2}] {c}[/info]")
                    console.print("[info]Cards:[/info]")
                    for idx2, c in enumerate(card_keys):
                        console.print(f"[info]  [K{idx2}] {c}[/info]")
                    console.print("[info]Operators: [A]ND [O]R [X]OR, [B]racket, [D]one[/info]")
                    tok = console.input("[prompt]> [/prompt]").upper()
                    if tok == "D":
                        break
                    if tok in {"A","O","X"}:
                        expr.append({"A":"AND","O":"OR","X":"XOR"}[tok])
                    elif tok == "B":
                        expr.append(console.input("[prompt]Enter '(' or ')': [/prompt]").strip())
                    elif tok.startswith("C") and tok[1:].isdigit() and int(tok[1:]) < len(combo_keys):
                        expr.append(combo_keys[int(tok[1:])])
                    elif tok.startswith("K") and tok[1:].isdigit() and int(tok[1:]) < len(card_keys):
                        expr.append(card_keys[int(tok[1:])])
                    else:
                        console.print("[error]Invalid input.[/error]")
                        pause()
                        continue
                if validate_combo_set(expr):
                    data["expression"] = expr
                else:
                    console.print("[error]Invalid expression.[/error]")
                    pause()
                    continue

            if console.input("[prompt]Change deck/hand size? (y/N): [/prompt]").lower() == "y":
                d: int = int(console.input(f"[prompt]Deck size [{data['deck_size']}]: [/prompt]") or data['deck_size'])
                h: int = int(console.input(f"[prompt]Hand size [{data['hand_size']}]: [/prompt]") or data['hand_size'])
                data['deck_size'], data['hand_size'] = d, h

            if console.input("[prompt]Change mulligan settings? (y/N): [/prompt]").lower() == "y":
                m = data['mulligans']
                m["enabled"] = (console.input(f"[prompt]Enable? (y/N) [{m['enabled']}]: [/prompt]").lower() == "y")
                if m["enabled"]:
                    console.print("\n[info]Select mulligan type:[/info]")
                    console.print("[info]1. Traditional (redraw smaller hand)[/info]")
                    console.print("[info]2. London (draw full hand, return cards)[/info]")
                    console.print("[info]3. Partial (replace specific cards)[/info]")
                    mulligan_choice = console.input("[prompt]> [/prompt]").strip()
                    if mulligan_choice == "1":
                        m["type"] = "traditional"
                    elif mulligan_choice == "2":
                        m["type"] = "london"
                    elif mulligan_choice == "3":
                        m["type"] = "partial"
                    else:
                        console.print("[error]Invalid mulligan type. Defaulting to Traditional.[/error]")
                        m["type"] = "traditional"
                    total_min_required = 0
                    for c in data['expression']:
                        if c in state['combos']:
                            requirements = state['combos'][c]["requirements"]
                            total_min_required += sum(min_count for min_count, max_count in requirements.values())
                        elif c in state['cards']:
                            total_min_required += 1
                    default = max(0, data['hand_size'] - total_min_required)
                    try:
                        mull_count_str = console.input(f"[prompt]Count [{default}]: [/prompt]") or str(default)
                        m["count"] = int(mull_count_str)
                        if m["count"] < 0:
                            console.print("[error]Mulligan count cannot be negative.[/error]")
                            pause()
                            continue
                    except ValueError:
                        console.print("[error]Invalid mulligan count. Using default.[/error]")
                        m["count"] = default
                    m["first_free"] = (console.input(f"[prompt]First free? (y/N) [{m['first_free']}]: [/prompt]").lower() == "y")
                data['mulligans'] = m

            state['combo_sets'][key] = data
            console.print(f"[success]Updated '{key}'.[/success]")
            pause()
        elif choice == "3" or choice == "":
            return
        else:
            console.print("[error]Invalid option.[/error]")
            pause()

def infix_to_postfix(tokens: List[str]) -> List[str]:
    precedence: Dict[str, int] = {'OR': 1, 'XOR': 2, 'AND': 3}
    output: List[str] = []
    stack: List[str] = []
    for token in tokens:
        if token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop() 
        elif token in precedence:
            while stack and stack[-1] in precedence and precedence[stack[-1]] >= precedence[token]:
                output.append(stack.pop())
            stack.append(token)
        else:
            output.append(token)
    while stack:
        output.append(stack.pop())
    return output

def parse_postfix(tokens: List[str]) -> Optional[Union[str, Tuple[str, Any, Any]]]:
    stack: List[Union[str, Tuple[str, Any, Any]]] = []
    for token in tokens:
        if token in {"AND", "OR", "XOR"}:
            if len(stack) < 2:
                console.print(f"[error]Invalid expression: {tokens}[/error]")
                return None
            b: Union[str, Tuple[str, Any, Any]] = stack.pop()
            a: Union[str, Tuple[str, Any, Any]] = stack.pop()
            stack.append((token, a, b)) 
        else:
            stack.append(token)
    if len(stack) != 1:
        console.print(f"[error]Malformed expression: {tokens}[/error]")
        return None
    return stack[0]

def page_calculate_probability() -> None:
    if not state['combos']:
        console.print("[error]Define at least one combo first.[/error]")
        pause()
        return

    clear_screen()
    console.print("[header][6] Calculate Probability[/header]\n")

    card_counts: Dict[str, int] = dict(state["cards"])
    combos_input: Dict[str, Dict[str, Any]] = {}
    for name, combo in state['combos'].items():
        constraints: Dict[str, Any] = {}
        for T, bounds in combo["requirements"].items():
            constraints[T] = tuple(bounds)
        constraints["mulligans"]         = combo["mulligans"]["enabled"]
        constraints["mulligan_count"]    = combo["mulligans"]["count"]
        constraints["free_first_mulligan"] = combo["mulligans"]["first_free"]
        constraints["hand_size"]         = combo["hand_size"]
        constraints["deck_size"]         = combo["deck_size"]
        combos_input[name] = constraints

    extra_info_input: str = console.input("[prompt]Show detailed probability info (per mulligan attempt)? (y/N): [/prompt]").strip().lower()
    extra_info: bool = extra_info_input == "y"

    combo_results: Dict[str, Union[float, Dict[str, float]]] = combo_type_probability(
        state['deck_size'], 
        state['hand_size'],
        card_counts,
        state['card_types'],
        combos_input,
        extra_info=extra_info
    )

    console.print("[bold magenta]Combo Probabilities[/bold magenta]\n")
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Combo Name", style="bold")
    table.add_column("Label", style="green", no_wrap=True)
    table.add_column("Probability", justify="right", style="cyan")

    for name, result in combo_results.items():
        if isinstance(result, dict):
            if set(result.keys()) == {"No Mulligan", "Total"} and result["No Mulligan"] == result["Total"]:
                table.add_row(name, "-", f"{result['Total']*100:.4f}% ({result['Total']:.6f})")
            else:
                for label, val in result.items():
                    table.add_row(name, label, f"{val*100:.4f}% ({val:.6f})")
        else:
            table.add_row(name, "-", f"{result*100:.4f}% ({result:.6f})")

    console.print(table)
    parsed_sets: Dict[str, Union[str, Tuple[str, Any, Any]]] = {}
    for name, data in state['combo_sets'].items():
        postfix: List[str] = infix_to_postfix(data["expression"])
        tree: Optional[Union[str, Tuple[str, Any, Any]]] = parse_postfix(postfix)
        if tree is not None:
            parsed_sets[name] = tree

    if parsed_sets:
        console.print("\n[bold magenta]Combo Set Probabilities[/bold magenta]\n")
        set_table = Table(show_header=True, header_style="bold yellow")
        set_table.add_column("Combo Set", style="bold")
        set_table.add_column("Label", style="green")
        set_table.add_column("Probability", justify="right", style="cyan")

        for name, tree in parsed_sets.items():
            cfg: Dict[str, Any] = {
                "expression": tree,
                "deck_size": state['combo_sets'][name]["deck_size"],
                "hand_size": state['combo_sets'][name]["hand_size"],
                "mulligans": state['combo_sets'][name]["mulligans"]
            }
            res: Union[float, Dict[str, float]] = combo_set_probability(
                cfg,
                card_counts,
                state["card_types"],
                combos_input,
                extra_info=extra_info
            )
            if isinstance(res, dict):
                for label, p in res.items():
                    set_table.add_row(name, label, f"{p*100:.4f}% ({p:.6f})")
            else:
                set_table.add_row(name, "-", f"{res*100:.4f}% ({res:.6f})")

        console.print(set_table)
    else:
        console.print("[grey]No combo sets defined.[/grey]")

    pause()
    


def page_graph() -> None:
    clear_screen()
    width, height = size()
    console.print("[header][10] Graphs[/header]\n")
    
    console.print("[info]Load settings? (y/n)[/info]")
    load_settings_str: str = console.input("[prompt]> [/prompt]").lower()
    load_settings: bool = load_settings_str == 'y'
    settings: Dict[str, Any] = {}
    if load_settings:
        try:
            with open('plot_settings.json', 'r') as f:
                settings = load(f)
            console.print("[info]Settings loaded.[/info]")
        except Exception as e:
            console.print(f"[error]Failed to load settings: {e}[/error]")
    
    console.print("[info]Select graph type:[/info]")
    console.print("[info]1. Hypergeometric Probability[/info]")
    console.print("[info]2. Combo/Combo Set Probability[/info]")
    graph_type: str = console.input("[prompt]> [/prompt]") if not load_settings or 'graph_type' not in settings else settings['graph_type']
    if graph_type not in ["1", "2"]:
        console.print("[error]Invalid graph type.[/error]")
        pause()
        return

    console.print("\n[info]Customize plot style:[/info]")
    console.print("[info]Line style: 1. Solid (-), 2. Dashed (--), 3. Dotted (:)[/info]")
    line_style_choice: str = console.input("[prompt]> [/prompt]") if not load_settings or 'line_style' not in settings else settings['line_style']
    line_style: str = {'1': '-', '2': '--', '3': ':'}.get(line_style_choice, '-')
    console.print("[info]Marker style: 1. Circle (o), 2. Square (s), 3. Triangle (^)[/info]")
    marker_style_choice: str = console.input("[prompt]> [/prompt]") if not load_settings or 'marker_style' not in settings else settings['marker_style']
    marker_style: str = {'1': 'o', '2': 's', '3': '^'}.get(marker_style_choice, 'o')
    colors: List[str] = ['cyan', 'magenta', 'yellow', 'green', 'red', 'blue', 'orange', 'pink']

    def get_int_input(prompt: str, minimum: int = 0) -> Optional[int]:
        val: str = console.input(f"[prompt]{prompt} > [/prompt]")
        if val.isdigit() and int(val) >= minimum:
            return int(val)
        return None

    def get_multiple_ints(prompt: str) -> Optional[List[int]]:
        vals: str = console.input(f"[prompt]{prompt} > [/prompt]")
        try:
            vals_list: List[str] = vals.replace(',', ' ').split()
            ints: List[int] = [int(v) for v in vals_list if int(v) > 0]
            return ints if ints else None
        except:
            return None

    if graph_type == "2":
        console.print("\n[info]Select items to analyze (space-separated indices):[/info]")
        items: List[Tuple[str, str]] = [(name, "Combo") for name in state['combos']] + [(name, "Combo Set") for name in state['combo_sets']]
        if not items:
            console.print("[error]No combos or combo sets defined.[/error]")
            pause()
            return
        for i, (name, item_type) in enumerate(items):
            console.print(f"[info]{i}. {name} ({item_type})[/info]")
        item_indices_input: Optional[List[int]] = get_multiple_ints("[prompt]> [/prompt]") if not load_settings or 'item_idx' not in settings else settings['item_idx']
        if not item_indices_input or any(i >= len(items) for i in item_indices_input):
            console.print("[error]Invalid selection.[/error]")
            pause()
            return
        selected_items: List[Tuple[str, str]] = [items[i] for i in item_indices_input]
        
        console.print("\n[info]Set x-Axis (parameter to vary):[/info]")
        console.print("[info]1. Deck Size[/info]")
        console.print("[info]2. Hand Size[/info]")
        x_axis: str = console.input("[prompt]> [/prompt]") if not load_settings or 'x_axis' not in settings else settings['x_axis']
        if x_axis not in ["1", "2"]:
            console.print("[error]Invalid x-axis choice.[/error]")
            pause()
            return
        x_label: str = "Deck Size" if x_axis == "1" else "Hand Size"
        min_prompt: str = "Min Deck Size (≥1)" if x_axis == "1" else "Min Hand Size (≥1)"
        max_prompt: str = "Max Deck Size (≥ Min)" if x_axis == "1" else "Max Hand Size (≥ Min)"
        x_min: Optional[int] = get_int_input(f"[prompt]{min_prompt} [/prompt]", minimum=1)
        if x_min is None:
            console.print("[error]Invalid min value.[/error]")
            pause()
            return
        x_max: Optional[int] = get_int_input(f"[prompt]{max_prompt} [/prompt]", minimum=x_min)
        if x_max is None:
            console.print("[error]Invalid max value.[/error]")
            pause()
            return
        fixed_label: str = "Hand Size" if x_axis == "1" else "Deck Size"
        fixed_val_str: str = console.input(f"[prompt]Set {fixed_label} > [/prompt]") if not load_settings or 'fixed_val' not in settings else str(settings['fixed_val'])
        if not fixed_val_str.isdigit() or int(fixed_val_str) < 1:
            console.print(f"[error]Invalid {fixed_label}.[/error]")
            pause()
            return
        fixed_val: int = int(fixed_val_str)
        
        card_counts = dict(state["cards"])
        x_vals: np.ndarray = np.arange(x_min, x_max + 1)
        all_probabilities: List[Tuple[str, str, List[float]]] = []
        
        for selected_name, selected_type in selected_items:
            probabilities: List[float] = []
            if selected_type == "Combo":
                combo_data: Dict[str, Any] = state['combos'][selected_name]
                constraints = {k: tuple(v) for k, v in combo_data["requirements"].items()}
                constraints["mulligans"] = combo_data["mulligans"]["enabled"]
                constraints["mulligan_count"] = combo_data["mulligans"]["count"]
                constraints["free_first_mulligan"] = combo_data["mulligans"]["first_free"]
                constraints["mulligan_type"] = combo_data["mulligans"].get("type", "traditional")
                
                for x_val in x_vals:
                    deck_size_calc: int = x_val if x_axis == "1" else fixed_val
                    hand_size_calc: int = x_val if x_axis == "2" else fixed_val
                    if hand_size_calc > deck_size_calc:
                        console.print(f"[error]Hand Size ({hand_size_calc}) cannot exceed Deck Size ({deck_size_calc}).[/error]")
                        pause()
                        return
                    constraints["deck_size"] = deck_size_calc
                    constraints["hand_size"] = hand_size_calc
                    result_calc: Union[float, Dict[str, float]] = combo_type_probability(
                        deck_size_calc,
                        hand_size_calc,
                        card_counts,
                        state['card_types'],
                        {selected_name: constraints},
                        extra_info=False
                    )
                    if isinstance(result_calc, dict):
                        probabilities.append(result_calc["Total"])
                    else:
                        probabilities.append(result_calc)
            
            else:
                combo_set_data: Dict[str, Any] = state['combo_sets'][selected_name]
                postfix: List[str] = infix_to_postfix(combo_set_data["expression"])
                tree: Optional[Union[str, Tuple[str, Any, Any]]] = parse_postfix(postfix)
                if tree is None:
                    console.print(f"[error]Invalid combo set expression for {selected_name}.[/error]")
                    pause()
                    return
                cfg: Dict[str, Any] = {
                    "expression": tree,
                    "deck_size": combo_set_data["deck_size"],
                    "hand_size": combo_set_data["hand_size"],
                    "mulligans": combo_set_data["mulligans"]
                }
                combos_input_calc: Dict[str, Dict[str, Any]] = {k: {t: tuple(v) for t, v in c["requirements"].items()} for k, c in state['combos'].items()}
                for k, c in combos_input_calc.items():
                    c["mulligans"] = state['combos'][k]["mulligans"]["enabled"]
                    c["mulligan_count"] = state['combos'][k]["mulligans"]["count"]
                    c["free_first_mulligan"] = state['combos'][k]["mulligans"]["first_free"]
                    c["mulligan_type"] = state['combos'][k]["mulligans"].get("type", "traditional")
                
                for x_val in x_vals:
                    deck_size_calc = x_val if x_axis == "1" else fixed_val
                    hand_size_calc = x_val if x_axis == "2" else fixed_val
                    if hand_size_calc > deck_size_calc:
                        console.print(f"[error]Hand Size ({hand_size_calc}) cannot exceed Deck Size ({deck_size_calc}).[/error]")
                        pause()
                        return
                    cfg["deck_size"] = deck_size_calc
                    cfg["hand_size"] = hand_size_calc
                    result_calc = combo_set_probability(
                        cfg,
                        card_counts,
                        state["card_types"],
                        combos_input_calc,
                        extra_info=False
                    )
                    if isinstance(result_calc, dict):
                        probabilities.append(result_calc["Total"])
                    else:
                        probabilities.append(result_calc)
            
            all_probabilities.append((selected_name, selected_type, probabilities))
        
        fig, ax = plt.subplots(figsize=(width/150, height/150))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        for idx, (name, type_, probs) in enumerate(all_probabilities):
            ax.plot(x_vals, probs, marker=marker_style, linestyle=line_style, color=colors[idx % len(colors)], label=f"P({name})")
            probs_np: np.ndarray = np.array(probs)
            max_indices: np.ndarray = np.where(probs_np == probs_np.max())[0]
            min_indices: np.ndarray = np.where(probs_np == probs_np.min())[0]
            max_x: List[Any] = [x_vals[i] for i in max_indices]
            max_y: List[Any] = [probs[i] for i in max_indices]
            min_x: List[Any] = [x_vals[i] for i in min_indices]
            min_y: List[Any] = [probs[i] for i in min_indices]
            ax.scatter(max_x, max_y, color=colors[idx % len(colors)], marker='*', s=200, zorder=5)
            ax.scatter(min_x, min_y, color=colors[idx % len(colors)], marker='v', s=200, zorder=5)
        
        ax.set_xlabel(x_label, color='white')
        ax.set_ylabel("Probability", color='white')
        ax.tick_params(colors='white')
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.set_yticks(np.arange(0, 1.01, 0.02), minor=True)
        ax.grid(True, linestyle='--', alpha=0.5, color='white')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, 1)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_color('white')
        plt.title(f"Probability vs {x_label}", color='white')
        plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        console.print("[info]Export table to CSV? (y/n)[/info]")
        export_csv_str: str = console.input("[prompt]> [/prompt]").lower()
        export_csv: bool = export_csv_str == 'y'
        if export_csv:
            csv_filename: str = console.input("[prompt]Enter CSV filename (default: combo_table.csv) > [/prompt]")
            if not csv_filename:
                csv_filename = "combo_table.csv"
            if not csv_filename.endswith('.csv'):
                csv_filename += '.csv'
        
        table = Table(title=f"Probability vs {x_label}", show_header=True, header_style="bold white")
        table.add_column(x_label, style="bold")
        csv_data: List[List[str]] = [[x_label] + [f"P({name})" for name, _, _ in all_probabilities]]
        for name, _, _ in all_probabilities:
            table.add_column(f"P({name})", style="bold")
        for i, x_val in enumerate(x_vals):
            row: List[str] = [str(x_val)]
            for _, _, probs in all_probabilities:
                row.append(f"{probs[i]:.4f}")
            table.add_row(*row)
            csv_data.append(row)
        console.print(table)
        
        if export_csv:
            try:
                with open(csv_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([f"Probability vs {x_label}"])
                    writer.writerows(csv_data)
            except Exception as e:
                console.print(f"[error]Failed to write CSV file: {e}[/error]")
        
        console.print("[info]Save settings? (y/n)[/info]")
        if console.input("[prompt]> [/prompt]").lower() == 'y':
            settings = {
                'graph_type': graph_type,
                'line_style': line_style,
                'marker_style': marker_style,
                'item_idx': item_indices_input,
                'x_axis': x_axis,
                'x_min': x_min,
                'x_max': x_max,
                'fixed_val': fixed_val
            }
            try:
                with open('plot_settings.json', 'w') as f:
                    dump(settings, f)
                console.print("[info]Settings saved.[/info]")
            except Exception as e:
                    console.print(f"[error]Failed to save settings: {e}[/error]")
        
        plt.show()
    
    else:
        from scipy.stats import hypergeom
        
        def get_y(y_axis: str, k: int, N: int, K: int, n: int) -> float:
            if y_axis == "1":
                return float(hypergeom.pmf(k, N, K, n))
            if y_axis == "2":
                return 1 - float(hypergeom.pmf(k, N, K, n))
            if y_axis == "3":
                return float(hypergeom.cdf(k, N, K, n))
            if y_axis == "4":
                return 1 - float(hypergeom.cdf(k - 1, N, K, n))
            return 0.0
        
        def get_ylabel(y_axis: str) -> str:
            return {
                "1": "P(X = k)",
                "2": "P(X ≠ k)",
                "3": "P(X ≤ k)",
                "4": "P(X ≥ k)"
            }.get(y_axis, "Probability")
        
        console.print("[info]Select number of varying variables:[/info]")
        console.print("[info]1. One varying variable (line plot)[/info]")
        console.print("[info]2. Two varying variables (3D plot or heatmap)[/info]")
        num_vary: str = console.input("[prompt]> [/prompt]") if not load_settings or 'num_vary' not in settings else settings['num_vary']
        if num_vary not in ["1", "2"]:
            console.print("[error]Invalid choice.[/error]")
            pause()
            return
        
        console.print("[info]Set x-Axis (first varying parameter):[/info]")
        console.print("[info]1. Deck Size[/info]")
        console.print("[info]2. Hand Size[/info]")
        console.print("[info]3. Number of Success Cards[/info]")
        x_axis = console.input("[prompt]> [/prompt]") if not load_settings or 'x_axis' not in settings else settings['x_axis']
        if x_axis not in ["1", "2", "3"]:
            console.print("[error]Invalid x-axis choice.[/error]")
            pause()
            return
        x_label = {"1": "Deck Size", "2": "Hand Size", "3": "Number of Success Cards"}[x_axis]
        
        vary_options: Dict[str, str] = {
            "1": "Deck Size",
            "2": "Hand Size",
            "3": "Number of Success Cards"
        }
        valid_vary: Dict[str, str] = {k: v for k, v in vary_options.items() if k != x_axis}
        
        if num_vary == "1":
            console.print("[info]Which parameter to vary?[/info]")
            for key, label in valid_vary.items():
                console.print(f"[info]{key}. {label}[/info]")
            vary_choice: str = console.input("[prompt]> [/prompt]") if not load_settings or 'vary_choice' not in settings else settings['vary_choice']
            if vary_choice not in valid_vary:
                console.print("[error]Invalid vary choice.[/error]")
                pause()
                return
            vary_label: str = valid_vary[vary_choice]
            
            varying_vals_input: Optional[List[int]] = get_multiple_ints(f"[prompt]Enter {vary_label} values to vary (space or comma separated, >0) > [/prompt]")
            if not varying_vals_input:
                console.print("[error]Invalid varying values.[/error]")
                pause()
                return
            varying_vals: List[int] = varying_vals_input
            fixed_vals_keys: Set[str] = set(vary_options.keys()) - {x_axis, vary_choice}
            fixed_choice: str = fixed_vals_keys.pop()
            fixed_label = vary_options[fixed_choice]
            fixed_val: Optional[int] = get_int_input(f"[prompt]Set {fixed_label} (single value, >0) > [/prompt]", minimum=1)
            if fixed_val is None:
                console.print(f"[error]Invalid {fixed_label}.[/error]")
                pause()
                return
            min_prompt_dict: Dict[str, str] = {
                "1": "Min Deck Size (≥1)",
                "2": "Min Hand Size (≥1)",
                "3": "Min Number of Success Cards (≥1)"
            }
            max_prompt_dict: Dict[str, str] = {
                "1": "Max Deck Size (≥ Min)",
                "2": "Max Hand Size (≥ Min)",
                "3": "Max Number of Success Cards (≥ Min)"
            }
            x_min_input: Optional[int] = get_int_input(f"[prompt]{min_prompt_dict[x_axis]} > [/prompt]", minimum=1)
            if x_min_input is None:
                console.print("[error]Invalid min value.[/error]")
                pause()
                return
            x_min = x_min_input
            x_max_input: Optional[int] = get_int_input(f"[prompt]{max_prompt_dict[x_axis]} > [/prompt]", minimum=x_min)
            if x_max_input is None:
                console.print("[error]Invalid max value.[/error]")
                pause()
                return
            x_max = x_max_input
            
            console.print("[info]Set y-Axis[/info]")
            y_axis_options: Dict[str, str] = {
                "1": "P(X = k)",
                "2": "P(X ≠ k)",
                "3": "P(X ≤ k)",
                "4": "P(X ≥ k)"
            }
            
            for key, label in y_axis_options.items():
                console.print(f"[info]{key}. {label}[/info]")
            y_axis_choice: str = console.input("[prompt]> [/prompt]") if not load_settings or 'y_axis' not in settings else settings['y_axis']
            if y_axis_choice not in y_axis_options:
                console.print("[error]Invalid y-axis choice.[/error]")
                pause()
                return
            y_axis = y_axis_choice
            
            k_str: str = console.input("[prompt]Set k (≥0) > [/prompt]") if not load_settings or 'k' not in settings else str(settings['k'])
            if not (k_str.isdigit() and int(k_str) >= 0):
                console.print("[error]Invalid k.[/error]")
                pause()
                return
            k: int = int(k_str)
            
            x: np.ndarray = np.arange(x_min, x_max + 1)
            
            def get_param_value(param: str, val_x: int, val_vary: int, val_fixed: int) -> int:
                if param == x_axis:
                    return val_x
                elif param == vary_choice:
                    return val_vary
                else:
                    return val_fixed
            
            fig, ax = plt.subplots(figsize=(width/150, height/150))
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            
            for i, v in enumerate(varying_vals):
                y_vals: List[float] = []
                for val_x in x:
                    N: int = get_param_value("1", val_x, v, fixed_val)
                    K: int = get_param_value("3", val_x, v, fixed_val)
                    n: int = get_param_value("2", val_x, v, fixed_val)
                    if n > N or K > N:
                        console.print(f"[error]Invalid: Hand Size ({n}) or Success Cards ({K}) > Deck Size ({N})[/error]")
                        pause()
                        return
                    y_vals.append(get_y(y_axis, k, N, K, n))
                
                y_vals_np: np.ndarray = np.array(y_vals)
                max_indices = np.where(y_vals_np == y_vals_np.max())[0]
                min_indices = np.where(y_vals_np == y_vals_np.min())[0]
                max_x_plot: List[Any] = [x[i] for i in max_indices]
                max_y_plot: List[Any] = [y_vals[i] for i in max_indices]
                min_x_plot: List[Any] = [x[i] for i in min_indices]
                min_y_plot: List[Any] = [y_vals[i] for i in min_indices]
                if x_min == 1 and x_axis == "3":
                    x = np.append(0, x)
                    if y_axis == "1":
                        y_vals.insert(0, 1.0 if k == 0 else 0.0)
                    elif y_axis == "2":
                        y_vals.insert(0, 0.0 if k == 0 else 1.0)
                    elif y_axis == "3":
                        y_vals.insert(0, 1.0)
                    elif y_axis == "4":
                        y_vals.insert(0, 1.0 if k == 0 else 0.0)
                ax.plot(x, y_vals, marker=marker_style, linestyle=line_style, color=colors[i % len(colors)], label=f"{vary_label} = {v}")
                ax.scatter(max_x_plot, max_y_plot, color=colors[i % len(colors)], marker='*', s=200, zorder=5)
                ax.scatter(min_x_plot, min_y_plot, color=colors[i % len(colors)], marker='v', s=200, zorder=5)
            
            ax.set_xlabel(x_label, color='white')
            ax.set_ylabel(get_ylabel(y_axis), color='white')
            ax.tick_params(colors='white')
            ax.set_yticks(np.arange(0, 1.01, 0.1))
            ax.set_yticks(np.arange(0, 1.01, 0.02), minor=True)
            ax.grid(True, linestyle='--', alpha=0.5, color='white')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(0, 1)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            
            legend = ax.legend()
            for text in legend.get_texts():
                text.set_color('white')
            plt.title(f"Probability vs {x_label}", color='white')
            plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            console.print("[info]Export tables to CSV? (y/n)[/info]")
            export_csv_str = console.input("[prompt]> [/prompt]").lower()
            export_csv = export_csv_str == 'y'
            if export_csv:
                csv_filename = console.input("[prompt]Enter CSV filename (default: hypergeometric_table.csv) > [/prompt]")
                if not csv_filename:
                    csv_filename = "hypergeometric_table.csv"
                if not csv_filename.endswith('.csv'):
                    csv_filename += '.csv'
            
            for v_val in varying_vals:
                table = Table(title=f"{vary_label} = {v_val}", show_header=True, header_style="bold white")
                table.add_column(x_label, style="bold")
                table.add_column(get_ylabel(y_axis), style="bold")
                csv_data = [[x_label, get_ylabel(y_axis)]]
                
                for j, val_x in enumerate(x):
                    N = get_param_value("1", val_x, v_val, fixed_val)
                    K = get_param_value("3", val_x, v_val, fixed_val)
                    n = get_param_value("2", val_x, v_val, fixed_val)
                    val_y = get_y(y_axis, k, N, K, n)
                    table.add_row(str(val_x), f"{val_y:.4f}")
                    csv_data.append([str(val_x), f"{val_y:.4f}"])
                console.print(table)
                
                if export_csv:
                    try:
                        with open(f"{vary_label}_{v_val}_{csv_filename}", 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([f"{vary_label} = {v_val}"])
                            writer.writerows(csv_data)
                    except Exception as e:
                        console.print(f"[error]Failed to write CSV file: {e}[/error]")
            
            console.print("[info]Save settings? (y/n)[/info]")
            if console.input("[prompt]> [/prompt]").lower() == 'y':
                settings = {
                    'graph_type': graph_type,
                    'line_style': line_style,
                    'marker_style': marker_style,
                    'num_vary': num_vary,
                    'x_axis': x_axis,
                    'vary_choice': vary_choice,
                    'varying_vals': varying_vals,
                    'fixed_val': fixed_val,
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_axis': y_axis,
                    'k': k
                }
                try:
                    with open('plot_settings.json', 'w') as f:
                        dump(settings, f)
                    console.print("[info]Settings saved.[/info]")
                except Exception as e:
                    console.print(f"[error]Failed to save settings: {e}[/error]")
            
            plt.show()
        
        else:
            console.print("[info]Select second varying parameter (y-axis):[/info]")
            for key, label in valid_vary.items():
                console.print(f"[info]{key}. {label}[/info]")
            y_axis_var: str = console.input("[prompt]> [/prompt]") if not load_settings or 'y_axis_var' not in settings else settings['y_axis_var']
            if y_axis_var not in valid_vary:
                console.print("[error]Invalid y-axis choice.[/error]")
                pause()
                return
            y_label = valid_vary[y_axis_var]
            
            fixed_vals_keys = set(vary_options.keys()) - {x_axis, y_axis_var}
            fixed_choice = fixed_vals_keys.pop()
            fixed_label = vary_options[fixed_choice]
            fixed_val = get_int_input(f"[prompt]Set {fixed_label} (single value, >0) > [/prompt]", minimum=1)
            if fixed_val is None:
                console.print(f"[error]Invalid {fixed_label}.[/error]")
                pause()
                return
            
            min_prompt_dict = {
                "1": "Min Deck Size (≥1)",
                "2": "Min Hand Size (≥1)",
                "3": "Min Number of Success Cards (≥1)"
            }
            max_prompt_dict = {
                "1": "Max Deck Size (≥ Min)",
                "2": "Max Hand Size (≥ Min)",
                "3": "Max Number of Success Cards (≥ Min)"
            }
            x_min_input = get_int_input(f"[prompt]{min_prompt_dict[x_axis]} > [/prompt]", minimum=1)
            if x_min_input is None:
                console.print("[error]Invalid x min value.[/error]")
                pause()
                return
            x_min = x_min_input
            x_max_input = get_int_input(f"[prompt]{max_prompt_dict[x_axis]} > [/prompt]", minimum=x_min)
            if x_max_input is None:
                console.print("[error]Invalid x max value.[/error]")
                pause()
                return
            x_max = x_max_input
            y_min_input = get_int_input(f"[prompt]{min_prompt_dict[y_axis_var]} > [/prompt]", minimum=1)
            if y_min_input is None:
                console.print("[error]Invalid y min value.[/error]")
                pause()
                return
            y_min = y_min_input
            y_max_input = get_int_input(f"[prompt]{max_prompt_dict[y_axis_var]} > [/prompt]", minimum=y_min)
            if y_max_input is None:
                console.print("[error]Invalid y max value.[/error]")
                pause()
                return
            y_max = y_max_input
            
            console.print("[info]Select plot type:[/info]")
            console.print("[info]1. 3D Surface Plot[/info]")
            console.print("[info]2. Heatmap[/info]")
            plot_type: str = console.input("[prompt]> [/prompt]") if not load_settings or 'plot_type' not in settings else settings['plot_type']
            if plot_type not in ["1", "2"]:
                console.print("[error]Invalid plot type.[/error]")
                pause()
                return
            
            console.print("[info]Set probability type:[/info]")
            y_axis_options = {
                "1": "P(X = k)",
                "2": "P(X ≠ k)",
                "3": "P(X ≤ k)",
                "4": "P(X ≥ k)"
            }
            for key, label in y_axis_options.items():
                console.print(f"[info]{key}. {label}[/info]")
            y_axis_prob_type: str = console.input("[prompt]> [/prompt]") if not load_settings or 'y_axis' not in settings else settings['y_axis']
            if y_axis_prob_type not in y_axis_options:
                console.print("[error]Invalid probability type.[/error]")
                pause()
                return
            y_axis = y_axis_prob_type
            k_str = console.input("[prompt]Set k (≥0) > [/prompt]") if not load_settings or 'k' not in settings else str(settings['k'])
            if not (k_str.isdigit() and int(k_str) >= 0):
                console.print("[error]Invalid k.[/error]")
                pause()
                return
            k = int(k_str)
            
            x_vals_np: np.ndarray = np.arange(x_min, x_max + 1)
            y_vals_np: np.ndarray = np.arange(y_min, y_max + 1)
            X, Y = np.meshgrid(x_vals_np, y_vals_np)
            Z: np.ndarray = np.zeros(X.shape)
            
            def get_param_value_3d(param: str, val_x: int, val_y: int, val_fixed: int) -> int:
                if param == x_axis:
                    return val_x
                elif param == y_axis_var:
                    return val_y
                else:
                    return val_fixed
            
            for i in range(len(y_vals_np)):
                for j in range(len(x_vals_np)):
                    N = get_param_value_3d("1", x_vals_np[j], y_vals_np[i], fixed_val)
                    K = get_param_value_3d("3", x_vals_np[j], y_vals_np[i], fixed_val)
                    n = get_param_value_3d("2", x_vals_np[j], y_vals_np[i], fixed_val)
                    if n > N or K > N:
                        console.print(f"[error]Invalid: Hand Size ({n}) or Success Cards ({K}) > Deck Size ({N})[/error]")
                        pause()
                        return
                    Z[i, j] = get_y(y_axis, k, N, K, n)
            
            fig = plt.figure(figsize=(width/150, height/150))
            fig.patch.set_facecolor('black')
            
            if plot_type == "1":
                ax = fig.add_subplot(111, projection='3d')
                ax.set_facecolor('black')
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', rstride=1, cstride=1, shade=True)
                ax.view_init(elev=30, azim=45)
                ax.set_xlabel(x_label, color='white')
                ax.set_ylabel(y_label, color='white')
                ax.set_zlabel(get_ylabel(y_axis), color='white')
                ax.tick_params(colors='white')
                ax.set_zticks(np.arange(0, 1.01, 0.1))
                ax.set_zticks(np.arange(0, 1.01, 0.02), minor=True)
                ax.xaxis.line.set_color('white')
                ax.yaxis.line.set_color('white')
                ax.zaxis.line.set_color('white')
                ax.xaxis.set_pane_color((0, 0, 0, 1))
                ax.yaxis.set_pane_color((0, 0, 0, 1))
                ax.zaxis.set_pane_color((0, 0, 0, 1))
                fig.colorbar(surf, ax=ax, label='Probability', pad=0.1, shrink=0.8).ax.yaxis.set_tick_params(color='white', labelcolor='white')
                plt.title(f"Probability vs {x_label} and {y_label}", color='white')
            
            else:
                ax = fig.add_subplot(111)
                ax.set_facecolor('black')
                im = ax.imshow(Z, cmap='viridis', origin='lower', extent=[x_min, x_max, y_min, y_max])
                ax.set_xlabel(x_label, color='white')
                ax.set_ylabel(y_label, color='white')
                ax.tick_params(colors='white')
                ax.set_xticks(np.arange(x_min, x_max + 1))
                ax.set_yticks(np.arange(y_min, y_max + 1))
                fig.colorbar(im, ax=ax, label='Probability').ax.yaxis.set_tick_params(color='white', labelcolor='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_linewidth(1.5)
                ax.spines['left'].set_linewidth(1.5)
                
                max_indices = np.where(Z == Z.max())
                min_indices = np.where(Z == Z.min())
                for i, j in zip(max_indices[0], max_indices[1]):
                    ax.text(x_vals_np[j], y_vals_np[i], 'Max', color='white', ha='center', va='center', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
                for i, j in zip(min_indices[0], min_indices[1]):
                    ax.text(x_vals_np[j], y_vals_np[i], 'Min', color='white', ha='center', va='center', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
                
                plt.title(f"Probability vs {x_label} and {y_label}", color='white')
            
            plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            console.print("[info]Export table to CSV? (y/n)[/info]")
            export_csv_str = console.input("[prompt]> [/prompt]").lower()
            export_csv = export_csv_str == 'y'
            if export_csv:
                csv_filename = console.input("[prompt]Enter CSV filename (default: hypergeometric_table.csv) > [/prompt]")
                if not csv_filename:
                    csv_filename = "hypergeometric_table.csv"
                if not csv_filename.endswith('.csv'):
                    csv_filename += '.csv'
            
            table = Table(title=f"Probability ({get_ylabel(y_axis)})", show_header=True, header_style="bold white")
            table.add_column(x_label, style="bold")
            csv_data = [[x_label] + [f"{y_label} = {y_val}" for y_val in y_vals_np]]
            for y_val in y_vals_np:
                table.add_column(f"{y_label} = {y_val}", style="bold")
            for j, x_val in enumerate(x_vals_np):
                row = [str(x_val)]
                for i in range(len(y_vals_np)):
                    row.append(f"{Z[i, j]:.4f}")
                table.add_row(*row)
                csv_data.append(row)
            console.print(table)
            
            if export_csv:
                try:
                    with open(csv_filename, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([f"Hypergeometric Probability ({get_ylabel(y_axis)})"])
                        writer.writerows(csv_data)
                except Exception as e:
                    console.print(f"[error]Failed to write CSV file: {e}[/error]")
            
            console.print("[info]Save settings? (y/n)[/info]")
            if console.input("[prompt]> [/prompt]").lower() == 'y':
                settings = {
                    'graph_type': graph_type,
                    'line_style': line_style,
                    'marker_style': marker_style,
                    'num_vary': num_vary,
                    'x_axis': x_axis,
                    'y_axis_var': y_axis_var,
                    'fixed_val': fixed_val,
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                    'plot_type': plot_type,
                    'y_axis': y_axis,
                    'k': k
                }
                try:
                    with open('plot_settings.json', 'w') as f:
                        dump(settings, f)
                    console.print("[info]Settings saved.[/info]")
                except Exception as e:
                    console.print(f"[error]Failed to save settings: {e}[/error]")
            
            plt.show()
    
    pause()

    
def page_save() -> None:
    clear_screen()
    console.print("[header][7] Save Deck State[/header]")
    path: str = console.input("[prompt]Enter filename to save (e.g., deck.json): [/prompt]")
    try:
        with open(path, "w") as f:
            dump(state, f, indent=4)
        console.print(f"[success]State saved to {path}.[/success]")
    except Exception as e:
        console.print(f"[error]Error saving file: {e}[/error]")
    pause()

def page_load(from_first_screen: bool = False) -> None:
    clear_screen()
    if not from_first_screen:
        console.print("[header][8] Load Deck State[/header]")
    else:
        console.print("[header]Load Deck State[/header]")
    files: List[str] = [f for f in listdir() if f.endswith(".json") and isfile(f)]

    if files:
        console.print("[info]Available .json files in current directory:[/info]")
        for i, fname in enumerate(files):
            console.print(f"[option]{i + 1}[/option]: {fname}")
    else:
        console.print("[info]No .json files found in current directory.[/info]")
    console.print("[prompt]Enter the number of the file to load, or enter a path to a json:[/prompt]")
    while True:
        user_input: str = console.input("[prompt]> [/prompt]").strip()
        if user_input == "":
            return
        if user_input.isdigit():
            idx: int = int(user_input) - 1
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
            loaded: Dict[str, Any] = load(f)
        required: Set[str] = {"deck_size", "hand_size", "cards", "card_types", "combos", "combo_sets"}
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
def page_list() -> None:
    clear_screen()
    console.print("[header][9] State Summary[/header]\n")
    console.print(f"[info]Deck size: {state['deck_size']}[/info]")
    console.print(f"[info]Hand size: {state['hand_size']}[/info]\n")
    console.print("[info]Cards:[/info]")
    for card, qty in state['cards'].items():
        console.print(f"[info]  {card}: {qty}[/info]")
    console.print("\n[info]Card Types:[/info]")
    for t, members in state['card_types'].items():
        console.print(f"[info]  {t}: {members}[/info]")
    console.print("\n[info]Combos:[/info]")
    for name, combo in state['combos'].items():
        console.print(f"[info]  {name}: {combo}[/info]")
    console.print("\n[info]Combo Sets:[/info]")
    for name, expr in state['combo_sets'].items():
        console.print(f"[info]  {name}: {expr}[/info]")
    pause()

def page_exit() -> None:
    clear_screen()
    console.print("[header]Exiting program.[/header]")
    exit()

def main() -> None:
    import_deck_prompt()
    while True:
        clear_screen()
        console.print("[header]Main Page[/header]\n")
        console.print("[info]1.[/info] [text]Deck and Hand Size[/text]")
        console.print("[info]2.[/info] [text]Cards and Quantities[/text]")
        console.print("[info]3.[/info] [text]Card Types[/text]")
        console.print("[info]4.[/info] [text]Combo Requirements[/text]")
        console.print("[info]5.[/info] [text]Combo Sets[/text]")
        console.print("[info]6.[/info] [text]Calculate Probability[/text]")
        console.print("[info]7.[/info] [text]Save to File[/text]")
        console.print("[info]8.[/info] [text]Load Deck/JSON[/text]")
        console.print("[info]9.[/info] [text]Show All State[/text]")
        console.print("[info]10.[/info] [text]Graphs[text]")
        console.print("[info]11.[/info] [text]Exit[text]")
        choice: str = console.input("[prompt]> [/prompt]")
        if choice == "1": page_deck_hand_size()
        elif choice == "2": page_cards()
        elif choice == "3": page_card_types()
        elif choice == "4": page_combos()
        elif choice == "5": page_combo_sets()
        elif choice == "6": page_calculate_probability()
        elif choice == "7": page_save()
        elif choice == "8": import_deck_prompt(from_first_screen=False)
        elif choice == "9": page_list()
        elif choice == "10": page_graph()
        elif choice == "11": page_exit()
        else:
            console.print("[error]Invalid option.[/error]")
            pause()

if __name__ == "__main__":
    main()


def update_card_type_cache():
    state["card_type_cache"].clear()
    for card_type in state["card_types"]:
        state["card_type_cache"][card_type] = sum(state["cards"].get(card, 0) for card in state["card_types"].get(card_type, []))

def get_card_type_quantity(card_type_name):
    if card_type_name in state["card_type_cache"]:
        return state["card_type_cache"][card_type_name]
    qty = sum(state["cards"].get(card, 0) for card in state["card_types"].get(card_type_name, []))
    state["card_type_cache"][card_type_name] = qty
    return qty

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


def import_ydk_file(path):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        ids = []
        in_main = False
        for line in lines:
            line = line.strip()
            if line == "#main": in_main = True; continue
            if line.startswith("#"): in_main = False
            if in_main and line.isdigit(): ids.append(int(line))
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
    if choice == "1":
        file_path = console.input("[prompt]Input the file location of the YDK file: [/prompt]")
        import_ydk_file(file_path)
    elif choice == "2":
        import_ydke_link()
    else:
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
    if choice == "1":
        import_deck_ygo()
    elif choice == "2":
        page_load(from_first_screen=from_first_screen)
    elif choice == "3" or choice == "":
        return
    else:
        console.print("[error]Invalid option.[/error]")
        pause()
        import_deck_prompt()

state = {
    "deck_size": None,
    "hand_size": None,
    "cards": {},
    "card_types": {},
    "combos": {},
    "combo_sets": {},
    "card_type_cache": {},
}

def page_deck_hand_size():
    while True:
        clear_screen()
        console.print("[header][1] Deck and Hand Size[/header]\n")
        console.print(f"[info]Current deck size: {state['deck_size']}[/info]")
        console.print(f"[info]Current hand size: {state['hand_size']}[/info]\n")
        console.print("[info]1. Set Deck Size[/info]")
        console.print("[info]2. Set Hand Size[/info]")
        console.print("[info]3. Back[/info]")
        choice = console.input("[prompt]> [/prompt]")
        if choice == "1":
            val = console.input("[prompt]Enter new deck size: [/prompt]")
            if val.isdigit() and int(val) > 0:
                old_deck_size = state['deck_size']
                state['deck_size'] = int(val)
                for combo in state['combos'].values():
                    if combo["deck_size"] == old_deck_size:
                        combo["deck_size"] = int(val)
                for combo_set in state['combo_sets'].values():
                    if combo_set["deck_size"] == old_deck_size:
                        combo_set["deck_size"] = int(val)
            else:
                console.print("[error]Invalid deck size.[/error]")
                pause()
        elif choice == "2":
            old_hand_size = state['hand_size']
            val = console.input("[prompt]Enter new hand size: [/prompt]")
            if val.isdigit() and int(val) > 0:
                state['hand_size'] = int(val)
                for combo in state['combos'].values():
                    if combo["hand_size"] == old_hand_size:
                        combo["hand_size"] = int(val)
                for combo_set in state['combo_sets'].values():
                    if combo_set["hand_size"] == old_hand_size:
                        combo_set["hand_size"] = int(val)
            else:
                console.print("[error]Invalid hand size.[/error]")
                pause()
        elif choice == "3" or choice == "":
            return
        else:
            console.print("[error]Invalid option.[/error]")
            pause()

def page_cards():
    if state['deck_size'] is None:
        console.print("[error]Deck size must be set first.[/error]")
        pause()
        return
    while True:
        clear_screen()
        console.print("[header][2] Cards and Quantities[/header]\n")
        total = sum(state['cards'].values())
        for i, (card, qty) in enumerate(state['cards'].items()):
            console.print(f"[info][{i}] {card}: {qty}[/info]")
        console.print(f"[info]Unspecified cards: {state['deck_size'] - total}[/info]\n")
        console.print("[info]1. Add a Card[/info]")
        console.print("[info]2. Edit/Remove a Card[/info]")
        console.print("[info]3. Back[/info]")
        choice = console.input("[prompt]> [/prompt]")
        if choice == "1":
            name = console.input("[prompt]Card name: [/prompt]")
            qty = console.input("[prompt]Quantity: [/prompt]")
            if not qty.isdigit() or int(qty) < 0:
                console.print("[error]Invalid quantity.[/error]")
                pause()
                continue
            qty = int(qty)
            if sum(state['cards'].values()) + qty > state['deck_size']:
                console.print("[error]Card quantities exceed deck size. Check your input.[/error]")
                pause()
                continue
            state['cards'][name] = qty
            update_card_type_cache()
        elif choice == "2":
            index = console.input("[prompt]Enter card index to edit/remove: [/prompt]")
            if not index.isdigit() or int(index) >= len(state['cards']):
                console.print("[error]Invalid index.[/error]")
                pause()
                continue
            card = list(state['cards'].keys())[int(index)]
            console.print(f"[info]Selected: {card} (current qty: {state['cards'][card]})[/info]")
            new_val = console.input("[prompt]New quantity (or empty to remove): [/prompt]")
            if new_val == "":
                del state['cards'][card]
                update_card_type_cache()
            elif new_val.isdigit() and int(new_val) >= 0:
                qty = int(new_val)
                if sum(state['cards'].values()) - state['cards'][card] + qty > state['deck_size']:
                    console.print("[error]Card quantities exceed deck size.[/error]")
                    pause()
                    continue
                state['cards'][card] = qty
                update_card_type_cache()
            else:
                console.print("[error]Invalid quantity.[/error]")
                pause()
        elif choice == "3" or choice == "":
            return
        else:
            console.print("[error]Invalid option.[/error]")
            pause()

def page_card_types():
    if not state['cards']:
        console.print("[error]Cards must be added first.[/error]")
        pause()
        return
    while True:
        clear_screen()
        console.print("[header][3] Card Types[/header]\n")
        for i, (ctype, cards) in enumerate(state['card_types'].items()):
            console.print(f"[info][{i}] {ctype}: {cards}[/info]")
        console.print("\n[info]1. Add Card Type[/info]")
        console.print("[info]2. Edit/Remove Card Type[/info]")
        console.print("[info]3. Back[/info]")
        choice = console.input("[prompt]> [/prompt]")
        if choice == "1":
            name = console.input("[prompt]Card Type name: [/prompt]")
            if name in state['card_types']:
                console.print("[error]Card Type already exists.[/error]")
                pause()
                continue
            available_cards = list(state['cards'].keys())
            console.print("[info]Available Cards:[/info]")
            for i, card in enumerate(available_cards):
                console.print(f"[info][{i}] {card}[/info]")
            indexes = console.input("[prompt]Enter indices of cards to add (comma-separated): [/prompt]")
            try:
                indices = list(map(int, indexes.split(",")))
                state['card_types'][name] = list({available_cards[i] for i in indices if 0 <= i < len(available_cards)})
                update_card_type_cache()
            except:
                console.print("[error]Invalid input.[/error]")
                pause()
        elif choice == "2":
            idx = console.input("[prompt]Enter index of Card Type to edit/remove: [/prompt]")
            if not idx.isdigit() or int(idx) >= len(state['card_types']):
                console.print("[error]Invalid index.[/error]")
                pause()
                continue
            ctype_name = list(state['card_types'].keys())[int(idx)]
            console.print(f"[info]Selected: {ctype_name}[/info]")
            console.print("[info]1. Edit[/info]")
            console.print("[info]2. Remove[/info]")
            sub = console.input("[prompt]> [/prompt]")
            if sub == "1":
                available_cards = list(state['cards'].keys())
                console.print("[info]Available Cards:[/info]")
                for i, card in enumerate(available_cards):
                    console.print(f"[info][{i}] {card}[/info]")
                indexes = console.input("[prompt]Enter indices of cards to set (comma-separated): [/prompt]")
                try:
                    indices = list(map(int, indexes.split(",")))
                    state['card_types'][ctype_name] = list({available_cards[i] for i in indices if 0 <= i < len(available_cards)})
                    update_card_type_cache()
                except:
                    console.print("[error]Invalid input.[/error]")
                    pause()
            elif sub == "2":
                del state['card_types'][ctype_name]
                update_card_type_cache()
            else:
                console.print("[error]Invalid option.[/error]")
                pause()
        elif choice == "3" or choice == "":
            return
        else:
            console.print("[error]Invalid option.[/error]")
            pause()

def page_combos():
    clear_screen()
    console.print("[header][4] Combos[/header]\n")

    if not state["card_types"] and not state["cards"]:
        console.print("[error]You must define card types or add cards before adding combos.[/error]")
        pause()
        return

    console.print("[info]Defined combos:[/info]")
    for i, name in enumerate(state["combos"]):
        console.print(f"[info]{i}. {name}[/info]")

    console.print("\n[info]1. Add Combo[/info]")
    console.print("[info]2. Edit/Remove Combo[/info]")
    console.print("[info]3. Back[/info]")
    choice = console.input("[prompt]> [/prompt]")

    if choice == "1":
        name = console.input("[prompt]Enter combo name: [/prompt]").strip()
        if not name:
            console.print("[error]Invalid name.[/error]")
            pause()
            return

        console.print("[info]Available card types:[/info]")
        card_type_keys = list(state["card_types"].keys())
        for i, ct in enumerate(card_type_keys):
            console.print(f"[info]T{i}. {ct}[/info]")
        console.print("[info]Available cards:[/info]")
        card_keys = list(state["cards"].keys())
        for i, card in enumerate(card_keys):
            console.print(f"[info]C{i}. {card}[/info]")
        selected = console.input("[prompt]Enter indices of card types (T#) or cards (C#) to include (comma-separated): [/prompt]")
        try:
            selected_indices = [i.strip() for i in selected.split(",")]
        except:
            console.print("[error]Invalid input.[/error]")
            pause()
            return

        combo_requirements = {}
        for idx in selected_indices:
            if not (idx.startswith("T") or idx.startswith("C")):
                console.print(f"[error]Invalid index format: {idx}[/error]")
                pause()
                return
            is_card_type = idx.startswith("T")
            idx_num = idx[1:]
            if not idx_num.isdigit():
                console.print(f"[error]Invalid index number: {idx}[/error]")
                pause()
                return
            idx_num = int(idx_num)
            if is_card_type:
                if idx_num >= len(card_type_keys):
                    console.print(f"[error]Invalid card type index: {idx}[/error]")
                    pause()
                    return
                item_name = card_type_keys[idx_num]
                max_available = get_card_type_quantity(item_name)
            else:
                if idx_num >= len(card_keys):
                    console.print(f"[error]Invalid card index: {idx}[/error]")
                    pause()
                    return
                item_name = card_keys[idx_num]
                max_available = state["cards"].get(item_name, 0)

            console.print(f"\n[info]{'Card Type' if is_card_type else 'Card'}: {item_name} (Total quantity in deck: {max_available})[/info]")
            try:
                min_ct = console.input("[prompt]Minimum required (default 1): [/prompt]").strip()
                max_ct = console.input(f"[prompt]Maximum allowed (default {max_available}): [/prompt]").strip()
                min_ct = int(min_ct) if min_ct else 1
                max_ct = int(max_ct) if max_ct else max_available
                if max_ct > max_available:
                    console.print(f"[error]Max cannot exceed total available ({max_available}).[/error]")
                    pause()
                    return
                if min_ct > max_ct:
                    console.print("[error]Minimum cannot exceed maximum.[/error]")
                    pause()
                    return
                combo_requirements[item_name] = (min_ct, max_ct)
            except:
                console.print("[error]Invalid input for min/max.[/error]")
                pause()
                return

        console.print("\n[prompt]Use custom deck/hand size? (y/N): [/prompt]", end="")
        custom = console.input().lower().strip() == "y"
        deck_size = state["deck_size"]
        hand_size = state["hand_size"]
        if custom:
            try:
                deck_size = int(console.input(f"[prompt]Custom deck size (default {deck_size}): [/prompt]") or deck_size)
                hand_size = int(console.input(f"[prompt]Custom hand size (default {hand_size}): [/prompt]") or hand_size)
                if deck_size <= 0 or hand_size <= 0:
                    console.print("[error]Deck and hand sizes must be positive.[/error]")
                    pause()
                    return
            except:
                console.print("[error]Invalid custom size.[/error]")
                pause()
                return

        console.print("\n[prompt]Include mulligans? (y/N): [/prompt]", end="")
        include_mulligans = console.input().lower().strip() == "y"
        mulligans = {
            "enabled": include_mulligans,
            "count": 0,
            "first_free": False,
            "type": "traditional"
        }
        if include_mulligans:
            console.print("\n[info]Select mulligan type:[/info]")
            console.print("[info]1. Traditional (redraw smaller hand)[/info]")
            console.print("[info]2. London (draw full hand, return cards)[/info]")
            console.print("[info]3. Partial (replace specific cards)[/info]")
            mulligan_choice = console.input("[prompt]> [/prompt]").strip()
            if mulligan_choice == "1":
                mulligans["type"] = "traditional"
            elif mulligan_choice == "2":
                mulligans["type"] = "london"
            elif mulligan_choice == "3":
                mulligans["type"] = "partial"
            else:
                console.print("[error]Invalid mulligan type. Defaulting to Traditional.[/error]")
                mulligans["type"] = "traditional"

            default_mull_count = hand_size - sum([v[0] for v in combo_requirements.values()])
            try:
                count = console.input(f"[prompt]Number of mulligans (default {default_mull_count}): [/prompt]").strip()
                mulligans["count"] = int(count) if count else default_mull_count
                if mulligans["count"] < 0:
                    console.print("[error]Mulligan count cannot be negative.[/error]")
                    pause()
                    return
            except:
                console.print("[error]Invalid mulligan count.[/error]")
                pause()
                return
            first = console.input("[prompt]Is the first mulligan free? (y/N): [/prompt]").strip().lower()
            mulligans["first_free"] = first == "y"

        state["combos"][name] = {
            "requirements": combo_requirements,
            "deck_size": deck_size,
            "hand_size": hand_size,
            "mulligans": mulligans
        }
        console.print(f"[success]Combo '{name}' added.[/success]")
        pause()

    elif choice == "2":
        if not state["combos"]:
            console.print("[error]No combos to edit.[/error]")
            pause()
            return
        try:
            idx = int(console.input("[prompt]Enter index of combo to edit: [/prompt]"))
            name = list(state["combos"].keys())[idx]
        except:
            console.print("[error]Invalid index.[/error]")
            pause()
            return
        console.print("[info]1. Edit[/info]")
        console.print("[info]2. Remove[/info]")
        sub = console.input("[prompt]> [/prompt]")
        if sub == "1":
            combo = state["combos"][name]
            console.print(f"\n[info]Editing combo '{name}':[/info]")

            change_ct = console.input("[prompt]Do you want to change the card types/cards in the combo? (y/N): [/prompt]").strip().lower()
            if change_ct == "y":
                combo_requirements = {}
                card_type_keys = list(state["card_types"].keys())
                card_keys = list(state["cards"].keys())
                console.print("[info]Available card types:[/info]")
                for i, ct in enumerate(card_type_keys):
                    console.print(f"[info]T{i}. {ct}[/info]")
                console.print("[info]Available cards:[/info]")
                for i, card in enumerate(card_keys):
                    console.print(f"[info]C{i}. {card}[/info]")
                selected = console.input("[prompt]Enter indices of card types (T#) or cards (C#) to include (comma-separated): [/prompt]")
                try:
                    selected_indices = [i.strip() for i in selected.split(",")]
                except:
                    console.print("[error]Invalid input.[/error]")
                    pause()
                    return

                for idx in selected_indices:
                    if not (idx.startswith("T") or idx.startswith("C")):
                        console.print(f"[error]Invalid index format: {idx}[/error]")
                        pause()
                        return
                    is_card_type = idx.startswith("T")
                    idx_num = idx[1:]
                    if not idx_num.isdigit():
                        console.print(f"[error]Invalid index number: {idx}[/error]")
                        pause()
                        return
                    idx_num = int(idx_num)
                    if is_card_type:
                        if idx_num >= len(card_type_keys):
                            console.print(f"[error]Invalid card type index: {idx}[/error]")
                            pause()
                            return
                        item_name = card_type_keys[idx_num]
                        max_available = get_card_type_quantity(item_name)
                    else:
                        if idx_num >= len(card_keys):
                            console.print(f"[error]Invalid card index: {idx}[/error]")
                            pause()
                            return
                        item_name = card_keys[idx_num]
                        max_available = state["cards"].get(item_name, 0)

                    console.print(f"\n[info]{'Card Type' if is_card_type else 'Card'}: {item_name} (Total quantity in deck: {max_available})[/info]")
                    try:
                        min_ct = console.input("[prompt]Minimum required (default 1): [/prompt]").strip()
                        max_ct = console.input(f"[prompt]Maximum allowed (default {max_available}): [/prompt]").strip()
                        min_ct = int(min_ct) if min_ct else 1
                        max_ct = int(max_ct) if max_ct else max_available
                        if max_ct > max_available:
                            console.print(f"[error]Max cannot exceed total available ({max_available}).[/error]")
                            pause()
                            return
                        if min_ct > max_ct:
                            console.print("[error]Minimum cannot exceed maximum.[/error]")
                            pause()
                            return
                        combo_requirements[item_name] = (min_ct, max_ct)
                    except:
                        console.print("[error]Invalid input for min/max.[/error]")
                        pause()
                        return
                combo["requirements"] = combo_requirements

            change_mull = console.input("[prompt]Do you want to change the mulligan settings? (y/N): [/prompt]").strip().lower()
            if change_mull == "y":
                include_mulligans = console.input("[prompt]Include mulligans? (y/N): [/prompt]").strip().lower() == "y"
                mulligans = {
                    "enabled": include_mulligans,
                    "count": 0,
                    "first_free": False,
                    "type": combo["mulligans"].get("type", "traditional")
                }
                if include_mulligans:
                    console.print("\n[info]Select mulligan type:[/info]")
                    console.print("[info]1. Traditional (redraw smaller hand)[/info]")
                    console.print("[info]2. London (draw full hand, return cards)[/info]")
                    console.print("[info]3. Partial (replace specific cards)[/info]")
                    mulligan_choice = console.input("[prompt]> [/prompt]").strip()
                    if mulligan_choice == "1":
                        mulligans["type"] = "traditional"
                    elif mulligan_choice == "2":
                        mulligans["type"] = "london"
                    elif mulligan_choice == "3":
                        mulligans["type"] = "partial"
                    else:
                        console.print("[error]Invalid mulligan type. Defaulting to Traditional.[/error]")
                        mulligans["type"] = "traditional"

                    default_mull_count = combo["hand_size"] - sum([v[0] for v in combo["requirements"].values()])
                    try:
                        count = console.input(f"[prompt]Number of mulligans (default {default_mull_count}): [/prompt]").strip()
                        mulligans["count"] = int(count) if count else default_mull_count
                        if mulligans["count"] < 0:
                            console.print("[error]Mulligan count cannot be negative.[/error]")
                            pause()
                            return
                    except:
                        console.print("[error]Invalid mulligan count.[/error]")
                        pause()
                        return
                    first = console.input("[prompt]Is the first mulligan free? (y/N): [/prompt]").strip().lower()
                    mulligans["first_free"] = first == "y"
                combo["mulligans"] = mulligans

            change_sizes = console.input("[prompt]Do you want to change the deck size and hand size for this combo? (y/N): [/prompt]").strip().lower()
            if change_sizes == "y":
                try:
                    deck_size = console.input(f"[prompt]Custom deck size (default {combo.get('deck_size', state['deck_size'])}): [/prompt]").strip()
                    hand_size = console.input(f"[prompt]Custom hand size (default {combo.get('hand_size', state['hand_size'])}): [/prompt]").strip()
                    if deck_size:
                        combo["deck_size"] = int(deck_size)
                    if hand_size:
                        combo["hand_size"] = int(hand_size)
                    if combo["deck_size"] <= 0 or combo["hand_size"] <= 0:
                        console.print("[error]Deck and hand sizes must be positive.[/error]")
                        pause()
                        return
                except:
                    console.print("[error]Invalid size input.[/error]")
                    pause()
                    return
            state["combos"][name] = combo
            console.print(f"[success]Combo '{name}' updated.[/success]")
            pause()
        elif sub == "2":
            del state["combos"][name]
        else:
            console.print("[error]Invalid option.[/error]")
            pause()

    elif choice == "3" or choice == "":
        return
    else:
        console.print("[error]Invalid option.[/error]")
        pause()

def validate_combo_set(expr):
    stack = 0
    last = "operator"
    for token in expr:
        if token == "(":
            stack += 1
            last = "open"
        elif token == ")":
            stack -= 1
            if stack < 0:
                return False
            last = "close"
        elif token in {"AND", "OR", "XOR"}:
            if last == "operator":
                return False
            last = "operator"
        else:
            if token not in state['combos'] and token not in state['cards']:
                return False
            last = "combo"
    return stack == 0 and last in {"combo", "close"}

def page_combo_sets():
    if not state['combos'] and not state['cards']:
        console.print("[error]Define combos or add cards first.[/error]")
        pause()
        return

    while True:
        clear_screen()
        console.print("[header][5] Combo Sets[/header]\n")
        for i, (name, data) in enumerate(state['combo_sets'].items()):
            expr = " ".join(data["expression"])
            m = data["mulligans"]
            console.print(f"[info][{i}] {name}: {expr}  (D{data['deck_size']} H{data['hand_size']} Mull:{m['enabled']}, Type:{m['type']})[/info]")
        console.print("\n[info]1. Add Combo Set[/info]")
        console.print("[info]2. Edit/Remove Combo Set[/info]")
        console.print("[info]3. Back[/info]")
        choice = console.input("[prompt]> [/prompt]")

        if choice == "1":
            name = console.input("[prompt]Combo Set name: [/prompt]").strip()
            expr = []
            combo_keys = list(state['combos'])
            card_keys = list(state['cards'])
            while True:
                clear_screen()
                console.print(f"[info]Expression so far: {' '.join(expr)}[/info]")
                console.print("[info]Combos:[/info]")
                for idx, c in enumerate(combo_keys):
                    console.print(f"[info]  [C{idx}] {c}[/info]")
                console.print("[info]Cards:[/info]")
                for idx, c in enumerate(card_keys):
                    console.print(f"[info]  [K{idx}] {c}[/info]")
                console.print("[info]Operators: [A]ND [O]R [X]OR, [B]racket, [D]one[/info]")
                tok = console.input("[prompt]> [/prompt]").upper()
                if tok == "D":
                    break
                if tok in {"A","O","X"}:
                    expr.append({"A":"AND","O":"OR","X":"XOR"}[tok])
                elif tok == "B":
                    expr.append(console.input("[prompt]Enter '(' or ')': [/prompt]").strip())
                elif tok.startswith("C") and tok[1:].isdigit() and int(tok[1:]) < len(combo_keys):
                    expr.append(combo_keys[int(tok[1:])])
                elif tok.startswith("K") and tok[1:].isdigit() and int(tok[1:]) < len(card_keys):
                    expr.append(card_keys[int(tok[1:])])
                else:
                    console.print("[error]Invalid input.[/error]")
                    pause()
                    continue
            if not validate_combo_set(expr):
                console.print("[error]Invalid expression.[/error]")
                pause()
                continue
            d_size = state['deck_size']
            h_size = state['hand_size']
            if console.input("[prompt]Custom deck/hand size? (y/N): [/prompt]").lower() == "y":
                d_size = int(console.input(f"[prompt]Deck size [{d_size}]: [/prompt]") or d_size)
                h_size = int(console.input(f"[prompt]Hand size [{h_size}]: [/prompt]") or h_size)
            mull = {"enabled": False, "count": 0, "first_free": False, "type": "traditional"}
            if console.input("[prompt]Enable mulligans? (y/N): [/prompt]").lower() == "y":
                mull["enabled"] = True
                console.print("\n[info]Select mulligan type:[/info]")
                console.print("[info]1. Traditional (redraw smaller hand)[/info]")
                console.print("[info]2. London (draw full hand, return cards)[/info]")
                console.print("[info]3. Partial (replace specific cards)[/info]")
                mulligan_choice = console.input("[prompt]> [/prompt]").strip()
                if mulligan_choice == "1":
                    mull["type"] = "traditional"
                elif mulligan_choice == "2":
                    mull["type"] = "london"
                elif mulligan_choice == "3":
                    mull["type"] = "partial"
                else:
                    console.print("[error]Invalid mulligan type. Defaulting to Traditional.[/error]")
                    mull["type"] = "traditional"

                total_min_required = 0
                for c in expr:
                    if c in state['combos']:
                        requirements = state['combos'][c]["requirements"]
                        total_min_required += sum(min_count for min_count, max_count in requirements.values())
                    elif c in state['cards']:
                        total_min_required += 1 
                default = max(0, h_size - total_min_required)
                try:
                    mull["count"] = int(console.input(f"[prompt]Count [{default}]: [/prompt]") or default)
                    if mull["count"] < 0:
                        console.print("[error]Mulligan count cannot be negative.[/error]")
                        pause()
                        continue
                except ValueError:
                    console.print("[error]Invalid mulligan count. Using default.[/error]")
                    mull["count"] = default
                mull["first_free"] = (console.input("[prompt]First free? (y/N): [/prompt]").lower() == "y")

            state['combo_sets'][name] = {
                "expression": expr,
                "deck_size": d_size,
                "hand_size": h_size,
                "mulligans": mull
            }
            console.print(f"[success]Combo Set '{name}' added.[/success]")
            pause()

        elif choice == "2":
            if not state['combo_sets']:
                console.print("[error]No combo sets to edit.[/error]")
                pause()
                continue
            try:
                idx = int(console.input("[prompt]Index to edit/remove: [/prompt]"))
                key = list(state['combo_sets'])[idx]
            except (ValueError, IndexError):
                console.print("[error]Invalid index.[/error]")
                pause()
                continue
            console.print(f"[prompt][1] Edit '{key}'  [2] Remove: [/prompt]")
            sub = console.input(f"[prompt]>[/prompt]")
            if sub == "2":
                del state['combo_sets'][key]
                console.print(f"[success]Removed '{key}'.[/success]")
                pause()
                continue
            data = state['combo_sets'][key]
            if console.input("[prompt]Change expression? (y/N): [/prompt]").lower() == "y":
                expr = []
                combo_keys = list(state['combos'])
                card_keys = list(state['cards'])
                while True:
                    clear_screen()
                    console.print(f"[info]Expression so far: {' '.join(expr)}[/info]")
                    console.print("[info]Combos:[/info]")
                    for idx2, c in enumerate(combo_keys):
                        console.print(f"[info]  [C{idx2}] {c}[/info]")
                    console.print("[info]Cards:[/info]")
                    for idx2, c in enumerate(card_keys):
                        console.print(f"[info]  [K{idx2}] {c}[/info]")
                    console.print("[info]Operators: [A]ND [O]R [X]OR, [B]racket, [D]one[/info]")
                    tok = console.input("[prompt]> [/prompt]").upper()
                    if tok == "D":
                        break
                    if tok in {"A","O","X"}:
                        expr.append({"A":"AND","O":"OR","X":"XOR"}[tok])
                    elif tok == "B":
                        expr.append(console.input("[prompt]Enter '(' or ')': [/prompt]").strip())
                    elif tok.startswith("C") and tok[1:].isdigit() and int(tok[1:]) < len(combo_keys):
                        expr.append(combo_keys[int(tok[1:])])
                    elif tok.startswith("K") and tok[1:].isdigit() and int(tok[1:]) < len(card_keys):
                        expr.append(card_keys[int(tok[1:])])
                    else:
                        console.print("[error]Invalid input.[/error]")
                        pause()
                        continue
                if validate_combo_set(expr):
                    data["expression"] = expr
                else:
                    console.print("[error]Invalid expression.[/error]")
                    pause()
                    continue

            if console.input("[prompt]Change deck/hand size? (y/N): [/prompt]").lower() == "y":
                d = int(console.input(f"[prompt]Deck size [{data['deck_size']}]: [/prompt]") or data['deck_size'])
                h = int(console.input(f"[prompt]Hand size [{data['hand_size']}]: [/prompt]") or data['hand_size'])
                data['deck_size'], data['hand_size'] = d, h

            if console.input("[prompt]Change mulligan settings? (y/N): [/prompt]").lower() == "y":
                m = data['mulligans']
                m["enabled"] = (console.input(f"[prompt]Enable? (y/N) [{m['enabled']}]: [/prompt]").lower() == "y")
                if m["enabled"]:
                    console.print("\n[info]Select mulligan type:[/info]")
                    console.print("[info]1. Traditional (redraw smaller hand)[/info]")
                    console.print("[info]2. London (draw full hand, return cards)[/info]")
                    console.print("[info]3. Partial (replace specific cards)[/info]")
                    mulligan_choice = console.input("[prompt]> [/prompt]").strip()
                    if mulligan_choice == "1":
                        m["type"] = "traditional"
                    elif mulligan_choice == "2":
                        m["type"] = "london"
                    elif mulligan_choice == "3":
                        m["type"] = "partial"
                    else:
                        console.print("[error]Invalid mulligan type. Defaulting to Traditional.[/error]")
                        m["type"] = "traditional"
                    total_min_required = 0
                    for c in data['expression']:
                        if c in state['combos']:
                            requirements = state['combos'][c]["requirements"]
                            total_min_required += sum(min_count for min_count, max_count in requirements.values())
                        elif c in state['cards']:
                            total_min_required += 1 
                    default = max(0, data['hand_size'] - total_min_required)
                    try:
                        m["count"] = int(console.input(f"[prompt]Count [{default}]: [/prompt]") or default)
                        if m["count"] < 0:
                            console.print("[error]Mulligan count cannot be negative.[/error]")
                            pause()
                            continue
                    except ValueError:
                        console.print("[error]Invalid mulligan count. Using default.[/error]")
                        m["count"] = default
                    m["first_free"] = (console.input(f"[prompt]First free? (y/N) [{m['first_free']}]: [/prompt]").lower() == "y")
                data['mulligans'] = m

            state['combo_sets'][key] = data
            console.print(f"[success]Updated '{key}'.[/success]")
            pause()
        elif choice == "3" or choice == "":
            return
        else:
            console.print("[error]Invalid option.[/error]")
            pause()

def infix_to_postfix(tokens):
    precedence = {'OR': 1, 'XOR': 2, 'AND': 3}
    output = []
    stack = []
    for token in tokens:
        if token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop() 
        elif token in precedence:
            while stack and stack[-1] in precedence and precedence[stack[-1]] >= precedence[token]:
                output.append(stack.pop())
            stack.append(token)
        else:
            output.append(token)
    while stack:
        output.append(stack.pop())
    return output

def parse_postfix(tokens):
    stack = []
    for token in tokens:
        if token in {"AND", "OR", "XOR"}:
            if len(stack) < 2:
                console.print(f"[error]Invalid expression: {tokens}[/error]")
                return None
            b = stack.pop()
            a = stack.pop()
            stack.append((token, a, b)) 
        else:
            stack.append(token)
    if len(stack) != 1:
        console.print(f"[error]Malformed expression: {tokens}[/error]")
        return None
    return stack[0]

def page_calculate_probability():
    if not state['combos']:
        console.print("[error]Define at least one combo first.[/error]")
        pause()
        return

    clear_screen()
    console.print("[header][6] Calculate Probability[/header]\n")

    card_counts = dict(state["cards"])
    combos_input = {}
    for name, combo in state['combos'].items():
        constraints = {}
        for T, bounds in combo["requirements"].items():
            constraints[T] = tuple(bounds)
        constraints["mulligans"]         = combo["mulligans"]["enabled"]
        constraints["mulligan_count"]    = combo["mulligans"]["count"]
        constraints["free_first_mulligan"] = combo["mulligans"]["first_free"]
        constraints["hand_size"]         = combo["hand_size"]
        constraints["deck_size"]         = combo["deck_size"]
        combos_input[name] = constraints

    extra_info_input = console.input("[prompt]Show detailed probability info (per mulligan attempt)? (y/N): [/prompt]").strip().lower()
    extra_info = extra_info_input == "y"

    combo_results = combo_type_probability(
        state['deck_size'],
        state['hand_size'],
        card_counts,
        state['card_types'],
        combos_input,
        extra_info=extra_info
    )

    console.print("[bold magenta]Combo Probabilities[/bold magenta]\n")
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Combo Name", style="bold")
    table.add_column("Label", style="green", no_wrap=True)
    table.add_column("Probability", justify="right", style="cyan")

    for name, result in combo_results.items():
        if isinstance(result, dict):
            if set(result.keys()) == {"No Mulligan", "Total"} and result["No Mulligan"] == result["Total"]:
                table.add_row(name, "-", f"{result['Total']*100:.4f}% ({result['Total']:.6f})")
            else:
                for label, val in result.items():
                    table.add_row(name, label, f"{val*100:.4f}% ({val:.6f})")
        else:
            table.add_row(name, "-", f"{result*100:.4f}% ({result:.6f})")

    console.print(table)
    parsed_sets = {}
    for name, data in state['combo_sets'].items():
        postfix = infix_to_postfix(data["expression"])
        tree = parse_postfix(postfix)
        if tree is not None:
            parsed_sets[name] = tree

    if parsed_sets:
        console.print("\n[bold magenta]Combo Set Probabilities[/bold magenta]\n")
        set_table = Table(show_header=True, header_style="bold yellow")
        set_table.add_column("Combo Set", style="bold")
        set_table.add_column("Label", style="green")
        set_table.add_column("Probability", justify="right", style="cyan")

        for name, tree in parsed_sets.items():
            cfg = {
                "expression": tree,
                "deck_size": state['combo_sets'][name]["deck_size"],
                "hand_size": state['combo_sets'][name]["hand_size"],
                "mulligans": state['combo_sets'][name]["mulligans"]
            }
            res = combo_set_probability(
                cfg,
                card_counts,
                state["card_types"],
                combos_input,
                extra_info=extra_info
            )
            if isinstance(res, dict):
                for label, p in res.items():
                    set_table.add_row(name, label, f"{p*100:.4f}% ({p:.6f})")
            else:
                set_table.add_row(name, "-", f"{res*100:.4f}% ({res:.6f})")

        console.print(set_table)
    else:
        console.print("[grey]No combo sets defined.[/grey]")

    pause()
    


def page_graph():
    clear_screen()
    width, height = size()
    console.print("[header][10] Graphs[/header]\n")
    
    console.print("[info]Load settings? (y/n)[/info]")
    load_settings = console.input("[prompt]> [/prompt]").lower() == 'y'
    settings = {}
    if load_settings:
        try:
            with open('plot_settings.json', 'r') as f:
                settings = load(f)
            console.print("[info]Settings loaded.[/info]")
        except Exception as e:
            console.print(f"[error]Failed to load settings: {e}[/error]")
    
    console.print("[info]Select graph type:[/info]")
    console.print("[info]1. Hypergeometric Probability[/info]")
    console.print("[info]2. Combo/Combo Set Probability[/info]")
    graph_type = console.input("[prompt]> [/prompt]") if not load_settings or 'graph_type' not in settings else settings['graph_type']
    if graph_type not in ["1", "2"]:
        console.print("[error]Invalid graph type.[/error]")
        pause()
        return

    console.print("\n[info]Customize plot style:[/info]")
    console.print("[info]Line style: 1. Solid (-), 2. Dashed (--), 3. Dotted (:)[/info]")
    line_style = {'1': '-', '2': '--', '3': ':'}.get(
        console.input("[prompt]> [/prompt]") if not load_settings or 'line_style' not in settings else settings['line_style'], '-')
    console.print("[info]Marker style: 1. Circle (o), 2. Square (s), 3. Triangle (^)[/info]")
    marker_style = {'1': 'o', '2': 's', '3': '^'}.get(
        console.input("[prompt]> [/prompt]") if not load_settings or 'marker_style' not in settings else settings['marker_style'], 'o')
    colors = ['cyan', 'magenta', 'yellow', 'green', 'red', 'blue', 'orange', 'pink']

    def get_int_input(prompt, minimum=0):
        val = console.input(f"[prompt]{prompt} > [/prompt]")
        if val.isdigit() and int(val) >= minimum:
            return int(val)
        return None

    def get_multiple_ints(prompt):
        vals = console.input(f"[prompt]{prompt} > [/prompt]")
        try:
            vals = vals.replace(',', ' ').split()
            ints = [int(v) for v in vals if int(v) > 0]
            return ints if ints else None
        except:
            return None

    if graph_type == "2":
        console.print("\n[info]Select items to analyze (space-separated indices):[/info]")
        items = [(name, "Combo") for name in state['combos']] + [(name, "Combo Set") for name in state['combo_sets']]
        if not items:
            console.print("[error]No combos or combo sets defined.[/error]")
            pause()
            return
        for i, (name, item_type) in enumerate(items):
            console.print(f"[info]{i}. {name} ({item_type})[/info]")
        item_indices = get_multiple_ints("[prompt]> [/prompt]") if not load_settings or 'item_idx' not in settings else settings['item_idx']
        if not item_indices or any(i >= len(items) for i in item_indices):
            console.print("[error]Invalid selection.[/error]")
            pause()
            return
        selected_items = [items[i] for i in item_indices]
        
        console.print("\n[info]Set x-Axis (parameter to vary):[/info]")
        console.print("[info]1. Deck Size[/info]")
        console.print("[info]2. Hand Size[/info]")
        x_axis = console.input("[prompt]> [/prompt]") if not load_settings or 'x_axis' not in settings else settings['x_axis']
        if x_axis not in ["1", "2"]:
            console.print("[error]Invalid x-axis choice.[/error]")
            pause()
            return
        x_label = "Deck Size" if x_axis == "1" else "Hand Size"
        min_prompt = "Min Deck Size (≥1)" if x_axis == "1" else "Min Hand Size (≥1)"
        max_prompt = "Max Deck Size (≥ Min)" if x_axis == "1" else "Max Hand Size (≥ Min)"
        x_min = get_int_input(f"[prompt]{min_prompt} > [/prompt]", minimum=1)
        if x_min is None:
            console.print("[error]Invalid min value.[/error]")
            pause()
            return
        x_max = get_int_input(f"[prompt]{max_prompt} > [/prompt]", minimum=x_min)
        if x_max is None:
            console.print("[error]Invalid max value.[/error]")
            pause()
            return
        fixed_label = "Hand Size" if x_axis == "1" else "Deck Size"
        fixed_val = console.input(f"[prompt]Set {fixed_label} > [/prompt]") if not load_settings or 'fixed_val' not in settings else str(settings['fixed_val'])
        if not fixed_val.isdigit() or int(fixed_val) < 1:
            console.print(f"[error]Invalid {fixed_label}.[/error]")
            pause()
            return
        fixed_val = int(fixed_val)
        
        card_counts = dict(state["cards"])
        x_vals = np.arange(x_min, x_max + 1)
        all_probabilities = []
        
        for selected_name, selected_type in selected_items:
            probabilities = []
            if selected_type == "Combo":
                combo_data = state['combos'][selected_name]
                constraints = {k: tuple(v) for k, v in combo_data["requirements"].items()}
                constraints["mulligans"] = combo_data["mulligans"]["enabled"]
                constraints["mulligan_count"] = combo_data["mulligans"]["count"]
                constraints["free_first_mulligan"] = combo_data["mulligans"]["first_free"]
                constraints["mulligan_type"] = combo_data["mulligans"].get("type", "traditional")
                
                for x_val in x_vals:
                    deck_size = x_val if x_axis == "1" else fixed_val
                    hand_size = x_val if x_axis == "2" else fixed_val
                    if hand_size > deck_size:
                        console.print(f"[error]Hand Size ({hand_size}) cannot exceed Deck Size ({deck_size}).[/error]")
                        pause()
                        return
                    constraints["deck_size"] = deck_size
                    constraints["hand_size"] = hand_size
                    result = combo_type_probability(
                        deck_size,
                        hand_size,
                        card_counts,
                        state['card_types'],
                        {selected_name: constraints},
                        extra_info=False
                    )
                    probabilities.append(result[selected_name])
            
            else:
                combo_set_data = state['combo_sets'][selected_name]
                postfix = infix_to_postfix(combo_set_data["expression"])
                tree = parse_postfix(postfix)
                if tree is None:
                    console.print(f"[error]Invalid combo set expression for {selected_name}.[/error]")
                    pause()
                    return
                cfg = {
                    "expression": tree,
                    "deck_size": combo_set_data["deck_size"],
                    "hand_size": combo_set_data["hand_size"],
                    "mulligans": combo_set_data["mulligans"]
                }
                combos_input = {k: {t: tuple(v) for t, v in c["requirements"].items()} for k, c in state['combos'].items()}
                for k, c in combos_input.items():
                    c["mulligans"] = state['combos'][k]["mulligans"]["enabled"]
                    c["mulligan_count"] = state['combos'][k]["mulligans"]["count"]
                    c["free_first_mulligan"] = state['combos'][k]["mulligans"]["first_free"]
                    c["mulligan_type"] = state['combos'][k]["mulligans"].get("type", "traditional")
                
                for x_val in x_vals:
                    deck_size = x_val if x_axis == "1" else fixed_val
                    hand_size = x_val if x_axis == "2" else fixed_val
                    if hand_size > deck_size:
                        console.print(f"[error]Hand Size ({hand_size}) cannot exceed Deck Size ({deck_size}).[/error]")
                        pause()
                        return
                    cfg["deck_size"] = deck_size
                    cfg["hand_size"] = hand_size
                    result = combo_set_probability(
                        cfg,
                        card_counts,
                        state["card_types"],
                        combos_input,
                        extra_info=False
                    )
                    probabilities.append(result)
            
            all_probabilities.append((selected_name, selected_type, probabilities))
        
        fig, ax = plt.subplots(figsize=(width/150, height/150))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        for idx, (name, type_, probs) in enumerate(all_probabilities):
            ax.plot(x_vals, probs, marker=marker_style, linestyle=line_style, color=colors[idx % len(colors)], label=f"P({name})")
            probs_np = np.array(probs)
            max_indices = np.where(probs_np == probs_np.max())[0]
            min_indices = np.where(probs_np == probs_np.min())[0]
            max_x = [x_vals[i] for i in max_indices]
            max_y = [probs[i] for i in max_indices]
            min_x = [x_vals[i] for i in min_indices]
            min_y = [probs[i] for i in min_indices]
            ax.scatter(max_x, max_y, color=colors[idx % len(colors)], marker='*', s=200, zorder=5)
            ax.scatter(min_x, min_y, color=colors[idx % len(colors)], marker='v', s=200, zorder=5)
        
        ax.set_xlabel(x_label, color='white')
        ax.set_ylabel("Probability", color='white')
        ax.tick_params(colors='white')
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.set_yticks(np.arange(0, 1.01, 0.02), minor=True)
        ax.grid(True, linestyle='--', alpha=0.5, color='white')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, 1)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_color('white')
        plt.title(f"Probability vs {x_label}", color='white')
        plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        console.print("[info]Export table to CSV? (y/n)[/info]")
        export_csv = console.input("[prompt]> [/prompt]").lower() == 'y'
        if export_csv:
            csv_filename = console.input("[prompt]Enter CSV filename (default: combo_table.csv) > [/prompt]")
            if not csv_filename:
                csv_filename = "combo_table.csv"
            if not csv_filename.endswith('.csv'):
                csv_filename += '.csv'
        
        table = Table(title=f"Probability vs {x_label}", show_header=True, header_style="bold white")
        table.add_column(x_label, style="bold")
        csv_data = [[x_label] + [f"P({name})" for name, _, _ in all_probabilities]]
        for name, _, _ in all_probabilities:
            table.add_column(f"P({name})", style="bold")
        for i, x_val in enumerate(x_vals):
            row = [str(x_val)]
            for _, _, probs in all_probabilities:
                row.append(f"{probs[i]:.4f}")
            table.add_row(*row)
            csv_data.append(row)
        console.print(table)
        
        if export_csv:
            try:
                with open(csv_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([f"Probability vs {x_label}"])
                    writer.writerows(csv_data)
            except Exception as e:
                console.print(f"[error]Failed to write CSV file: {e}[/error]")
        
        console.print("[info]Save settings? (y/n)[/info]")
        if console.input("[prompt]> [/prompt]").lower() == 'y':
            settings = {
                'graph_type': graph_type,
                'line_style': line_style,
                'marker_style': marker_style,
                'item_idx': item_indices,
                'x_axis': x_axis,
                'x_min': x_min,
                'x_max': x_max,
                'fixed_val': fixed_val
            }
            try:
                with open('plot_settings.json', 'w') as f:
                    dump(settings, f)
                console.print("[info]Settings saved.[/info]")
            except Exception as e:
                console.print(f"[error]Failed to save settings: {e}[/error]")
        
        plt.show()
    
    else:
        from scipy.stats import hypergeom
        
        def get_y(y_axis, k, N, K, n):
            if y_axis == "1":
                return hypergeom.pmf(k, N, K, n)
            if y_axis == "2":
                return 1 - hypergeom.pmf(k, N, K, n)
            if y_axis == "3":
                return hypergeom.cdf(k, N, K, n)
            if y_axis == "4":
                return 1 - hypergeom.cdf(k - 1, N, K, n)
            return 0
        
        def get_ylabel(y_axis):
            return {
                "1": "P(X = k)",
                "2": "P(X ≠ k)",
                "3": "P(X ≤ k)",
                "4": "P(X ≥ k)"
            }.get(y_axis, "Probability")
        
        console.print("[info]Select number of varying variables:[/info]")
        console.print("[info]1. One varying variable (line plot)[/info]")
        console.print("[info]2. Two varying variables (3D plot or heatmap)[/info]")
        num_vary = console.input("[prompt]> [/prompt]") if not load_settings or 'num_vary' not in settings else settings['num_vary']
        if num_vary not in ["1", "2"]:
            console.print("[error]Invalid choice.[/error]")
            pause()
            return
        
        console.print("[info]Set x-Axis (first varying parameter):[/info]")
        console.print("[info]1. Deck Size[/info]")
        console.print("[info]2. Hand Size[/info]")
        console.print("[info]3. Number of Success Cards[/info]")
        x_axis = console.input("[prompt]> [/prompt]") if not load_settings or 'x_axis' not in settings else settings['x_axis']
        if x_axis not in ["1", "2", "3"]:
            console.print("[error]Invalid x-axis choice.[/error]")
            pause()
            return
        x_label = {"1": "Deck Size", "2": "Hand Size", "3": "Number of Success Cards"}[x_axis]
        
        vary_options = {
            "1": "Deck Size",
            "2": "Hand Size",
            "3": "Number of Success Cards"
        }
        valid_vary = {k: v for k, v in vary_options.items() if k != x_axis}
        
        if num_vary == "1":
            console.print("[info]Which parameter to vary?[/info]")
            for key, label in valid_vary.items():
                console.print(f"[info]{key}. {label}[/info]")
            vary_choice = console.input("[prompt]> [/prompt]") if not load_settings or 'vary_choice' not in settings else settings['vary_choice']
            if vary_choice not in valid_vary:
                console.print("[error]Invalid vary choice.[/error]")
                pause()
                return
            vary_label = valid_vary[vary_choice]
            
            varying_vals = get_multiple_ints(f"[prompt]Enter {vary_label} values to vary (space or comma separated, >0) > [/prompt]")
            if not varying_vals:
                console.print("[error]Invalid varying values.[/error]")
                pause()
                return
            fixed_vals = set(vary_options.keys()) - {x_axis, vary_choice}
            fixed_choice = fixed_vals.pop()
            fixed_label = vary_options[fixed_choice]
            fixed_val = get_int_input(f"[prompt]Set {fixed_label} (single value, >0) > [/prompt]", minimum=1)
            if fixed_val is None:
                console.print(f"[error]Invalid {fixed_label}.[/error]")
                pause()
                return
            min_prompt = {
                "1": "Min Deck Size (≥1)",
                "2": "Min Hand Size (≥1)",
                "3": "Min Number of Success Cards (≥1)"
            }
            max_prompt = {
                "1": "Max Deck Size (≥ Min)",
                "2": "Max Hand Size (≥ Min)",
                "3": "Max Number of Success Cards (≥ Min)"
            }
            x_min = get_int_input(f"[prompt]{min_prompt[x_axis]}  [/prompt]", minimum=1)
            if x_min is None:
                console.print("[error]Invalid min value.[/error]")
                pause()
                return
            x_max = get_int_input(f"[prompt]{max_prompt[x_axis]}  [/prompt]", minimum=x_min)
            if x_max is None:
                console.print("[error]Invalid max value.[/error]")
                pause()
                return
            
            console.print("[info]Set y-Axis[/info]")
            y_axis_options = {
                "1": "P(X = k)",
                "2": "P(X ≠ k)",
                "3": "P(X ≤ k)",
                "4": "P(X ≥ k)"
            }
            
            for key, label in y_axis_options.items():
                console.print(f"[info]{key}. {label}[/info]")
            y_axis = console.input("[prompt]> [/prompt]") if not load_settings or 'y_axis' not in settings else settings['y_axis']
            if y_axis not in y_axis_options:
                console.print("[error]Invalid y-axis choice.[/error]")
                pause()
                return
            
            k = console.input("[prompt]Set k (≥0) > [/prompt]") if not load_settings or 'k' not in settings else str(settings['k'])
            if not (k.isdigit() and int(k) >= 0):
                console.print("[error]Invalid k.[/error]")
                pause()
                return
            k = int(k)
            
            x = np.arange(x_min, x_max + 1)
            
            def get_param_value(param, val_x, val_vary, val_fixed):
                if param == x_axis:
                    return val_x
                elif param == vary_choice:
                    return val_vary
                else:
                    return val_fixed
            
            fig, ax = plt.subplots(figsize=(width/150, height/150))
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            
            for i, v in enumerate(varying_vals):
                y_vals = []
                for val_x in x:
                    N = get_param_value("1", val_x, v, fixed_val)
                    K = get_param_value("3", val_x, v, fixed_val)
                    n = get_param_value("2", val_x, v, fixed_val)
                    if n > N or K > N:
                        console.print(f"[error]Invalid: Hand Size ({n}) or Success Cards ({K}) > Deck Size ({N})[/error]")
                        pause()
                        return
                    y_vals.append(get_y(y_axis, k, N, K, n))
                
                y_vals_np = np.array(y_vals)
                max_indices = np.where(y_vals_np == y_vals_np.max())[0]
                min_indices = np.where(y_vals_np == y_vals_np.min())[0]
                max_x = [x[i] for i in max_indices]
                max_y = [y_vals[i] for i in max_indices]
                min_x = [x[i] for i in min_indices]
                min_y = [y_vals[i] for i in min_indices]
                if x_min == 1 and x_axis == "3":
                    x = np.append(0, x)
                    if y_axis == "1":
                        y_vals.insert(0, 1 if k == 0 else 0)
                    elif y_axis == "2":
                        y_vals.insert(0, 0 if k == 0 else 1)
                    elif y_axis == "3":
                        y_vals.insert(0, 1)
                    elif y_axis == "4":
                        y_vals.insert(0, 1 if k == 0 else 0)
                ax.plot(x, y_vals, marker=marker_style, linestyle=line_style, color=colors[i % len(colors)], label=f"{vary_label} = {v}")
                ax.scatter(max_x, max_y, color=colors[i % len(colors)], marker='*', s=200, zorder=5)
                ax.scatter(min_x, min_y, color=colors[i % len(colors)], marker='v', s=200, zorder=5)
            
            ax.set_xlabel(x_label, color='white')
            ax.set_ylabel(get_ylabel(y_axis), color='white')
            ax.tick_params(colors='white')
            ax.set_yticks(np.arange(0, 1.01, 0.1))
            ax.set_yticks(np.arange(0, 1.01, 0.02), minor=True)
            ax.grid(True, linestyle='--', alpha=0.5, color='white')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(0, 1)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            
            legend = ax.legend()
            for text in legend.get_texts():
                text.set_color('white')
            plt.title(f"Probability vs {x_label}", color='white')
            plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            console.print("[info]Export tables to CSV? (y/n)[/info]")
            export_csv = console.input("[prompt]> [/prompt]").lower() == 'y'
            if export_csv:
                csv_filename = console.input("[prompt]Enter CSV filename (default: hypergeometric_table.csv) > [/prompt]")
                if not csv_filename:
                    csv_filename = "hypergeometric_table.csv"
                if not csv_filename.endswith('.csv'):
                    csv_filename += '.csv'
            
            for v in varying_vals:
                table = Table(title=f"{vary_label} = {v}", show_header=True, header_style="bold white")
                table.add_column(x_label, style="bold")
                table.add_column(get_ylabel(y_axis), style="bold")
                csv_data = [[x_label, get_ylabel(y_axis)]]
                
                for j, val_x in enumerate(x):
                    N = get_param_value("1", val_x, v, fixed_val)
                    K = get_param_value("3", val_x, v, fixed_val)
                    n = get_param_value("2", val_x, v, fixed_val)
                    val_y = get_y(y_axis, k, N, K, n)
                    table.add_row(str(val_x), f"{val_y:.4f}")
                    csv_data.append([str(val_x), f"{val_y:.4f}"])
                console.print(table)
                
                if export_csv:
                    try:
                        with open(f"{vary_label}_{v}_{csv_filename}", 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([f"{vary_label} = {v}"])
                            writer.writerows(csv_data)
                    except Exception as e:
                        console.print(f"[error]Failed to write CSV file: {e}[/error]")
            
            console.print("[info]Save settings? (y/n)[/info]")
            if console.input("[prompt]> [/prompt]").lower() == 'y':
                settings = {
                    'graph_type': graph_type,
                    'line_style': line_style,
                    'marker_style': marker_style,
                    'num_vary': num_vary,
                    'x_axis': x_axis,
                    'vary_choice': vary_choice,
                    'varying_vals': varying_vals,
                    'fixed_val': fixed_val,
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_axis': y_axis,
                    'k': k
                }
                try:
                    with open('plot_settings.json', 'w') as f:
                        dump(settings, f)
                    console.print("[info]Settings saved.[/info]")
                except Exception as e:
                    console.print(f"[error]Failed to save settings: {e}[/error]")
            
            plt.show()
        
        else:
            console.print("[info]Select second varying parameter (y-axis):[/info]")
            for key, label in valid_vary.items():
                console.print(f"[info]{key}. {label}[/info]")
            y_axis_var = console.input("[prompt]> [/prompt]") if not load_settings or 'y_axis_var' not in settings else settings['y_axis_var']
            if y_axis_var not in valid_vary:
                console.print("[error]Invalid y-axis choice.[/error]")
                pause()
                return
            y_label = valid_vary[y_axis_var]
            
            fixed_vals = set(vary_options.keys()) - {x_axis, y_axis_var}
            fixed_choice = fixed_vals.pop()
            fixed_label = vary_options[fixed_choice]
            fixed_val = get_int_input(f"[prompt]Set {fixed_label} (single value, >0) > [/prompt]", minimum=1)
            if fixed_val is None:
                console.print(f"[error]Invalid {fixed_label}.[/error]")
                pause()
                return
            
            min_prompt = {
                "1": "Min Deck Size (≥1)",
                "2": "Min Hand Size (≥1)",
                "3": "Min Number of Success Cards (≥1)"
            }
            max_prompt = {
                "1": "Max Deck Size (≥ Min)",
                "2": "Max Hand Size (≥ Min)",
                "3": "Max Number of Success Cards (≥ Min)"
            }
            x_min = get_int_input(f"[prompt]{min_prompt[x_axis]} > [/prompt]", minimum=1)
            if x_min is None:
                console.print("[error]Invalid x min value.[/error]")
                pause()
                return
            x_max = get_int_input(f"[prompt]{max_prompt[x_axis]} > [/prompt]", minimum=x_min)
            if x_max is None:
                console.print("[error]Invalid x max value.[/error]")
                pause()
                return
            y_min = get_int_input(f"[prompt]{min_prompt[y_axis_var]} > [/prompt]", minimum=1)
            if y_min is None:
                console.print("[error]Invalid y min value.[/error]")
                pause()
                return
            y_max = get_int_input(f"[prompt]{max_prompt[y_axis_var]} > [/prompt]", minimum=y_min)
            if y_max is None:
                console.print("[error]Invalid y max value.[/error]")
                pause()
                return
            
            console.print("[info]Select plot type:[/info]")
            console.print("[info]1. 3D Surface Plot[/info]")
            console.print("[info]2. Heatmap[/info]")
            plot_type = console.input("[prompt]> [/prompt]") if not load_settings or 'plot_type' not in settings else settings['plot_type']
            if plot_type not in ["1", "2"]:
                console.print("[error]Invalid plot type.[/error]")
                pause()
                return
            
            console.print("[info]Set probability type:[/info]")
            y_axis_options = {
                "1": "P(X = k)",
                "2": "P(X ≠ k)",
                "3": "P(X ≤ k)",
                "4": "P(X ≥ k)"
            }
            for key, label in y_axis_options.items():
                console.print(f"[info]{key}. {label}[/info]")
            y_axis = console.input("[prompt]> [/prompt]") if not load_settings or 'y_axis' not in settings else settings['y_axis']
            if y_axis not in y_axis_options:
                console.print("[error]Invalid probability type.[/error]")
                pause()
                return
            k = console.input("[prompt]Set k (≥0) > [/prompt]") if not load_settings or 'k' not in settings else str(settings['k'])
            if not (k.isdigit() and int(k) >= 0):
                console.print("[error]Invalid k.[/error]")
                pause()
                return
            k = int(k)
            
            x_vals = np.arange(x_min, x_max + 1)
            y_vals = np.arange(y_min, y_max + 1)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = np.zeros(X.shape)
            
            def get_param_value(param, val_x, val_y, val_fixed):
                if param == x_axis:
                    return val_x
                elif param == y_axis_var:
                    return val_y
                else:
                    return val_fixed
            
            for i in range(len(y_vals)):
                for j in range(len(x_vals)):
                    N = get_param_value("1", x_vals[j], y_vals[i], fixed_val)
                    K = get_param_value("3", x_vals[j], y_vals[i], fixed_val)
                    n = get_param_value("2", x_vals[j], y_vals[i], fixed_val)
                    if n > N or K > N:
                        console.print(f"[error]Invalid: Hand Size ({n}) or Success Cards ({K}) > Deck Size ({N})[/error]")
                        pause()
                        return
                    Z[i, j] = get_y(y_axis, k, N, K, n)
            
            fig = plt.figure(figsize=(width/150, height/150))
            fig.patch.set_facecolor('black')
            
            if plot_type == "1":
                ax = fig.add_subplot(111, projection='3d')
                ax.set_facecolor('black')
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', rstride=1, cstride=1, shade=True)
                ax.view_init(elev=30, azim=45)
                ax.set_xlabel(x_label, color='white')
                ax.set_ylabel(y_label, color='white')
                ax.set_zlabel(get_ylabel(y_axis), color='white')
                ax.tick_params(colors='white')
                ax.set_zticks(np.arange(0, 1.01, 0.1))
                ax.set_zticks(np.arange(0, 1.01, 0.02), minor=True)
                ax.xaxis.line.set_color('white')
                ax.yaxis.line.set_color('white')
                ax.zaxis.line.set_color('white')
                ax.xaxis.set_pane_color((0, 0, 0, 1))
                ax.yaxis.set_pane_color((0, 0, 0, 1))
                ax.zaxis.set_pane_color((0, 0, 0, 1))
                fig.colorbar(surf, ax=ax, label='Probability', pad=0.1, shrink=0.8).ax.yaxis.set_tick_params(color='white', labelcolor='white')
                plt.title(f"Probability vs {x_label} and {y_label}", color='white')
            
            else:
                ax = fig.add_subplot(111)
                ax.set_facecolor('black')
                im = ax.imshow(Z, cmap='viridis', origin='lower', extent=[x_min, x_max, y_min, y_max])
                ax.set_xlabel(x_label, color='white')
                ax.set_ylabel(y_label, color='white')
                ax.tick_params(colors='white')
                ax.set_xticks(np.arange(x_min, x_max + 1))
                ax.set_yticks(np.arange(y_min, y_max + 1))
                fig.colorbar(im, ax=ax, label='Probability').ax.yaxis.set_tick_params(color='white', labelcolor='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_linewidth(1.5)
                ax.spines['left'].set_linewidth(1.5)
                
                max_indices = np.where(Z == Z.max())
                min_indices = np.where(Z == Z.min())
                for i, j in zip(max_indices[0], max_indices[1]):
                    ax.text(x_vals[j], y_vals[i], 'Max', color='white', ha='center', va='center', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
                for i, j in zip(min_indices[0], min_indices[1]):
                    ax.text(x_vals[j], y_vals[i], 'Min', color='white', ha='center', va='center', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
                
                plt.title(f"Probability vs {x_label} and {y_label}", color='white')
            
            plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            console.print("[info]Export table to CSV? (y/n)[/info]")
            export_csv = console.input("[prompt]> [/prompt]").lower() == 'y'
            if export_csv:
                csv_filename = console.input("[prompt]Enter CSV filename (default: hypergeometric_table.csv) > [/prompt]")
                if not csv_filename:
                    csv_filename = "hypergeometric_table.csv"
                if not csv_filename.endswith('.csv'):
                    csv_filename += '.csv'
            
            table = Table(title=f"Probability ({get_ylabel(y_axis)})", show_header=True, header_style="bold white")
            table.add_column(x_label, style="bold")
            csv_data = [[x_label] + [f"{y_label} = {y_val}" for y_val in y_vals]]
            for y_val in y_vals:
                table.add_column(f"{y_label} = {y_val}", style="bold")
            for j, x_val in enumerate(x_vals):
                row = [str(x_val)]
                for i in range(len(y_vals)):
                    row.append(f"{Z[i, j]:.4f}")
                table.add_row(*row)
                csv_data.append(row)
            console.print(table)
            
            if export_csv:
                try:
                    with open(csv_filename, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([f"Hypergeometric Probability ({get_ylabel(y_axis)})"])
                        writer.writerows(csv_data)
                except Exception as e:
                    console.print(f"[error]Failed to write CSV file: {e}[/error]")
            
            console.print("[info]Save settings? (y/n)[/info]")
            if console.input("[prompt]> [/prompt]").lower() == 'y':
                settings = {
                    'graph_type': graph_type,
                    'line_style': line_style,
                    'marker_style': marker_style,
                    'num_vary': num_vary,
                    'x_axis': x_axis,
                    'y_axis_var': y_axis_var,
                    'fixed_val': fixed_val,
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                    'plot_type': plot_type,
                    'y_axis': y_axis,
                    'k': k
                }
                try:
                    with open('plot_settings.json', 'w') as f:
                        dump(settings, f)
                    console.print("[info]Settings saved.[/info]")
                except Exception as e:
                    console.print(f"[error]Failed to save settings: {e}[/error]")
            
            plt.show()
    
    pause()

    
def page_save():
    clear_screen()
    console.print("[header][7] Save Deck State[/header]")
    path = console.input("[prompt]Enter filename to save (e.g., deck.json): [/prompt]")
    try:
        with open(path, "w") as f:
            dump(state, f, indent=4)
        console.print(f"[success]State saved to {path}.[/success]")
    except Exception as e:
        console.print(f"[error]Error saving file: {e}[/error]")
    pause()

def page_load(from_first_screen=False):
    clear_screen()
    if not from_first_screen:
        console.print("[header][8] Load Deck State[/header]")
    else:
        console.print("[header]Load Deck State[/header]")
    files = [f for f in listdir() if f.endswith(".json") and isfile(f)]

    if files:
        console.print("[info]Available .json files in current directory:[/info]")
        for i, fname in enumerate(files):
            console.print(f"[option]{i + 1}[/option]: {fname}")
    else:
        console.print("[info]No .json files found in current directory.[/info]")
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
def page_list():
    clear_screen()
    console.print("[header][9] State Summary[/header]\n")
    console.print(f"[info]Deck size: {state['deck_size']}[/info]")
    console.print(f"[info]Hand size: {state['hand_size']}[/info]\n")
    console.print("[info]Cards:[/info]")
    for card, qty in state['cards'].items():
        console.print(f"[info]  {card}: {qty}[/info]")
    console.print("\n[info]Card Types:[/info]")
    for t, members in state['card_types'].items():
        console.print(f"[info]  {t}: {members}[/info]")
    console.print("\n[info]Combos:[/info]")
    for name, combo in state['combos'].items():
        console.print(f"[info]  {name}: {combo}[/info]")
    console.print("\n[info]Combo Sets:[/info]")
    for name, expr in state['combo_sets'].items():
        console.print(f"[info]  {name}: {expr}[/info]")
    pause()

def page_exit():
    clear_screen()
    console.print("[header]Exiting program.[/header]")
    exit()

def main():
    import_deck_prompt()
    while True:
        clear_screen()
        console.print("[header]Main Page[/header]\n")
        console.print("[info]1.[/info] [text]Deck and Hand Size[/text]")
        console.print("[info]2.[/info] [text]Cards and Quantities[/text]")
        console.print("[info]3.[/info] [text]Card Types[/text]")
        console.print("[info]4.[/info] [text]Combo Requirements[/text]")
        console.print("[info]5.[/info] [text]Combo Sets[/text]")
        console.print("[info]6.[/info] [text]Calculate Probability[/text]")
        console.print("[info]7.[/info] [text]Save to File[/text]")
        console.print("[info]8.[/info] [text]Load Deck/JSON[/text]")
        console.print("[info]9.[/info] [text]Show All State[/text]")
        console.print("[info]10.[/info] [text]Graphs[text]")
        console.print("[info]11.[/info] [text]Exit[text]")
        choice = console.input("[prompt]> [/prompt]")
        if choice == "1": page_deck_hand_size()
        elif choice == "2": page_cards()
        elif choice == "3": page_card_types()
        elif choice == "4": page_combos()
        elif choice == "5": page_combo_sets()
        elif choice == "6": page_calculate_probability()
        elif choice == "7": page_save()
        elif choice == "8": import_deck_prompt(from_first_screen=False)
        elif choice == "9": page_list()
        elif choice == "10": page_graph()
        elif choice == "11": page_exit()
        else:
            console.print("[error]Invalid option.[/error]")
            pause()

if __name__ == "__main__":
    main()
