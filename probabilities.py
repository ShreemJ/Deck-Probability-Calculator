from collections import defaultdict, Counter, deque
from math import comb
import numpy as np
from random import shuffle
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.table import Table
from functools import lru_cache

from utility import *
from config import console, state

def hand_satisfies_combo(hand, constraints_and_settings, card_types):
    constraints = {
        k: v if isinstance(v, tuple) else tuple(v)
        for k, v in constraints_and_settings.items()
        if isinstance(v, (list, tuple)) and len(v) == 2
    }

    roles = []
    for T, (min_required, _) in constraints.items():
        roles += [T] * min_required

    cards = list(hand)

    card_to_types = defaultdict(set)
    for T, card_list in card_types.items():
        for c in card_list:
            card_to_types[c].add(T)
    for card in hand:
        card_to_types[card].add(card)

    n_roles = len(roles)
    n_cards = len(cards)

    adj = defaultdict(list)
    for role_idx, role in enumerate(roles):
        for card_idx, card in enumerate(cards):
            if role in card_to_types[card]:
                adj[role_idx].append(card_idx)

    pair_U = {u: None for u in range(n_roles)}
    pair_V = {v: None for v in range(n_cards)}
    dist = {}

    def bfs():
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

    def dfs(u):
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

def evaluate_combo_expression(hand, combo_expr, combos, card_types):
    if isinstance(combo_expr, str):
        if combo_expr in combos:
            return hand_satisfies_combo(hand, combos[combo_expr], card_types)
        else:
            return hand.count(combo_expr) >= 1
    op = combo_expr[0]
    args = combo_expr[1:]

    match op:
        case "AND":
            return all(evaluate_combo_expression(hand, arg, combos, card_types) for arg in args)
        case "OR":
            return any(evaluate_combo_expression(hand, arg, combos, card_types) for arg in args)
        case "XOR":
            return sum(evaluate_combo_expression(hand, arg, combos, card_types) for arg in args) == 1
        case _:
            raise ValueError(f"Unknown logical operator: {op}")

def combo_type_probability(deck_size, hand_size, card_count, card_types, combos, debug=False, extra_info=False):
    card_counts = card_count
    total_card_count = sum(card_counts.values())
    if total_card_count < deck_size:
        missing = deck_size - total_card_count
        card_counts = dict(card_counts)
        card_counts["Unspecified"] = missing
        if debug:
            console.print(f"[info]Added {missing} Unspecified cards to reach deck size {deck_size}[/info]")
    elif total_card_count > deck_size:
        raise ValueError(f"Sum of card counts ({total_card_count}) exceeds deck size ({deck_size})")
    results = {}

    @lru_cache(maxsize=None)
    def generate_counts(cards, hand_size, current_counts, index, constraints, result, card_counts):
        if index == len(cards):
            if sum(current_counts.values()) == hand_size:
                hand = []
                for card, count in current_counts.items():
                    hand.extend([card] * count)
                if not hand_satisfies_combo(hand, constraints, card_types):
                    return
                prob = 1.0
                for card, count in current_counts.items():
                    prob *= comb(card_counts[card], count)
                result[0] += prob
            return

        card = cards[index]
        max_count = min(card_counts[card], hand_size - sum(current_counts.values()))
        for count in range(max_count + 1):
            current_counts[card] = count
            generate_counts(cards, hand_size, current_counts, index + 1, constraints, result, card_counts)
            current_counts[card] = 0

    def generate_counts_with_replacements(cards, hand_size, current_counts, index, constraints, replacements, deck_counts, result):
        if index == len(cards):
            if sum(current_counts.values()) == hand_size:
                hand = []
                for card, count in current_counts.items():
                    hand.extend([card] * count)
                if hand_satisfies_combo(hand, constraints, card_types):
                    prob = 1.0
                    for card, count in current_counts.items():
                        prob *= comb(deck_counts[card], count)
                    result[0] += prob
            return

        card = cards[index]
        max_count = min(deck_counts[card], hand_size - sum(current_counts.values()))
        for count in range(max_count + 1):
            current_counts[card] = count
            generate_counts_with_replacements(cards, hand_size, current_counts, index + 1, constraints, replacements, deck_counts, result)
            current_counts[card] = 0

    def prob_with_mulligans(constraints, mulligans, mulligan_count, free_first_mulligan, hand_size, mulligan_type):
        max_mulls = mulligan_count if mulligans and mulligan_count is not None else hand_size - 1
        if max_mulls + (1 if free_first_mulligan else 0) <= 0:
            max_mulls = 0
        match mulligan_type:
            case "traditional":
                attempts = [hand_size]
                if free_first_mulligan:
                    attempts.append(hand_size)
                for i in range(max_mulls):
                    next_size = hand_size - (i + 1)
                    if next_size <= 0:
                        break
                    attempts.append(next_size)

                p_hit = []
                for h in attempts:
                    result = [0.0]
                    generate_counts(list(all_cards_this_combo.keys()), h, defaultdict(int), 0, constraints, result, all_cards_this_combo)
                    p_hit.append(result[0] / comb(deck_size, h))

            case "london":
                attempts = []
                if free_first_mulligan:
                    attempts.append((hand_size, 0))
                    attempts.append((hand_size, 1))
                else:
                    attempts.append((hand_size, 0))
                for i in range(max_mulls):
                    cards_to_return = i + (1 if not free_first_mulligan else 2)
                    final_hand_size = hand_size - cards_to_return
                    if final_hand_size <= 0:
                        break
                    attempts.append((hand_size, cards_to_return))

                p_hit = []
                for h, cards_to_return in attempts:
                    final_hand_size = h - cards_to_return
                    if final_hand_size <= 0:
                        break
                    result = [0.0]
                    generate_counts(list(all_cards_this_combo.keys()), h, defaultdict(int), 0, constraints, result, all_cards_this_combo)
                    prob = result[0] / comb(deck_size, h)
                    p_hit.append(prob)

            case "partial":
                p_hit = []
                result = [0.0]
                generate_counts(list(all_cards_this_combo.keys()), hand_size, defaultdict(int), 0, constraints, result, all_cards_this_combo)
                p_initial = result[0] / comb(deck_size, hand_size)
                p_hit.append(p_initial)
                min_required = sum(min_count for card_type, (min_count, max_count) in constraints.items()
                                  if card_type not in ("mulligans", "mulligan_count", "free_first_mulligan", "hand_size", "deck_size", "mulligan_type"))

                remaining_mulls = max_mulls + (1 if free_first_mulligan else 0)
                for _ in range(remaining_mulls):
                    result = [0.0]
                    def gen_initial(idx, curr, hand_size):
                        if idx == len(all_cards_this_combo):
                            if sum(curr.values()) != hand_size:
                                return
                            hand = []
                            for c, n in curr.items():
                                hand += [c] * n
                            if hand_satisfies_combo(hand, constraints, card_types):
                                prob = 1.0
                                for c, n in curr.items():
                                    prob *= comb(all_cards_this_combo[c], n)
                                result[0] += prob
                                return

                            non_contributing = sum(n for c, n in curr.items() if c == "Unspecified" or not any(c in card_types.get(t, [t]) for t in constraints))
                            cards_to_replace = min(non_contributing, hand_size - min_required)
                            if cards_to_replace <= 0:
                                return

                            new_deck = dict(all_cards_this_combo)
                            for c, n in curr.items():
                                new_deck[c] -= n
                                if new_deck[c] < 0:
                                    return
                            new_deck["Unspecified"] = new_deck.get("Unspecified", 0) + cards_to_replace
                            new_result = [0.0]
                            generate_counts_with_replacements(list(new_deck.keys()), cards_to_replace, defaultdict(int), 0, constraints, cards_to_replace, new_deck, new_result)
                            prob = 1.0
                            for c, n in curr.items():
                                prob *= comb(all_cards_this_combo[c], n)
                            result[0] += prob * (new_result[0] / comb(deck_size - hand_size, cards_to_replace))
                        c = list(all_cards_this_combo.keys())[idx]
                        max_n = min(all_cards_this_combo[c], hand_size - sum(curr.values()))
                        for take in range(max_n + 1):
                            curr[c] = take
                            gen_initial(idx + 1, curr, hand_size)
                            curr[c] = 0

                    gen_initial(0, defaultdict(int), hand_size)
                    p_hit.append(result[0] / comb(deck_size, hand_size))

            case _:
                raise ValueError(f"Unknown mulligan type: {mulligan_type}")

        if not extra_info or not mulligans:
            p_fail_all = 1.0
            for p in p_hit:
                p_fail_all *= (1 - p)
            return 1 - p_fail_all

        info = {}
        p_fail_so_far = 1.0
        for i, p in enumerate(p_hit):
            attempt_name = "No Mulligan" if i == 0 else f"Mulligan {i}"
            p_reach_attempt = p_fail_so_far
            p_success_this_attempt = p * p_reach_attempt
            info[attempt_name] = p_success_this_attempt
            p_fail_so_far *= (1 - p)

        info["Total"] = 1 - p_fail_so_far
        return info

    for combo_name, data in combos.items():
        if "requirements" in data:
            constraints = data["requirements"]
        else:
            constraints = {k: v for k, v in data.items() if k not in ("mulligans", "mulligan_count", "free_first_mulligan", "hand_size", "deck_size", "mulligan_type")}

        mulligans_config = data.get("mulligans", False)
        if isinstance(mulligans_config, dict):
            mulligans = mulligans_config.get("enabled", False)
            mulligan_count = mulligans_config.get("count", None)
            free_first_mulligan = mulligans_config.get("first_free", False)
            mulligan_type = mulligans_config.get("type", "traditional")
        else:
            mulligans = mulligans_config
            mulligan_count = None
            free_first_mulligan = False
            mulligan_type = "traditional"

        h_size = data.get("hand_size", hand_size)
        d_size = data.get("deck_size", deck_size)

        combo_cards = set()
        for item in constraints:
            if item in card_types:
                combo_cards.update(card_types.get(item, [item]))
            else:
                combo_cards.add(item)

        all_cards_this_combo = {}
        total_combo_cards = 0
        for card in combo_cards:
            count = card_counts.get(card, 0)
            all_cards_this_combo[card] = count
            total_combo_cards += count

        unspecified_count = d_size - total_combo_cards
        if unspecified_count > 0:
            all_cards_this_combo["Unspecified"] = unspecified_count

        if mulligans:
            results[combo_name] = prob_with_mulligans(constraints, mulligans, mulligan_count, free_first_mulligan, h_size, mulligan_type)
        else:
            result = [0.0]
            generate_counts(list(all_cards_this_combo.keys()), h_size, defaultdict(int), 0, constraints, result, all_cards_this_combo)
            prob = result[0] / comb(d_size, h_size)
            if extra_info:
                results[combo_name] = {"No Mulligan": prob, "Total": prob}
            else:
                results[combo_name] = prob

    return results

def combo_set_probability(set_config, card_counts, card_types, combos_constraints, extra_info=False):
    d = set_config["deck_size"]
    h0 = set_config["hand_size"]
    mulligan_settings = set_config["mulligans"]
    def collect_combo_cards(expression):
        combo_cards = set()
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

    combo_cards = collect_combo_cards(set_config["expression"])
    all_cards_this_combo = {}
    total_combo_cards = 0
    for card in combo_cards:
        count = card_counts.get(card, 0)
        all_cards_this_combo[card] = count
        total_combo_cards += count
    unspecified_count = d - total_combo_cards
    if unspecified_count > 0:
        all_cards_this_combo["Unspecified"] = unspecified_count

    @lru_cache(maxsize=None)
    def generate_counts(cards, hand_size, current_counts, index, expression, replacements, deck_counts, result):
        if index == len(cards):
            if sum(current_counts.values()) == hand_size:
                hand = []
                for card, count in current_counts.items():
                    hand.extend([card] * count)
                if evaluate_combo_expression(hand, expression, combos_constraints, card_types):
                    prob = 1.0
                    for card, count in current_counts.items():
                        prob *= comb(deck_counts[card], count)
                    result[0] += prob
            return

        card = cards[index]
        max_count = min(deck_counts[card], hand_size - sum(current_counts.values()))
        for count in range(max_count + 1):
            current_counts[card] = count
            generate_counts(cards, hand_size, current_counts, index + 1, expression, replacements, deck_counts, result)
            current_counts[card] = 0

    @lru_cache(maxsize=None)
    def single_hit(h, cards_to_return=0):
        cards = list(all_cards_this_combo.keys())
        result = [0.0]

        def gen(idx, curr):
            if idx == len(cards):
                if sum(curr.values()) != h:
                    return
                hand = []
                for c, n in curr.items():
                    hand += [c] * n
                if evaluate_combo_expression(hand, set_config["expression"], combos_constraints, card_types):
                    ways = 1.0
                    for c, n in curr.items():
                        ways *= comb(all_cards_this_combo[c], n)
                    result[0] += ways
                return

            c = cards[idx]
            max_n = min(all_cards_this_combo[c], h - sum(curr.values()))
            for take in range(max_n + 1):
                curr[c] = take
                gen(idx + 1, curr)
            curr.pop(c, None)

        gen(0, {})
        final_hand_size = h - cards_to_return
        if final_hand_size <= 0:
            return 0.0
        prob = result[0] / comb(d, h)
        return prob

    mulligan_type = mulligan_settings.get("type", "traditional")
    match mulligan_type:
        case "traditional":
            attempts = [h0]
            if mulligan_settings["first_free"]:
                attempts.append(h0)
            for i in range(mulligan_settings["count"] if mulligan_settings["enabled"] else 0):
                nh = h0 - (i + 1)
                if nh <= 0:
                    break
                attempts.append(nh)
            p_hits = [single_hit(h) for h in attempts]

        case "london":
            attempts = []
            if mulligan_settings["first_free"]:
                attempts.append((h0, 0))
                attempts.append((h0, 1))
            else:
                attempts.append((h0, 0))
            for i in range(mulligan_settings["count"] if mulligan_settings["enabled"] else 0):
                nh = h0
                cards_to_return = i + (1 if not mulligan_settings["first_free"] else 2)
                if nh - cards_to_return <= 0:
                    break
                attempts.append((nh, cards_to_return))
            p_hits = [single_hit(h, cards_to_return) for h, cards_to_return in attempts]

        case "partial":
            p_hits = []
            p_initial = single_hit(h0)
            p_hits.append(p_initial)
            def get_min_required(expression):
                if isinstance(expression, str):
                    if expression in combos_constraints:
                        return sum(min_count for card_type, (min_count, _) in combos_constraints[expression].items()
                                   if card_type not in ("mulligans", "mulligan_count", "free_first_mulligan", "hand_size", "deck_size", "mulligan_type"))
                    else:
                        return 1
                elif isinstance(expression, tuple):
                    op, left, right = expression
                    match op:
                        case "AND":
                            return get_min_required(left) + get_min_required(right)
                        case "OR":
                            return min(get_min_required(left), get_min_required(right))
                        case "XOR":
                            return min(get_min_required(left), get_min_required(right))
                        case _:
                            return 0
                return 0

            min_required = get_min_required(set_config["expression"])
            max_mulls = mulligan_settings["count"] if mulligan_settings["enabled"] else 0
            if mulligan_settings["first_free"]:
                max_mulls += 1

            for _ in range(max_mulls):
                result = [0.0]
                def gen_initial(idx, curr, hand_size):
                    if idx == len(all_cards_this_combo):
                        if sum(curr.values()) != hand_size:
                            return
                        hand = []
                        for c, n in curr.items():
                            hand += [c] * n
                        if evaluate_combo_expression(hand, set_config["expression"], combos_constraints, card_types):
                            prob = 1.0
                            for c, n in curr.items():
                                prob *= comb(all_cards_this_combo[c], n)
                            result[0] += prob
                            return
                        non_contributing = sum(n for c, n in curr.items() if c == "Unspecified" or not any(c in card_types.get(t, [t]) for t in combos_constraints))
                        cards_to_replace = min(non_contributing, hand_size - min_required)
                        if cards_to_replace <= 0:
                            return
                        new_deck = dict(all_cards_this_combo)
                        for c, n in curr.items():
                            new_deck[c] -= n
                            if new_deck[c] < 0:
                                return
                        new_deck["Unspecified"] = new_deck.get("Unspecified", 0) + cards_to_replace
                        new_result = [0.0]
                        generate_counts(list(new_deck.keys()), cards_to_replace, defaultdict(int), 0, set_config["expression"], cards_to_replace, new_deck, new_result)
                        prob = 1.0
                        for c, n in curr.items():
                            prob *= comb(all_cards_this_combo[c], n)
                        result[0] += prob * (new_result[0] / comb(d - hand_size, cards_to_replace))
                    c = list(all_cards_this_combo.keys())[idx]
                    max_n = min(all_cards_this_combo[c], hand_size - sum(curr.values()))
                    for take in range(max_n + 1):
                        curr[c] = take
                        gen_initial(idx + 1, curr, hand_size)
                        curr[c] = 0

                gen_initial(0, defaultdict(int), h0)
                p_hits.append(result[0] / comb(d, h0))

        case _:
            raise ValueError(f"Unknown mulligan type: {mulligan_type}")

    if not extra_info or not mulligan_settings["enabled"]:
        prod = 1.0
        for p in p_hits:
            prod *= (1 - p)
        return 1 - prod

    info = {}
    p_fail = 1.0
    for i, p in enumerate(p_hits):
        label = "No Mulligan" if i == 0 else f"Mulligan {i}"
        succ = p * p_fail
        info[label] = succ
        p_fail *= (1 - p)
    info["Total"] = 1 - p_fail
    return info

def expand_deck(deck_counts):
    return [card for card, count in deck_counts.items() for _ in range(count)]

def get_combo_relevant_cards(combo_expr, combos, card_types):
    cards = set()
    def helper(expr):
        if isinstance(expr, str):
            if expr in combos:
                for key in combos[expr]:
                    if key not in ("mulligans", "mulligan_count", "free_first_mulligan", "hand_size", "deck_size", "mulligan_type"):
                        if key in card_types:
                            cards.update(card_types[key])
                        else:
                            cards.add(key)
            else:
                cards.add(expr)
        elif isinstance(expr, tuple):
            operator, left, right = expr
            helper(left)
            helper(right)
    helper(combo_expr)
    return cards

def build_full_deck_with_unspecified(deck_counts, deck_size):
    full_deck = []
    for card, count in deck_counts.items():
        full_deck.extend([card] * count)
    if deck_size > len(full_deck):
        full_deck.extend(["Unspecified"] * (deck_size - len(full_deck)))
    return np.array(full_deck)

def single_trial_combo_type(deck_counts, default_hand_size, card_types, combo_data):
    deck_size = combo_data.get("deck_size", sum(deck_counts.values()))
    hand_size = combo_data.get("hand_size", default_hand_size)
    full_deck = build_full_deck_with_unspecified(deck_counts, deck_size)
    indices = np.random.choice(len(full_deck), hand_size, replace=False)
    hand = full_deck[indices]
    return 1 if hand_satisfies_combo(hand, combo_data["requirements"], card_types) else 0

def monte_carlo_combo_type_probability(deck_counts, default_hand_size, card_types, combos, trials=10000, max_workers=None):
    results = {name: 0 for name in combos}
    batch_size = trials // max_workers if max_workers else trials
    def run_batch(start, end):
        hits = 0
        for _ in range(start, end):
            hits += single_trial_combo_type(deck_counts, default_hand_size, card_types, combos[name])
        return hits

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_batch, i, min(i + batch_size, trials)): name
            for name in combos
            for i in range(0, trials, batch_size)
        }
        for future in as_completed(futures):
            results[futures[future]] += future.result()

    for name in results:
        results[name] /= trials
    return results

def single_trial_combo_set_probability(deck_counts, deck_size, hand_size, combo_tree, combos, card_types):
    deck = build_full_deck_with_unspecified(deck_counts, deck_size)
    shuffle(deck)
    trial_deck = deck[:deck_size]
    hand = trial_deck[:hand_size]
    if evaluate_combo_expression(hand, combo_tree, combos, card_types):
        return 1
    return 0

def monte_carlo_combo_set_probability(deck_counts, global_hand_size, combos, combo_tree, set_cfg, card_types, trials=10000, max_workers=None):
    deck_size = set_cfg.get("deck_size", sum(deck_counts.values()))
    hand_size = set_cfg.get("hand_size", global_hand_size)

    full_deck = build_full_deck_with_unspecified(deck_counts, deck_size)

    if deck_size > len(full_deck):
        raise ValueError(f"Deck size {deck_size} exceeds available cards {len(full_deck)}")
    if hand_size > deck_size:
        raise ValueError(f"Hand size {hand_size} exceeds deck size {deck_size}")

    hits = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                single_trial_combo_set_probability,
                deck_counts,
                deck_size,
                hand_size,
                combo_tree,
                combos,
                card_types
            )
            for _ in range(trials)
        ]
        for future in as_completed(futures):
            hits += future.result()

    return hits / trials

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
        constraints["mulligans"] = combo["mulligans"]["enabled"]
        constraints["mulligan_count"] = combo["mulligans"]["count"]
        constraints["free_first_mulligan"] = combo["mulligans"]["first_free"]
        constraints["hand_size"] = combo["hand_size"]
        constraints["deck_size"] = combo["deck_size"]
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
        expression = parse_expression(data["expression"])
        if expression is not None:
            parsed_sets[name] = expression

    if parsed_sets:
        console.print("\n[bold magenta]Combo Set Probabilities[/bold magenta]\n")
        set_table = Table(show_header=True, header_style="bold yellow")
        set_table.add_column("Combo Set", style="bold")
        set_table.add_column("Label", style="green")
        set_table.add_column("Probability", justify="right", style="cyan")

        for name, expression in parsed_sets.items():
            cfg = {
                "expression": expression,
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

def page_estimate_probability():
    clear_screen()
    if not state['combos']:
        console.print("[error]Define at least one combo first.[/error]")
        pause()
        return

    console.print("[header][7] Estimate Probability (Monte Carlo Simulation)[/header]\n")

    card_counts = dict(state["cards"])
    combos_input = {}
    for name, combo in state['combos'].items():
        constraints = {
            "requirements": {
                T: tuple(bounds)
                for T, bounds in combo["requirements"].items()
            },
            "mulligans": combo["mulligans"]["enabled"],
            "mulligan_count": combo["mulligans"]["count"],
            "free_first_mulligan": combo["mulligans"]["first_free"],
            "hand_size": combo["hand_size"],
            "deck_size": combo["deck_size"]
        }
        combos_input[name] = constraints

    number_of_simulations = int(console.input("[prompt]How many simulations to run?: [/prompt]"))

    combo_results = monte_carlo_combo_type_probability(
        card_counts,
        state['hand_size'],
        state['card_types'],
        combos_input,
        trials=number_of_simulations
    )

    console.print("[bold magenta]Estimated Combo Probabilities[/bold magenta]\n")
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
        expression = parse_expression(data["expression"])
        if expression is not None:
            parsed_sets[name] = expression

    if parsed_sets:
        console.print("\n[bold magenta]Estimated Combo Set Probabilities[/bold magenta]\n")
        set_table = Table(show_header=True, header_style="bold yellow")
        set_table.add_column("Combo Set", style="bold")
        set_table.add_column("Label", style="green")
        set_table.add_column("Probability", justify="right", style="cyan")

        card_counts = dict(state["cards"])
        combos_input = {}
        for name, combo in state['combos'].items():
            constraints = {
                **{T: tuple(bounds) for T, bounds in combo["requirements"].items()},
                "mulligans": combo["mulligans"]["enabled"],
                "mulligan_count": combo["mulligans"]["count"],
                "free_first_mulligan": combo["mulligans"]["first_free"],
                "hand_size": combo["hand_size"],
                "deck_size": combo["deck_size"]
            }
            combos_input[name] = constraints

        for name, expression in parsed_sets.items():
            res = monte_carlo_combo_set_probability(
                card_counts,
                state['hand_size'],
                combos_input,
                expression,
                state['combo_sets'][name],
                state['card_types'],
                number_of_simulations
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
