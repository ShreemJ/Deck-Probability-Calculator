from config import *
from utility import *

def get_card_type_quantity(card_type_name):
    if card_type_name in state["card_type_cache"]:
        return state["card_type_cache"][card_type_name]
    qty: int = sum(state["cards"].get(card, 0) for card in state["card_types"].get(card_type_name, []))
    state["card_type_cache"][card_type_name] = qty
    return qty

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
        match choice:
            case "1":
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
            case "2":
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
            case "3" | "":
                return
            case _:
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
        match choice:
            case "1":
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
            case "2":
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
            case "3" | "":
                return
            case _:
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
        match choice:
            case "1":
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
            case "2":
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
                match sub:
                    case "1":
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
                    case "2":
                        del state['card_types'][ctype_name]
                        update_card_type_cache()
                    case _:
                        console.print("[error]Invalid option.[/error]")
                        pause()
            case "3" | "":
                return
            case _:
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

    match choice:
        case "1":
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
                for index, mulligan_type, mulligan_description in enumerate(zip(mulligan_types, mulligan_descriptions)):
                    console.print(f"[info]{index + 1}. {mulligan_type} ({mulligan_description})[/info]")
                mulligan_choice = console.input("[prompt]> [/prompt]").strip()
                match mulligan_choice:
                    case "1":
                        mulligans["type"] = "traditional"
                    case "2":
                        mulligans["type"] = "london"
                    case "3":
                        mulligans["type"] = "partial"
                    case _:
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

        case "2":
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
            match sub:
                case "1":
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
                            match mulligan_choice:
                                case "1":
                                    mulligans["type"] = "traditional"
                                case "2":
                                    mulligans["type"] = "london"
                                case "3":
                                    mulligans["type"] = "partial"
                                case _:
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
                case "2":
                    del state["combos"][name]
                case _:
                    console.print("[error]Invalid option.[/error]")
                    pause()
        case "3" | "":
            return
        case _:
            console.print("[error]Invalid option.[/error]")
            pause()

def validate_combo_set(expr):
    stack = 0
    last = "operator"
    for token in expr:
        match token:
            case "(":
                stack += 1
                last = "open"
            case ")":
                stack -= 1
                if stack < 0:
                    return False
                last = "close"
            case "AND" | "OR" | "XOR":
                if last == "operator":
                    return False
                last = "operator"
            case _:
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

        match choice:
            case "1":
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
                    match tok:
                        case "D":
                            break
                        case "A" | "O" | "X":
                            expr.append({"A": "AND", "O": "OR", "X": "XOR"}[tok])
                        case "B":
                            expr.append(console.input("[prompt]Enter '(' or ')': [/prompt]").strip())
                        case _ if tok.startswith("C") and tok[1:].isdigit() and int(tok[1:]) < len(combo_keys):
                            expr.append(combo_keys[int(tok[1:])])
                        case _ if tok.startswith("K") and tok[1:].isdigit() and int(tok[1:]) < len(card_keys):
                            expr.append(card_keys[int(tok[1:])])
                        case _:
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
                    match mulligan_choice:
                        case "1":
                            mull["type"] = "traditional"
                        case "2":
                            mull["type"] = "london"
                        case "3":
                            mull["type"] = "partial"
                        case _:
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

            case "2":
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
                match sub:
                    case "2":
                        del state['combo_sets'][key]
                        console.print(f"[success]Removed '{key}'.[/success]")
                        pause()
                        continue
                    case "1":
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
                                match tok:
                                    case "D":
                                        break
                                    case "A" | "O" | "X":
                                        expr.append({"A": "AND", "O": "OR", "X": "XOR"}[tok])
                                    case "B":
                                        expr.append(console.input("[prompt]Enter '(' or ')': [/prompt]").strip())
                                    case _ if tok.startswith("C") and tok[1:].isdigit() and int(tok[1:]) < len(combo_keys):
                                        expr.append(combo_keys[int(tok[1:])])
                                    case _ if tok.startswith("K") and tok[1:].isdigit() and int(tok[1:]) < len(card_keys):
                                        expr.append(card_keys[int(tok[1:])])
                                    case _:
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
                                match mulligan_choice:
                                    case "1":
                                        m["type"] = "traditional"
                                    case "2":
                                        m["type"] = "london"
                                    case "3":
                                        m["type"] = "partial"
                                    case _:
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
                    case _:
                        console.print("[error]Invalid option.[/error]")
                        pause()
            case "3" | "":
                return
            case _:
                console.print("[error]Invalid option.[/error]")
                pause()

def page_list():
    clear_screen()
    console.print("[header][10] State Summary[/header]\n")
    items_per_page = 20
    page = 0
    all_items = (
        [(f"Deck size: {state['deck_size']}", None)] +
        [(f"Hand size: {state['hand_size']}", None)] +
        [(f"Card: {card}: {qty}", "card") for card, qty in state['cards'].items()] +
        [(f"Card Type: {t}: {members}", "type") for t, members in state['card_types'].items()] +
        [(f"Combo: {name}: {combo}", "combo") for name, combo in state['combos'].items()] +
        [(f"Combo Set: {name}: {expr}", "combo_set") for name, expr in state['combo_sets'].items()]
    )

    while True:
        clear_screen()
        console.print("[header][10] State Summary[/header]\n")
        start = page * items_per_page
        end = min(start + items_per_page, len(all_items))
        for item, _ in all_items[start:end]:
            console.print(f"[info]{item}[/info]")
        console.print(f"\n[info]Page {page + 1} of {(len(all_items) + items_per_page - 1) // items_per_page}[/info]")
        console.print("[info]N: Next page, P: Previous page, Q: Quit[/info]")
        choice = console.input("[prompt]> [/prompt]").lower()
        if choice == "n" and end < len(all_items):
            page += 1
        elif choice == "p" and page > 0:
            page -= 1
        elif choice == "q" or choice == "":
            break
