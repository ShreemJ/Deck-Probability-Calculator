from matplotlib.ticker import MaxNLocator
import csv
from pyautogui import size
import matplotlib.pyplot as plt
import numpy as np
from json import load,dump
from rich.table import Table


from utility import *
from probabilities import *
from config import console


def graph_set_up():
    width, height = size()
    fig, ax = plt.subplots(figsize=(width/150, height/150))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
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
    return fig, ax


def page_graph():
    def graph_settings():
        console.print("\n[info]Customize plot style:[/info]")
        console.print("[info]Line style: 1. Solid (-), 2. Dashed (--), 3. Dotted (:)[/info]")
        line_style = {'1': '-', '2': '--', '3': ':'}.get(
            console.input("[prompt]> [/prompt]") if not load_settings or 'line_style' not in settings else settings['line_style'], '-')
        console.print("[info]Marker style: 1. Circle (o), 2. Square (s), 3. Triangle (^)[/info]")
        marker_style = {'1': 'o', '2': 's', '3': '^'}.get(
            console.input("[prompt]> [/prompt]") if not load_settings or 'marker_style' not in settings else settings['marker_style'], 'o')
        colors = ['cyan', 'magenta', 'yellow', 'green', 'red', 'blue', 'orange', 'pink']
        return line_style,marker_style,colors

    clear_screen()
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

    if graph_type == "2":
        line_style, marker_style, colors = graph_settings()
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
                expression = parse_expression(combo_set_data["expression"])
                if expression is None:
                    console.print(f"[error]Invalid combo set expression for {selected_name}.[/error]")
                    pause()
                    return
                cfg = {
                    "expression": expression,
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
        
        fig, ax = graph_set_up()
        ax.set_xlabel(x_label, color='white')
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
        
        def get_y(y_axis, k, N, K, n, k2 = 0):
            if y_axis == "1":
                return hypergeom.pmf(k, N, K, n)
            if y_axis == "2":
                return 1 - hypergeom.pmf(k, N, K, n)
            if y_axis == "3":
                return hypergeom.cdf(k, N, K, n)
            if y_axis == "4":
                return 1 - hypergeom.cdf(k - 1, N, K, n)
            if y_axis == "5":
                return hypergeom.cdf(k2, N, K, n) - hypergeom.cdf(k - 1, N, K, n)
            return 0
        
        def get_ylabel(y_axis):
            return {
                "1": "P(X = k)",
                "2": "P(X ≠ k)",
                "3": "P(X ≤ k)",
                "4": "P(X ≥ k)",
                "5": "P(k_1 ≤ X ≤ k_2)"
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
            line_style, marker_style, colors = graph_settings()
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

            
            for key, label in y_axis_options.items():
                console.print(f"[info]{key}. {label}[/info]")
            y_axis = console.input("[prompt]> [/prompt]") if not load_settings or 'y_axis' not in settings else settings['y_axis']
            if y_axis not in y_axis_options:
                console.print("[error]Invalid y-axis choice.[/error]")
                pause()
                return
            
            if y_axis != "5":
                k = console.input("[prompt]Set k (≥0) > [/prompt]") if not load_settings or 'k' not in settings else str(settings['k'])
                if not (k.isdigit() and int(k) >= 0):
                    console.print("[error]Invalid k.[/error]")
                    pause()
                    return
                k = int(k)
                k2 = 0
            else:
                k = console.input("[prompt]Set k_1 (≥0) > [/prompt]") if not load_settings or 'k' not in settings else str(settings['k'])
                if not (k.isdigit() and int(k) >= 0):
                    console.print("[error]Invalid k.[/error]")
                    pause()
                    return
                k = int(k)
                k2 = console.input("[prompt]Set k_2 (≥k_1) > [/prompt]") if not load_settings or 'k' not in settings else str(settings['k'])
                if not (k2.isdigit() and int(k2) >= k):
                    console.print("[error]Invalid k.[/error]")
                    pause()
                    return
                k2 = int(k2)
            
            x = np.arange(x_min, x_max + 1)
            
            def get_param_value(param, val_x, val_vary, val_fixed):
                if param == x_axis:
                    return val_x
                elif param == vary_choice:
                    return val_vary
                else:
                    return val_fixed
            
            fig, ax = graph_set_up()
            
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
                    y_vals.append(get_y(y_axis, k, N, K, n, k2))
                
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
                    elif y_axis == "5":
                        y_vals.insert(0, 1 if k == 0 else 0)
                ax.plot(x, y_vals, marker=marker_style, linestyle=line_style, color=colors[i % len(colors)], label=f"{vary_label} = {v}")
                ax.scatter(max_x, max_y, color=colors[i % len(colors)], marker='*', s=200, zorder=5)
                ax.scatter(min_x, min_y, color=colors[i % len(colors)], marker='v', s=200, zorder=5)
            
            ax.set_xlabel(x_label, color='white')
            
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
                    val_y = get_y(y_axis, k, N, K, n, k2)
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

            for key, label in y_axis_options.items():
                console.print(f"[info]{key}. {label}[/info]")
            y_axis = console.input("[prompt]> [/prompt]") if not load_settings or 'y_axis' not in settings else settings['y_axis']
            if y_axis not in y_axis_options:
                console.print("[error]Invalid probability type.[/error]")
                pause()
                return
            if y_axis != "5":
                k = console.input("[prompt]Set k (≥0) > [/prompt]") if not load_settings or 'k' not in settings else str(settings['k'])
                if not (k.isdigit() and int(k) >= 0):
                    console.print("[error]Invalid k.[/error]")
                    pause()
                    return
                k = int(k)
                k2 = 0
            else:
                k = console.input("[prompt]Set k_1 (≥0) > [/prompt]") if not load_settings or 'k' not in settings else str(settings['k'])
                if not (k.isdigit() and int(k) >= 0):
                    console.print("[error]Invalid k.[/error]")
                    pause()
                    return
                k = int(k)
                k2 = console.input("[prompt]Set k_2 (≥k_1) > [/prompt]") if not load_settings or 'k' not in settings else str(settings['k'])
                if not (k2.isdigit() and int(k2) >= k):
                    console.print("[error]Invalid k.[/error]")
                    pause()
                    return
                k2 = int(k2)
            
            
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
                    Z[i, j] = get_y(y_axis, k, N, K, n, k2)
            
            width, height = size()
            fig = plt.figure(figsize=(width/150, height/150))
            
            if plot_type == "1":
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='white', linewidth=0.5, alpha=0.8)
                ax.set_facecolor('black')
                fig.patch.set_facecolor('black')
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', rstride=1, cstride=1, shade=True)
                ax.view_init(elev=30, azim=45)
                ax.set_xlabel(x_label, color='white')
                ax.set_ylabel(y_label, color='white')
                ax.set_zlabel(get_ylabel(y_axis), color='white')
                ax.tick_params(colors='white')
                ax.set_zticks(np.arange(0, 1.01, 0.1))
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
                fig.patch.set_facecolor('black')
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
            
            plt.show()

            console.print("[info]Export table to CSV? (y/n)[/info]")
            export_csv = console.input("[prompt]> [/prompt]").lower() == 'y'
            if export_csv:
                csv_filename = console.input("[prompt]Enter CSV filename (default: hypergeometric_table.csv) > [/prompt]")
                if not csv_filename:
                    csv_filename = "hypergeometric_table.csv"
                if not csv_filename.endswith('.csv'):
                    csv_filename += '.csv'

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
            
    
    pause()
