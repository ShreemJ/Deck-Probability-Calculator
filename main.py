from sys import exit

from probabilities import page_calculate_probability, page_estimate_probability
from config import *
from utility import *
from graphing import page_graph
from loading import *
from data_management import *


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
        console.print("[info]7.[/info] [text]Estimate Probability[/text]")
        console.print("[info]8.[/info] [text]Save to File[/text]")
        console.print("[info]9.[/info] [text]Load Deck/JSON[/text]")
        console.print("[info]10.[/info] [text]Show All State[/text]")
        console.print("[info]11.[/info] [text]Graphs[/text]")
        console.print("[info]12.[/info] [text]Exit[/text]")
        choice = console.input("[prompt]> [/prompt]")

        match choice:
            case "1":
                page_deck_hand_size()
            case "2":
                page_cards()
            case "3":
                page_card_types()
            case "4":
                page_combos()
            case "5":
                page_combo_sets()
            case "6":
                page_calculate_probability()
            case "7":
                page_estimate_probability()
            case "8":
                page_save()
            case "9":
                import_deck_prompt(from_first_screen=False)
            case "10":
                page_list()
            case "11":
                page_graph()
            case "12":
                page_exit()
            case _:
                console.print("[error]Invalid option.[/error]")
                pause()


if __name__ == "__main__":
    main()
