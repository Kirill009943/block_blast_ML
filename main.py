"""Block Blast launcher.

Usage:
    python main.py                      # graphical UI, human mode (default)
    python main.py --mode watch         # watch an AI play (RL if a model exists)
    python main.py --mode watch --agent rl:models/main_final.zip
    python main.py --mode console       # original text-mode game
    python main.py --mode record        # human training: record demonstrations
    python main.py --mode feedback      # accept/override AI suggestions (recorded)

The console mode preserves the feel of the original implementation but
runs on the refactored engine in ``game/``.
"""

from __future__ import annotations

import argparse

from game import BOARD_SIZE, BlockBlastGame

SYMBOLS = "." + "ABCDEFGH"


def render(game: BlockBlastGame) -> None:
    print("\nScore:", game.score)
    print("   " + " ".join(str(i) for i in range(BOARD_SIZE)))
    for row_index, row in enumerate(game.board):
        print(f"{row_index:2} " + " ".join(SYMBOLS[cell] for cell in row))
    print("\nAvailable pieces:")
    for idx, piece in enumerate(game.pieces):
        if piece is None:
            print(f"  {idx}: used")
        else:
            print(f"  {idx}: {SYMBOLS[piece.color]} {piece.name} -> {list(piece.cells)}")


def read_move(game: BlockBlastGame):
    while True:
        try:
            choice = input("Choose a piece index (0-2): ").strip()
            if not choice:
                continue
            piece_index = int(choice)
            if 0 <= piece_index < len(game.pieces) and game.pieces[piece_index] is not None:
                break
            print("Invalid or already-used piece index.")
        except ValueError:
            print("Please enter a number.")

    while True:
        try:
            pos = input("Enter row and column (example: 2 3): ").split()
            if len(pos) != 2:
                raise ValueError
            row, col = map(int, pos)
            return piece_index, row, col
        except ValueError:
            print("Invalid position. Use: row col")


def play_console(seed=None) -> None:
    game = BlockBlastGame(seed=seed)
    print("Welcome to Block Blast!")
    while not game.is_game_over():
        render(game)
        piece_index, row, col = read_move(game)
        if not game.place_piece(piece_index, row, col).success:
            print("That placement is not valid. Try again.")
            continue
        print(f"Placed successfully! (+{game.score} total)")
    render(game)
    print("Game over! Final score:", game.score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block Blast with a self-learning AI.")
    parser.add_argument("--mode", choices=["human", "watch", "console", "brain",
                                           "record", "feedback"],
                        default="human",
                        help="human: play in the Pygame UI (default); "
                             "watch: watch an AI play; console: text mode; "
                             "brain: watch the RL agent with live decision visualization; "
                             "record: human training mode (records demonstrations); "
                             "feedback: accept/override AI suggestions (recorded)")
    parser.add_argument("--agent", type=str, default=None,
                        help="watch/brain/feedback mode agent: heuristic | random | "
                             "solver | imitation:<path> | rl:<path> "
                             "(default: best available)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ai-delay", type=float, default=0.3,
                        help="seconds between AI moves in watch/brain mode")
    parser.add_argument("--dataset-dir", type=str, default="results/demos",
                        help="where record/feedback modes save demonstrations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "console":
        play_console(seed=args.seed)
        return

    from training.evaluate import build_agent
    from ui.pygame_app import load_default_agent, run_app

    if args.mode == "watch":
        agent = build_agent(args.agent, seed=args.seed) if args.agent else load_default_agent()
        app_agent = agent
        from ui.pygame_app import PygameApp
        app = PygameApp(agent=app_agent, seed=args.seed)
        app.ai_delay = args.ai_delay
        app.run()
    elif args.mode == "brain":
        spec = args.agent or "rl:models/main_final.zip"
        agent = build_agent(spec, seed=args.seed)
        if not hasattr(agent, "action_probabilities"):
            raise SystemExit("Brain mode requires an RL agent (use --agent rl:<path>)")
        from ui.brain_app import run_brain

        run_brain(agent, seed=args.seed or 0, ai_delay=args.ai_delay)
    elif args.mode == "record":
        from ui.record_app import run_record_app

        run_record_app(dataset_dir=args.dataset_dir, seed=args.seed)
    elif args.mode == "feedback":
        from ui.record_app import run_feedback_app

        agent = (build_agent(args.agent, seed=args.seed)
                 if args.agent else load_default_agent())
        run_feedback_app(agent, dataset_dir=args.dataset_dir, seed=args.seed)
    else:
        run_app(agent=None, human=True, seed=args.seed)


if __name__ == "__main__":
    main()
