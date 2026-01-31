import argparse
import cchess
from rich.progress import track

from game import Game
from net import PolicyValueNet
from mcts import MCTS_AI
from mcts_pure import MCTS_Pure
from parameters import C_PUCT, PLAYOUT
from tools import log, move_id2move_action


class MoveAdapter:
    """Adapt players that return move IDs to the interface expected by ``Game``."""

    def __init__(self, player):
        self.player = player

    def set_player_idx(self, idx):
        if hasattr(self.player, "set_player_idx"):
            self.player.set_player_idx(idx)

    def reset_player(self):
        if hasattr(self.player, "reset_player"):
            self.player.reset_player()

    def get_action(self, board):
        move = self.player.get_action(board)
        if isinstance(move, cchess.Move) or move is None:
            return move
        return cchess.Move.from_uci(move_id2move_action[move])


def build_nn_player(
    model_path: str | None, n_playout: int, c_puct: float
) -> tuple[MCTS_AI, PolicyValueNet]:
    """Instantiate the neural-network guided MCTS player."""
    try:
        policy_value_net = (
            PolicyValueNet(model=model_path) if model_path else PolicyValueNet()
        )
    except Exception as exc:  # pragma: no cover - defensive
        log(f"Failed to load model {model_path}: {exc}", "ERROR")
        policy_value_net = PolicyValueNet()
    player = MCTS_AI(
        policy_value_net.policy_value_fn,
        c_puct=c_puct,
        n_playout=n_playout,
        is_selfplay=False,
    )
    return player, policy_value_net


def evaluate(
    model_path: str | None,
    games: int,
    nn_playout: int,
    pure_playout: int,
    c_puct: float,
    pure_rollout_limit: int,
    show: bool,
) -> None:
    """Play a series of matches between the neural player and the pure MCTS baseline."""
    nn_player, _ = build_nn_player(model_path, nn_playout, c_puct)
    pure_player = MCTS_Pure(
        c_puct=c_puct,
        n_playout=pure_playout,
        rollout_limit=pure_rollout_limit,
    )
    nn_adapter = MoveAdapter(nn_player)
    pure_adapter = MoveAdapter(pure_player)
    game = Game(cchess.Board())

    stats = {"nn": 0, "pure": 0, "draw": 0}

    for game_idx in track(range(games), description="Evaluating", transient=True):
        nn_is_red = game_idx % 2 == 0
        red_player = nn_adapter if nn_is_red else pure_adapter
        black_player = pure_adapter if nn_is_red else nn_adapter

        # Reset search trees before a new game starts
        for player in (red_player, black_player):
            if hasattr(player, "reset_player"):
                try:
                    player.reset_player()
                except Exception:  # pragma: no cover - defensive
                    pass

        winner = game.start_play(red_player, black_player, is_shown=show)

        if winner == cchess.RED:
            stats["nn" if nn_is_red else "pure"] += 1
        elif winner == cchess.BLACK:
            stats["pure" if nn_is_red else "nn"] += 1
        else:
            stats["draw"] += 1

        log(
            f"Game {game_idx + 1}/{games} finished: "
            f"winner={'RED' if winner == cchess.RED else ('BLACK' if winner == cchess.BLACK else 'DRAW')}"
        )

    nn_wins = stats["nn"]
    pure_wins = stats["pure"]
    draws = stats["draw"]
    total = max(1, games)
    win_rate = nn_wins / total

    log(
        "Evaluation summary | "
        f"NN wins: {nn_wins} | Pure wins: {pure_wins} | Draws: {draws} | NN win rate: {win_rate:.3f}"
    )

    print(
        "Match Results:\n"
        f"  Neural MCTS wins: {nn_wins}\n"
        f"  Pure MCTS wins: {pure_wins}\n"
        f"  Draws: {draws}\n"
        f"  Neural MCTS win rate: {win_rate:.3%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate NN-guided MCTS against pure MCTS"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path to the trained model file"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of games to play (alternating colors)",
    )
    parser.add_argument(
        "--nn-playout",
        type=int,
        default=PLAYOUT,
        help="Playouts for the neural MCTS player",
    )
    parser.add_argument(
        "--pure-playout",
        type=int,
        default=2000,
        help="Playouts for the pure MCTS baseline",
    )
    parser.add_argument(
        "--c-puct", type=float, default=C_PUCT, help="PUCT constant for both players"
    )
    parser.add_argument(
        "--pure-rollout-limit",
        type=int,
        default=300,
        help="Maximum depth for each pure MCTS rollout before declaring a draw",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display board state after each move"
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        games=args.games,
        nn_playout=args.nn_playout,
        pure_playout=args.pure_playout,
        c_puct=args.c_puct,
        pure_rollout_limit=args.pure_rollout_limit,
        show=args.show,
    )


if __name__ == "__main__":
    main()
