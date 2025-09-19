import h5py
import json
import numpy as np
import os
import time
from parameters import DATA_DIR
from tools import log


def _derive_out_paths(out_dir: str):
    """仅支持目录输出：固定文件名 states.npy/mcts.npy/winners.npy/meta.json"""
    root = out_dir if out_dir else "."
    return (
        os.path.join(root, "states.npy"),
        os.path.join(root, "mcts.npy"),
        os.path.join(root, "winners.npy"),
        os.path.join(root, "meta.json"),
    )


def convert_h5_to_npy(h5_path: str = None, out_dir: str = None):
    """将 data.h5 转换为目录下固定命名的 .npy 与 meta.json 文件。"""
    if h5_path is None:
        h5_path = os.path.join(DATA_DIR, "data.h5")
    if out_dir is None:
        out_dir = DATA_DIR
    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)

    out_states, out_mcts, out_winners, out_meta = _derive_out_paths(out_dir)

    log(f"Start converting {h5_path}")
    log(f"Output files:\n  - {out_states}\n  - {out_mcts}\n  - {out_winners}\n  - {out_meta}")

    total_steps = 0
    with h5py.File(h5_path, "r") as h5f:
        games_count = h5f.attrs.get("iters", 0)
        log(f"Total games: {games_count}")

        # First pass: count total steps
        for game_idx in range(games_count):
            game_group = h5f.get(f"game_{game_idx}")
            if game_group is not None and "states" in game_group:
                steps = game_group["states"].shape[0]
                total_steps += steps

        log(f"Total steps: {total_steps}")

        # Pre-allocate numpy arrays
        sample_state = h5f["game_0"]["states"][0]
        sample_mcts = h5f["game_0"]["mcts_probs"][0]

        log("Pre-alloc arrays and aggregate...")
        states_array = np.empty(
            (total_steps,) + sample_state.shape, dtype=sample_state.dtype
        )
        mcts_array = np.empty(
            (total_steps,) + sample_mcts.shape, dtype=sample_mcts.dtype
        )
        winners_array = np.empty(total_steps, dtype=np.float32)

        # Second pass: copy data
        current_idx = 0
        for game_idx in range(games_count):
            if game_idx % 100 == 0:
                log(f"Processing game {game_idx}/{games_count}")

            game_group = h5f.get(f"game_{game_idx}")
            if game_group is not None and "states" in game_group:
                states = game_group["states"][:]
                mcts_probs = game_group["mcts_probs"][:]
                winners = game_group["winners"][:]

                steps = states.shape[0]
                states_array[current_idx : current_idx + steps] = states
                mcts_array[current_idx : current_idx + steps] = mcts_probs
                winners_array[current_idx : current_idx + steps] = winners

                current_idx += steps

    # 保存为 .npy 三文件 + meta
    log(f"Saving to {out_states}, {out_mcts}, {out_winners}...")
    np.save(out_states, states_array)
    np.save(out_mcts, mcts_array)
    np.save(out_winners, winners_array)

    meta = {
        "total_count": int(total_steps),
        "states_shape": list(states_array.shape),
        "states_dtype": str(states_array.dtype),
        "mcts_shape": list(mcts_array.shape),
        "mcts_dtype": str(mcts_array.dtype),
        "winners_shape": list(winners_array.shape),
        "winners_dtype": str(winners_array.dtype),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log(f"Done! Total {total_steps} steps; generated .npy and meta.json")
    return True


if __name__ == "__main__":
    # 默认读取 data/data.h5，输出到 data/ 下的 states.npy/mcts.npy/winners.npy/meta.json
    convert_h5_to_npy()
