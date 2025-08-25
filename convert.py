import h5py
import pickle
import numpy as np
import time

def convert_h5_to_optimized_pkl(h5_path="data.h5", pkl_path="data.pkl"):
    """将HDF5转换为优化的pickle格式"""
    print(f"[{time.strftime('%H:%M:%S')}] 开始转换 {h5_path} -> {pkl_path}")
    
    # 预先分配数组来存储所有数据
    all_states = []
    all_mcts_probs = []
    all_winners = []
    total_steps = 0
    
    with h5py.File(h5_path, "r") as h5f:
        games_count = h5f.attrs.get("iters", 0)
        print(f"总游戏数: {games_count}")
        
        # 第一遍扫描，计算总步数
        for game_idx in range(games_count):
            game_group = h5f.get(f"game_{game_idx}")
            if game_group is not None and "states" in game_group:
                steps = game_group["states"].shape[0]
                total_steps += steps
        
        print(f"总步数: {total_steps}")
        
        # 预分配numpy数组
        sample_state = h5f["game_0"]["states"][0]
        sample_mcts = h5f["game_0"]["mcts_probs"][0]
        
        print(f"预分配数组...")
        states_array = np.empty((total_steps,) + sample_state.shape, dtype=sample_state.dtype)
        mcts_array = np.empty((total_steps,) + sample_mcts.shape, dtype=sample_mcts.dtype)
        winners_array = np.empty(total_steps, dtype=np.float32)
        
        # 第二遍扫描，复制数据
        current_idx = 0
        for game_idx in range(games_count):
            if game_idx % 100 == 0:
                print(f"处理游戏 {game_idx}/{games_count}")
                
            game_group = h5f.get(f"game_{game_idx}")
            if game_group is not None and "states" in game_group:
                states = game_group["states"][:]
                mcts_probs = game_group["mcts_probs"][:]
                winners = game_group["winners"][:]
                
                steps = states.shape[0]
                states_array[current_idx:current_idx+steps] = states
                mcts_array[current_idx:current_idx+steps] = mcts_probs
                winners_array[current_idx:current_idx+steps] = winners
                
                current_idx += steps
    
    # 保存为优化的格式
    data_dict = {
        'states': states_array,
        'mcts_probs': mcts_array,
        'winners': winners_array,
        'total_count': total_steps
    }
    
    print(f"[{time.strftime('%H:%M:%S')}] 保存到 {pkl_path}...")
    with open(pkl_path, "wb") as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"[{time.strftime('%H:%M:%S')}] 转换完成！总共 {total_steps} 步棋")
    return True

if __name__ == "__main__":
    convert_h5_to_optimized_pkl()
