# optuna.py

import optuna
import os
from optuna.samplers import TPESampler
import subprocess



def objective(trial):
    # ハイパーパラメータの候補を設定
    # pel_lr = trial.suggest_float('pel_lr', 1e-4, 1e-3, step=1e-4)
    # ur_lr = trial.suggest_float('ur_lr', 1e-4, 1e-3, step=1e-4)
    possible_values = [i/10 for i in range(1, 10)] + [i/100 for i in range(1, 10)] + [i/1000 for i in range(1, 10)] + [1]
    lamda = trial.suggest_categorical('lamda', possible_values)
    alpha = trial.suggest_categorical('alpha', possible_values)
    k = trial.suggest_categorical('k', range(10, 30))
    t_step = trial.suggest_categorical('t_step', range(3, 10))
    win_size = trial.suggest_categorical('win_size', range(3, 10))
    gamma = trial.suggest_categorical('gamma', possible_values)
    bias = trial.suggest_categorical('bias', possible_values)
    mem_nums = trial.suggest_categorical('mem_nums', range(10, 90, 10))
    
    
    # # main.pyをsubprocessを使って実行
    # cmd = f"python main-autotune.py --lr {lr} --lamda {lamda} --alpha {alpha}"

    # # os.systemを使用してコマンドを実行
    # result = os.system(cmd)
    result = subprocess.run(['python', 'main-autotune.py', 
                             '--lamda', str(lamda),
                             '--alpha', str(alpha),
                             '--k', str(k),
                             '--t_step', str(t_step),
                             '--win_size', str(win_size),
                             '--gamma', str(gamma),
                             '--bias',str(bias),
                             '--mem_num',str(mem_nums)],
                            capture_output=True, text=True)

    print(result.stdout)
    # main.pyから返される損失を取得
    # 例えば、main.pyが"Loss: xxx"という形式で損失を出力すると仮定
    try:
        auc = float(result.stdout.rstrip().split("\n")[-1])
    except:
        print(result.stderr)
    
    return auc

# Optunaの学習
study = optuna.create_study(sampler=TPESampler(seed=2024), direction='maximize')
study.enqueue_trial({'lamda': 1, 'alpha': 1, 'k': 20, 't_step': 9, 'win_size': 9, 'gamma': 0.6, 'bias': 0.2, 'mem_nums': 50})
study.optimize(objective, n_trials=70)

# 最良のハイパーパラメータを表示
print(study.best_params)