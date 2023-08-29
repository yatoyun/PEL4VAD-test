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
    lr = trial.suggest_float('lr', 1e-4, 1e-3, step=1e-4)
    
    # # main.pyをsubprocessを使って実行
    # cmd = f"python main-autotune.py --lr {lr} --lamda {lamda} --alpha {alpha}"

    # # os.systemを使用してコマンドを実行
    # result = os.system(cmd)
    result = subprocess.run(['python', 'main-autotune.py', 
                            '--lr', str(lr),
                            '--lamda', str(lamda),
                            '--alpha', str(alpha)],
                            capture_output=True, text=True)

    print(result.stdout)
    # main.pyから返される損失を取得
    # 例えば、main.pyが"Loss: xxx"という形式で損失を出力すると仮定
    auc = float(result.stdout.rstrip().split("\n")[-1])
    
    return auc

# Optunaの学習
study = optuna.create_study(sampler=TPESampler(seed=2024), direction='maximize')
study.optimize(objective, n_trials=50)

# 最良のハイパーパラメータを表示
print(study.best_params)