# optuna.py

import optuna
import os
from optuna.samplers import TPESampler
import subprocess



def objective(trial):
    # ハイパーパラメータの候補を設定
    pel_lr = trial.suggest_float('pel_lr', 1e-4, 1e-3, step=1e-4)
    ur_lr = trial.suggest_float('ur_lr', 1e-4, 1e-3, step=1e-4)
    lamda = trial.suggest_float('lamda', 0.005, 1, step=0.001)
    alpha = trial.suggest_float('alpha', 0.005, 1, step=0.001)
    
    # # main.pyをsubprocessを使って実行
    # cmd = f"python main-autotune.py --lr {lr} --lamda {lamda} --alpha {alpha}"

    # # os.systemを使用してコマンドを実行
    # result = os.system(cmd)
    result = subprocess.run(['python', 'main-autotune.py', 
                            '--PEL_lr', str(pel_lr),
                            '--UR_DMU_lr', str(ur_lr),
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
study.optimize(objective, n_trials=10)

# 最良のハイパーパラメータを表示
print(study.best_params)