# optuna.py

import optuna
import os

def objective(trial):
    # ハイパーパラメータの候補を設定
    lr = trial.suggest_float('lr', 1e-4, 1e-3, step=1e-4)
    lamda = trial.suggest_float('lamda', 0.005, 1, step=0.001)
    alpha = trial.suggest_float('alpha', 0.005, 1, step=0.001)
    
    # main.pyをsubprocessを使って実行
    cmd = f"python main-autotune.py --lr {lr} --lamda {lamda} --alpha {alpha}"

    # os.systemを使用してコマンドを実行
    result = os.system(cmd)
    
    # main.pyから返される損失を取得
    # 例えば、main.pyが"Loss: xxx"という形式で損失を出力すると仮定
    with open('auto-tune.txt', 'r') as f:
        auc = float(f.read())
    
    return auc

# Optunaの学習
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1)

# 最良のハイパーパラメータを表示
print(study.best_params)
# [2023-08-28 00:26:35,615][main-autotune.py][line:103][INFO] [Epoch:1/2]: lr:0.00050 | loss1:0.4189 loss2:1.2099 loss3:0.2794 | AUC:0.7970 Anomaly AUC:0.5909
# [2023-08-28 00:27:02,875][main-autotune.py][line:103][INFO] [Epoch:2/2]: lr:0.00050 | loss1:0.2427 loss2:1.0579 loss3:0.1092 | AUC:0.8234 Anomaly AUC:0.6212
# [2023-08-28 00:27:02,876][main-autotune.py][line:111][INFO] Training completes in 0m 54s | best AUC:0.8234 Anomaly AUC:0.6212

# [I 2023-08-28 00:27:03,603] Trial 0 finished with value: 0.8233690783684862 and parameters: {'lr': 0.0005, 'lamda': 0.055, 'alpha': 0.529}. Best is trial 0 with value: 0.8233690783684862.