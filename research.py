import matplotlib.pyplot as plt
import pandas as pd

def plot_video_prediction(preds, ground_truth, file_name):
    preds = restore_repeated_list(preds)
    ground_truth = restore_repeated_list(ground_truth)
    plt.figure(figsize=(10, 5))
    plt.plot(preds, label='Predictions', color='blue')
    plt.plot(ground_truth, label='Ground Truth', color='red')
    plt.xlabel('Frame Number')
    plt.ylabel('Prediction Value')
    plt.title('Video Prediction vs Ground Truth')
    plt.legend()
    plt.savefig(file_name)
    plt.close()

def restore_repeated_list(repeated_list):
    original_length = len(repeated_list) // 16
    restored_list = [repeated_list[i * 16] for i in range(original_length)]
    return restored_list

def make_csv(preds, ground_truth, file_name, seq_len_list):
    seq_len_list = seq_len_list if isinstance(seq_len_list, list) else seq_len_list.cpu().detach().tolist()
    
    video_ids = []
    for i, seq_len in enumerate(seq_len_list):
        video_ids.extend([i] * seq_len)

    df = pd.DataFrame({'video_id': video_ids, 'pred': restore_repeated_list(preds), 'gt': restore_repeated_list(ground_truth)})
    df.to_csv(file_name, index=False)


    