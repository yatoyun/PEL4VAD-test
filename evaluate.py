import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate(dataloader, encoder, classifier, args, noisy_label, clean_label, i, stat_logs, logger_wandb):
    encoder.eval()
    classifier.eval()
    feature_bank = []
    prediction = []

    ################################### feature extraction ###################################
    with torch.no_grad():
        # generate feature bank
        for (data, target, _, index) in tqdm(dataloader, desc='Feature extracting'):
            data = data.cuda()
            feature = encoder(data)
            feature_bank.append(feature)
            res = classifier(feature)
            prediction.append(res)
        feature_bank = F.normalize(torch.cat(feature_bank, dim=0), dim=1)

        ################################### sample relabelling ###################################
        prediction_cls = torch.softmax(torch.cat(prediction, dim=0), dim=1)
        his_score, his_label = prediction_cls.max(1)
        print(f'Prediction track: mean: {his_score.mean()} max: {his_score.max()} min: {his_score.min()}')
        conf_id = torch.where(his_score > args.theta_r)[0]
        modified_label = torch.clone(noisy_label).detach()
        modified_label[conf_id] = his_label[conf_id]
        
        ################################### sample selection ###################################
        prediction_knn = weighted_knn(feature_bank, feature_bank, modified_label, args.num_classes, args.k, 10)  # temperature in weighted KNN
        vote_y = torch.gather(prediction_knn, 1, modified_label.view(-1, 1)).squeeze()
        vote_max = prediction_knn.max(dim=1)[0]
        right_score = vote_y / vote_max
        clean_id = torch.where(right_score >= args.theta_s)[0]
        noisy_id = torch.where(right_score < args.theta_s)[0]

        ################################### SSR monitor ###################################
        TP = torch.sum(modified_label[clean_id] == clean_label[clean_id])
        FP = torch.sum(modified_label[clean_id] != clean_label[clean_id])
        TN = torch.sum(modified_label[noisy_id] != clean_label[noisy_id])
        FN = torch.sum(modified_label[noisy_id] == clean_label[noisy_id])
        print(f'Epoch [{i}/{args.epochs}] selection: theta_s:{args.theta_s} TP: {TP} FP:{FP} TN:{TN} FN:{FN}')
        logger_wandb.log({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})

        correct = torch.sum(modified_label[conf_id] == clean_label[conf_id])
        orginal = torch.sum(noisy_label[conf_id] == clean_label[conf_id])
        all = len(conf_id)
        logger_wandb.log({'correct': correct, 'original': orginal, 'total': all})
        print(f'Epoch [{i}/{args.epochs}] relabelling:  correct: {correct} original: {orginal} total: {all}')

        stat_logs.write(f'Epoch [{i}/{args.epochs}] selection: theta_s:{args.theta_s} TP: {TP} FP:{FP} TN:{TN} FN:{FN}\n')
        stat_logs.write(f'Epoch [{i}/{args.epochs}] relabelling:  correct: {correct} original: {orginal} total: {all}\n')
        stat_logs.flush()
    return clean_id, noisy_id, modified_label



def weighted_knn(cur_feature, feature, label, num_classes, knn_k=100, chunks=10, norm='global'):
    # distributed fast KNN and sample selection with three different modes
    num = len(cur_feature)
    num_class = torch.tensor([torch.sum(label == i).item() for i in range(num_classes)]).to(
        feature.device) + 1e-10
    pi = num_class / num_class.sum()
    split = torch.tensor(np.linspace(0, num, chunks + 1, dtype=int), dtype=torch.long).to(feature.device)
    score = torch.tensor([]).to(feature.device)
    pred = torch.tensor([], dtype=torch.long).to(feature.device)
    feature = torch.nn.functional.normalize(feature, dim=1)
    with torch.no_grad():
        for i in range(chunks):
            torch.cuda.empty_cache()
            part_feature = cur_feature[split[i]: split[i + 1]]

            part_score, part_pred = knn_predict(part_feature, feature.T, label, num_classes, knn_k)
            score = torch.cat([score, part_score], dim=0)
            pred = torch.cat([pred, part_pred], dim=0)

        # balanced vote
        if norm == 'global':
            # global normalization
            score = score / pi
        else:  # no normalization
            pass
        score = score/score.sum(1, keepdim=True)

    return score  # , preds