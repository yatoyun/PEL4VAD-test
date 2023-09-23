
def build_config(dataset):
    cfg = type('', (), {})()
    if dataset in ['ucf', 'ucf-crime']:
        cfg.dataset = 'ucf-crime'
        cfg.model_name = 'ucf_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = './data/ucf-i3d'#'/data/pyj/feat/ucf-i3d'
        cfg.train_list = './list/ucf/train.list'
        cfg.test_list = './list/ucf/test.list'
        cfg.token_feat = './list/ucf/ucf-prompt.npy'
        cfg.gt =  './list/ucf/ucf-gt.npy'
        # TCA settings
        cfg.win_size = 9
        cfg.gamma = 0.6
        cfg.bias = 0.2
        cfg.norm = True
        # CC settings
        cfg.t_step = 9
        # training settings
        cfg.temp = 0.09
        cfg.lamda = 0.288#1.0
        cfg.seed = 42 #9
        # test settings
        cfg.test_bs = 10
        cfg.smooth = 'slide'  # ['fixed': 10, slide': 7]
        cfg.kappa = 10  # smooth window
        cfg.ckpt_path = './ckpt/ucf__current.pkl'#'./ckpt/ucf__8636.pkl'
        # cfg.ckpt_bert_path = './ckpt/bert_current.pkl'
        
        # ur dmu
        cfg.a_nums = 50
        cfg.n_nums = 50

        cfg.clip_feat_prefix = '/home/yukaneko/dev/CLIP-TSA_dataset/ucf/features/'

    elif dataset in ['xd', 'xd-violence']:
        cfg.dataset = 'xd-violence'
        cfg.model_name = 'xd_'
        cfg.metrics = 'AP'
        cfg.feat_prefix = './data/xd-i3d'
        cfg.train_list = './list/xd/train.list'
        cfg.test_list = './list/xd/test.list'
        cfg.token_feat = './list/xd/xd-prompt.npy'
        cfg.gt = './list/xd/xd-gt.npy'
        # TCA settings
        cfg.win_size = 9
        cfg.gamma = 0.06
        cfg.bias = 0.02
        cfg.norm = False
        # CC settings
        cfg.t_step = 3
        # training settings
        cfg.temp = 0.05
        cfg.lamda = 0.5
        cfg.seed = 4
        # test settings
        cfg.test_bs = 5
        cfg.smooth = 'fixed'  # ['fixed': 8, slide': 3]
        cfg.kappa = 8  # smooth window
        cfg.ckpt_path = './ckpt/xd__8584.pkl'
        cfg.clip_feat_prefix = '/home/yukaneko/dev/CLIP-TSA_dataset/xd/features/'

    elif dataset in ['sh', 'SHTech']:
        cfg.dataset = 'shanghaiTech'
        cfg.model_name = 'SH_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = '/data/pyj/feat/SHTech-i3d'
        cfg.train_list = './list/sh/train.list'
        cfg.test_list = './list/sh/test.list'
        cfg.token_feat = './list/sh/sh-prompt.npy'
        cfg.abn_label = './list/sh/relabel.list'
        cfg.gt = './list/sh/sh-gt.npy'
        # TCA settings
        cfg.win_size = 5
        cfg.gamma = 0.08
        cfg.bias = 0.1
        cfg.norm = True
        # CC settings
        cfg.t_step = 3
        # training settings
        cfg.temp = 0.2
        cfg.lamda = 9
        cfg.seed = 0
        # test settings
        cfg.test_bs = 10
        cfg.smooth = 'slide'  # ['fixed': 5, slide': 3]
        cfg.kappa = 3  # smooth window
        cfg.ckpt_path = './ckpt/SH__98.pkl'

    # base settings
    cfg.feat_dim = 1024
    cfg.head_num = 1
    cfg.hid_dim = 128
    cfg.out_dim = 300
    cfg.lr = 1e-4
    cfg.dropout = 0.5
    cfg.train_bs = 64
    cfg.max_seqlen = 200
    cfg.max_epoch = 100
    cfg.workers = 8
    cfg.save_dir = './ckpt/'
    cfg.logs_dir = './log_info.log'

    return cfg
