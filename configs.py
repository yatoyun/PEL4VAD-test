
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
        # cfg.lamda = #1.0
        cfg.seed = 2023 #9
        # test settings
        cfg.test_bs = 10
        cfg.smooth = 'slide'  # ['fixed': 10, slide': 7]
        cfg.kappa = 8  # smooth window
        cfg.ckpt_path = './ckpt/ucf__current.pkl'#'./ckpt/ucf__8968.pkl'
        # cfg.ckpt_bert_path = './ckpt/bert_current.pkl'
        
        # ur dmu
        cfg.a_nums = 50
        cfg.n_nums = 50
        
        # contrasive
        cfg.k = 20
        
        cfg.lamda = 1.1
        cfg.alpha = 0.5
        
        # margin
        cfg.margin = 100
        cfg.max_epoch = 6

        cfg.clip_feat_prefix = '/home/yukaneko/dev/CLIP-TSA_dataset/ucf/features/'
        
        cfg.result_dir = './result/ucf/'

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
        # cfg.lamda = 0.5
        cfg.seed = 2 # 42
        # test settings
        cfg.test_bs = 5
        cfg.smooth = 'slide'  # ['fixed': 8, slide': 3]
        cfg.kappa = 2  # smooth window
        cfg.ckpt_path = './ckpt/xd__current.pkl'
        
        # ur dmu
        cfg.a_nums = 50
        cfg.n_nums = 50
        
        # contrasive
        cfg.k = 20
        
        cfg.lamda = 0.6
        cfg.alpha = 0.6
        
        # margin
        cfg.margin = 100
        cfg.max_epoch = 20
        
        cfg.clip_feat_prefix = '/home/yukaneko/dev/CLIP-TSA_dataset/xd/features/'
        
        cfg.result_dir = './result/xd/'

    elif dataset in ['sh', 'SHTech']:
        cfg.dataset = 'shanghaiTech'
        cfg.model_name = 'SH_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = './data/sh-i3d'
        cfg.train_list = './list/sh/train.list'
        cfg.test_list = './list/sh/test.list'
        cfg.token_feat = './list/sh/sh-prompt.npy'
        cfg.abn_label = './list/sh/relabel.list'
        cfg.gt = '/home/yukaneko/dev/AbnormalDetection/RTFM/list/gt-sh.npy' #./list/sh/sh-gt.npy'
        # TCA settings
        cfg.win_size = 5
        cfg.gamma = 0.08
        cfg.bias = 0.1
        cfg.norm = True
        # CC settings
        cfg.t_step = 3
        # training settings
        cfg.temp = 0.2
        # cfg.lamda = 9
        cfg.seed = 0
        # test settings
        cfg.test_bs = 10
        cfg.smooth = 'slide'  # ['fixed': 5, slide': 3]
        cfg.kappa = 3 # smooth window
        cfg.ckpt_path = './ckpt/SH__current.pkl'
        
        # ur dmu
        cfg.a_nums = 50
        cfg.n_nums = 50
        
        cfg.lamda = 1.2
        cfg.alpha = 0.4
        
        # contrasive
        cfg.k = 20
        
        # margin
        cfg.margin = 210
        cfg.max_epoch = 250
        
        cfg.clip_feat_prefix = '/home/yukaneko/dev/CLIP-TSA_dataset/sh/features/'
        
        cfg.result_dir = './result/sh/'

    # base settings
    cfg.feat_dim = 1024
    cfg.head_num = 1
    cfg.hid_dim = 128
    cfg.out_dim = 300
    cfg.lr = 1e-4
    cfg.dropout = 0.5
    cfg.train_bs = 32
    cfg.max_seqlen = 200
    
    cfg.workers = 8
    cfg.save_dir = './ckpt/'
    cfg.logs_dir = './log_info.log'

    return cfg
