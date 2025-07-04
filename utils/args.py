
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="Device: cuda:num or cpu")
    parser.add_argument("--path", type=str, default='./dataset/', help="Path of datasets")
    parser.add_argument("--DS", type=str, default="BDGP", help="Name of datasets")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="xx")
    parser.add_argument('--cluster_emb', dest='cluster_emb', type=int, default=10, help='cluster layer embedding dimension')
    parser.add_argument('--pre_dim', dest='pre_dim', type=int, default=10, help='first projectiong layer embedding dimension for multi-view data')
    parser.add_argument('--expert_num', dest='expert_num', type=int, default=10, help='the number of expert networks')
    parser.add_argument('--k', dest='k', type=int, default=3, help='top k experts')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=100, help='batch size')
    parser.add_argument("--n_repeated", type=int, default=5, help="Number of repeated times. Default is 10.")
    parser.add_argument('--d', dest='d', type=int, default=5, help='')
    parser.add_argument('--eta', dest='eta', type=int, default=2, help='')
    parser.add_argument("--save_results", action='store_true', default=True, help="xx")
    parser.add_argument("--save_all", action='store_true', default=True, help="xx")
    # parser.add_argument("--save_loss", action='store_true', default= True, help="xx")
    parser.add_argument("--eval", action='store_true', default=True, help="evaluation via saved models")

    # parser.add_argument("--knns", type=int, default=15, help="Number of k nearest neighbors")
    # parser.add_argument("--common_neighbors", type=int, default=2, help="Number of common neighbors (when using pruning strategy 2)")
    # parser.add_argument("--pr1", action='store_true', default=True, help="Using prunning strategy 1 or not")
    # parser.add_argument("--pr2", action='store_true', default=True, help="Using prunning strategy 2 or not")
    # parser.add_argument("--ghost", action='store_true', default=False, help="xx")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--ratio", type=float, default=0.1, help="Ratio of labeled samples")
    parser.add_argument("--num_epoch", type=int, default=700, help="Number of training epochs. Default is 200.")

    parser.add_argument("--dim1", type=int, default=8, help="Number of hidden dimensions")
    parser.add_argument("--dim2", type=int, default=32, help="Number of hidden dimensions")

    parser.add_argument("--epochs", type=float, default=2, help="Training Epochs")
    parser.add_argument("--w", type=float, default=2, help="Initilize of hidden w")
    parser.add_argument("--lamda", type=float, default=0.01, help="Initilize coefficent of distribution loss")
    parser.add_argument("--gamma", type=float, default=0.01, help="Initilize coefficient of clustering loss")



    args = parser.parse_args()

    return args