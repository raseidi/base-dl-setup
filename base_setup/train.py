import wandb
import engine
import models
import utils

from data_setup import get_test_data

from torch import nn
from torch.optim import Adam
from torch.nn import functional as F


def main(args):
    model = models.NeuralNet()
    model = model.cuda()
    optim = Adam(model.parameters(), lr=args.lr)  # ToDo via args
    data_loader = get_test_data(args.batch_size)
    logger = wandb.init(project="tesing", config=args, name=args.experiment_name)
    logger.watch(
        models=(model), criterion=F.cross_entropy, log="all", log_freq=5, log_graph=True
    )
    results = engine.train(
        model,
        data_loader,
        data_loader,
        F.cross_entropy,
        optim,  # remove and use run.config # ToDo
        logger,
    )
    utils.save_model(
        model=model,
        target_dir="models",
        model_name="05_going_modular_script_mode_tinyvgg_model.pth",
    )
    logger.finish()


if __name__ == "__main__":
    args = utils.get_args_parser().parse_args()
    main(args)
