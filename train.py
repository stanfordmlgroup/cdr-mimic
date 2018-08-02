import torch
import torch.nn as nn

import models
import optim
import util
from args import TrainArgParser
from data import Dataset, get_loader
from evaluator import ModelEvaluator
from logger import TrainLogger
from saver import ModelSaver

def train(args):
    train_loader, D_in = get_loader(args=args)
    if args.ckpt_path:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
        args.start_epoch = ckpt_info['epoch'] + 1
    else:
        model_fn = models.__dict__[args.model]
        args.D_in = D_in
        model = model_fn(**vars(args))
        # model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.train()

    # Get optimizer and scheduler
    optimizer = optim.get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
    # lr_scheduler = optim.get_scheduler(optimizer, args)
    if args.ckpt_path:
        ModelSaver.load_optimizer(args.ckpt_path, optimizer)  # , lr_scheduler)

    # Get logger, evaluator, saver
    loss_fn = optim.get_loss_fn(args.loss_fn, args)

    logger = TrainLogger(args, len(train_loader.dataset))
    eval_loaders = [get_loader(args, phase='train', is_training=False),
                    get_loader(args, phase='valid', is_training=False)]
    evaluator = ModelEvaluator(eval_loaders, logger, args.max_eval, args.epochs_per_eval)
    # evaluator = ModelEvaluator(args.loss_fn, args, eval_loaders, logger,
    #                         args.num_visuals, args.max_eval, args.epochs_per_eval)

    saver = ModelSaver(**vars(args))

    # Train model
    while not logger.is_finished_training():
        logger.start_epoch()

        for src, tgt in train_loader:
            logger.start_iter()
            with torch.set_grad_enabled(True):
                pred_params = model.forward(src.to(args.device))
                # print("pred_params:", pred_params)
                loss = loss_fn(pred_params, tgt.to(args.device))
                # print("loss out", loss)

                logger.log_iter(src, pred_params, tgt, loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.end_iter()

        # metrics = evaluator.evaluate(model, args.device, logger.epoch)
        # saver.save(logger.epoch, model, optimizer, lr_scheduler, args.device,
        #            metric_val=metrics.get(args.metric_name, None))
        # logger.end_epoch(metrics=metrics)
        # print(metrics)
        # logger.end_epoch({})
        # optim.step_scheduler(lr_scheduler, metrics, logger.epoch)


if __name__ == '__main__':
    parser = TrainArgParser()
    args = parser.parse_args()
    train(args)
