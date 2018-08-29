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
    train_loader = get_loader(args=args)
    if args.ckpt_path:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
        args.start_epoch = ckpt_info['epoch'] + 1
    else:
        model_fn = models.__dict__[args.model]
        args.D_in = train_loader.D_in
        model = model_fn(**vars(args))
        # model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.train()

    # Get optimizer and scheduler
    optimizer = optim.get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
    lr_scheduler = optim.get_scheduler(optimizer, args)
    if args.ckpt_path:
        ModelSaver.load_optimizer(args.ckpt_path, optimizer, lr_scheduler)

    # Get logger, evaluator, saver
    loss_fn = optim.get_loss_fn(args.loss_fn, args)

    # Embedding loss and optimizer
    # optimizer_embed = optim.SGD(model.parameters(), lr=0.001)
    loss_fn_embed = nn.NLLLoss()

    logger = TrainLogger(args, len(train_loader.dataset))
    eval_loaders = [get_loader(args, phase='train', is_training=False),
                    get_loader(args, phase='valid', is_training=False)]
    evaluator = ModelEvaluator(args, eval_loaders, logger, args.max_eval, args.epochs_per_eval)

    saver = ModelSaver(**vars(args))

    # Train model
    while not logger.is_finished_training():
        logger.start_epoch()

        for src, tgt in train_loader:
            logger.start_iter()
            with torch.set_grad_enabled(True):
                pred_params, embed_logits = model.forward(src.to(args.device))
                loss = loss_fn(pred_params, tgt.to(args.device))
                embed_loss = loss_fn_embed(embed_logits) # tgt?
                loss += embed_loss
                
                logger.log_iter(src, pred_params, tgt, loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.end_iter()

        metrics = evaluator.evaluate(model, args.device, logger.epoch)
        # print(metrics)
        saver.save(logger.epoch, model, optimizer, lr_scheduler, args.device,
                   metric_val=metrics.get(args.metric_name, None))
        logger.end_epoch(metrics=metrics)
        # logger.end_epoch({})
        # optim.step_scheduler(lr_scheduler, metrics, logger.epoch)


if __name__ == '__main__':
    parser = TrainArgParser()
    args = parser.parse_args()
    train(args)
