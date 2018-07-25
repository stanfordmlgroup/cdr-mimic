import torch
import torch.nn as nn

import models
import optim
import util
from args import TrainArgParser
from data.loader import load_data
from logger import TrainLogger
from saver import ModelSaver


def train(args):
    train_loader = load_data(args=args)
    if args.ckpt_path:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
        args.start_epoch = ckpt_info['epoch'] + 1
    else:
        model_fn = models.__dict__[args.model]
        model = model_fn(**vars(args))
        model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.train()
    #
    # # Get optimizer and scheduler
    optimizer = optim.get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
    # lr_scheduler = optim.get_scheduler(optimizer, args)
    if args.ckpt_path:
        ModelSaver.load_optimizer(args.ckpt_path, optimizer)  # , lr_scheduler)

    # # Get logger, evaluator, saver
    # loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    loss_fn = nn.KLDivLoss()
    # loss_fn = nn.MSELoss(size_average=False)

    logger = TrainLogger(args, len(train_loader.dataset))
    # eval_loaders = [data.DataLoader(SQuID(args, 'train', args.data_dir, is_training_set=False),
    #                                 args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
    #                 data.DataLoader(SQuID(args, 'dev', args.data_dir, is_training_set=False),
    #                                 args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)]
    # evaluator = ModelEvaluator(eval_loaders, logger, args.max_eval, args.epochs_per_eval)
    saver = ModelSaver(**vars(args))

    # Train model
    while not logger.is_finished_training():
        logger.start_epoch()

        for src, tgt in train_loader:
            logger.start_iter()
            with torch.set_grad_enabled(True):
                logits = model.forward(src.to(args.device))
                # print(logits)
                loss = loss_fn(logits, tgt.to(args.device))

                logger.log_iter(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            logger.end_iter()

        # metrics, curves = evaluator.evaluate(model, args.device, logger.epoch)
        # saver.save(logger.epoch, model, optimizer, lr_scheduler, args.device,
        #            metric_val=metrics.get(args.metric_name, None))
        # logger.end_epoch(metrics=metrics)
        logger.end_epoch({})
        # optim.step_scheduler(lr_scheduler, metrics, logger.epoch)


if __name__ == '__main__':
    util.set_spawn_enabled()
    parser = TrainArgParser()
    train(parser.parse_args())
