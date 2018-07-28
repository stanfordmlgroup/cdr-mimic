import torch
import torch.nn as nn

import models
import optim
import util
from args import TrainArgParser
from data.loader import load_data, Dataset
from evaluator import ModelEvaluator
from logger import TrainLogger
from saver import ModelSaver


# def loss(self, pred, tgt, is_alive):
def loss_fn(pred_params, tgts):
    loss_val = 0
    for pred_param, tgt in zip(pred_params, tgts):
        mu, s = pred_param[0], pred_param[1]
        pred = torch.distributions.LogNormal(mu, s.exp())
        tte, is_alive = tgt[0], tgt[1]
        print("tte", tte)
        print("log prob", pred.log_prob(tte + 1e-5))
        print("cdf", pred.cdf(tte))
        loss_val += - ((1 - is_alive) * pred.log_prob(tte + 1e-5) + (1 - pred.cdf(tte) + 1e-5).log() * is_alive)
    print("loss", loss_val)

    return loss_val / tgts.shape[0]

def train(args):
    train_loader = load_data(args=args)
    if args.ckpt_path:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
        args.start_epoch = ckpt_info['epoch'] + 1
    else:
        model_fn = models.__dict__[args.model]
        model = model_fn(**vars(args))
        model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device) # erik was here
    model.train()

    # Get optimizer and scheduler
    optimizer = optim.get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
    # lr_scheduler = optim.get_scheduler(optimizer, args)
    if args.ckpt_path:
        ModelSaver.load_optimizer(args.ckpt_path, optimizer)  # , lr_scheduler)

    # Get logger, evaluator, saver
    # loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    # loss_fn = nn.KLDivLoss()
    # loss_fn = nn.MSELoss(size_average=False)

    logger = TrainLogger(args, len(train_loader.dataset))
    valid_loader = torch.utils.data.DataLoader(Dataset(args, 'valid', is_training_set=False),
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               shuffle=False)
    valid_loader.phase = 'valid'
    eval_loaders = [valid_loader]
    # data.DataLoader(MIMICDataset(args, 'dev', args.data_dir, is_training_set=False),
    #                 args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)]
    evaluator = ModelEvaluator(eval_loaders, logger, args.max_eval, args.epochs_per_eval)
    saver = ModelSaver(**vars(args))

    # Train model
    while not logger.is_finished_training():
        logger.start_epoch()
        # todo: tte, is_alive = tgt[0], tgt[1]... tgt needs to have time to event, not dod, and second column in csv
        # with is_alive bool. if alive, the date in the first col is not time to death event, but the time
        # (from discharge or second day of admission) to last event recorded in db
        for src, tgt in train_loader:
            logger.start_iter()
            with torch.set_grad_enabled(True):
                pred_params = model.forward(src.to(args.device))
                print("pred_params:", pred_params)
                # print("training", logits.int())
                # loss = loss_fn(logits, tgt.to(args.device))
                loss = loss_fn(pred_params, tgt.to(args.device))

                logger.log_iter(loss)

                loss.backward()
                optimizer.zero_grad()
                optimizer.step()
            logger.end_iter()

        metrics = evaluator.evaluate(model, args.device, logger.epoch)
        # saver.save(logger.epoch, model, optimizer, lr_scheduler, args.device,
        #            metric_val=metrics.get(args.metric_name, None))
        logger.end_epoch(metrics=metrics)
        # print(metrics)
        # logger.end_epoch({})
        # optim.step_scheduler(lr_scheduler, metrics, logger.epoch)


if __name__ == '__main__':
    util.set_spawn_enabled()
    parser = TrainArgParser()
    train(parser.parse_args())
