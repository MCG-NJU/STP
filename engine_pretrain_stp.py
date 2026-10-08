"""
Engine for STP pre-training: one loss, two decoders.
"""
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for step, (current_frame, future_frame, _) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        if step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, step / len(data_loader) + epoch, args)

        current_frame = current_frame.to(device, non_blocking=True)
        future_frame = future_frame.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _, _ = model(current_frame, future_frame)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            # Do not abort the run, but make it loud -- a silently-swallowed
            # non-finite loss here means the run is wasted.
            print('Loss is {}, skipping step'.format(loss_value), file=sys.stderr)
            optimizer.zero_grad()
            continue

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(step + 1) % accum_iter == 0)
        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (step + 1) % accum_iter == 0:
            # epoch_1000x keeps the x-axis comparable across batch sizes
            epoch_1000x = int((step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch_1000x)

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
