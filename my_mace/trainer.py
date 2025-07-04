###########################################################################################
# Modified version of below code | Modified by Rushikesh

# Training script
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)

# Modifications, Simplified as per my needs and annoted with comments
###########################################################################################

import dataclasses
import logging
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from torchmetrics import Metric

from mace.tools import torch_geometric
from mace.tools.checkpoint import CheckpointHandler, CheckpointState
from mace.tools.torch_tools import to_numpy
from mace.tools.utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
)


@dataclasses.dataclass
class SWAContainer:
    model: AveragedModel
    scheduler: SWALR
    start: int
    loss_fn: torch.nn.Module


def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    log_errors,
    epoch=None,
    valid_loader_name="Default",
):
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch
    logger.log(eval_metrics)

    if epoch is None:
        inintial_phrase = "Initial"
    else:
        inintial_phrase = f"Epoch {epoch}"
        
    if log_errors == "PerAtomRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A"
        )
    
    elif log_errors == "TotalRMSE":
        error_e = eval_metrics["rmse_e"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "PerAtomMAE":
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "TotalMAE":
        error_e = eval_metrics["mae_e"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        )
    


def train(
            model: torch.nn.Module,
            loss_fn: torch.nn.Module,
            train_loader: DataLoader,
            valid_loaders: Dict[str, DataLoader],
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
            start_epoch: int,
            max_num_epochs: int,
            patience: int,
            checkpoint_handler: CheckpointHandler,
            logger: MetricsLogger,
            eval_interval: int,
            output_args: Dict[str, bool],
            device: torch.device,
            log_errors: str,
            swa: Optional[SWAContainer] = None,
            ema: Optional[ExponentialMovingAverage] = None,
            max_grad_norm: Optional[float] = 10.0,
        ):
    # some values to tracks
    lowest_loss = np.inf
    valid_loss = np.inf
    patience_counter = 0
    swa_start = True
    keep_last = False

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")

    logging.info("")
    logging.info("===========TRAINING===========")
    logging.info("Started training, reporting errors on validation set")
    logging.info("Loss metrics on validation set")
    epoch = start_epoch

    # log validation loss before _any_ training
    valid_loss = 0.0
    for valid_loader_name, valid_loader in valid_loaders.items():
        # evaluate val loss and metrics for each val_dataloader and log them
        valid_loss_head, eval_metrics = evaluate(
                                                    model=model,
                                                    loss_fn=loss_fn,
                                                    data_loader=valid_loader,
                                                    output_args=output_args,
                                                    device=device,
                                                )
        valid_err_log(
                        valid_loss_head, eval_metrics, logger, log_errors, None, valid_loader_name
                    )
    valid_loss = valid_loss_head  # consider only the last head for the checkpoint

    # till epoch is less than max_num_epochs
    while epoch < max_num_epochs:
        # LR scheduler and SWA update
        if swa is None or epoch < swa.start:
            if epoch > start_epoch:
                lr_scheduler.step(
                    metrics=valid_loss
                )  # Can break if exponential LR, TODO fix that!
        else:
            # swa_starts, so change loss and load model
            if swa_start:
                logging.info("Changing loss based on Stage Two Weights")
                lowest_loss = np.inf
                swa_start = False
                keep_last = True
            loss_fn = swa.loss_fn
            swa.model.update_parameters(model)
            if epoch > start_epoch:
                swa.scheduler.step()

        # Train
        if "ScheduleFree" in type(optimizer).__name__:
            optimizer.train()
        
        # trains one epoch
        train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            ema=ema,
            logger=logger,
            device=device,
        )

        # Validate
        if epoch % eval_interval == 0:
            model_to_evaluate = model
            param_context = (
                ema.average_parameters() if ema is not None else nullcontext()
            )
            if "ScheduleFree" in type(optimizer).__name__:
                optimizer.eval()
            # evaluate model on val loader, for each val loader
            with param_context:
                valid_loss = 0.0
                for valid_loader_name, valid_loader in valid_loaders.items():
                    valid_loss_head, eval_metrics = evaluate(
                                                                model=model_to_evaluate,
                                                                loss_fn=loss_fn,
                                                                data_loader=valid_loader,
                                                                output_args=output_args,
                                                                device=device,
                                                            )
                    # log the val metrics
                    valid_err_log(
                            valid_loss_head,
                            eval_metrics,
                            logger,
                            log_errors,
                            epoch,
                            valid_loader_name,
                        )
                valid_loss = (
                    valid_loss_head  # consider only the last head for the checkpoint
                )

            # if val loss increase
            if valid_loss >= lowest_loss:
                # increase patience counter
                patience_counter += 1
                # terminate if patience counter exceeds patience
                if patience_counter >= patience:
                    if swa is not None and epoch < swa.start:
                        logging.info(
                            f"Stopping optimization after {patience_counter} epochs without improvement and starting Stage Two"
                        )
                        epoch = swa.start
                    else:
                        logging.info(
                            f"Stopping optimization after {patience_counter} epochs without improvement"
                        )
                        break


            # val loss decreased, reset patience counter and save model
            else:
                lowest_loss = valid_loss
                patience_counter = 0
                param_context = (
                    ema.average_parameters() if ema is not None else nullcontext()
                )
                with param_context:
                    checkpoint_handler.save(
                        state=CheckpointState(model, optimizer, lr_scheduler),
                        epochs=epoch,
                        keep_last=keep_last,
                    )
                    keep_last = False

        epoch += 1

    logging.info("Training complete")


def train_one_epoch(
                    model: torch.nn.Module,
                    loss_fn: torch.nn.Module,
                    data_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    output_args: Dict[str, bool],
                    max_grad_norm: Optional[float],
                    ema: Optional[ExponentialMovingAverage],
                    logger: MetricsLogger,
                    device: torch.device,
                ) -> None:
    
    # iterate thorugh batches and take step
    for batch in data_loader:
        _, opt_metrics = take_step(
                                    model=model,
                                    loss_fn=loss_fn,
                                    batch=batch,
                                    optimizer=optimizer,
                                    ema=ema,
                                    output_args=output_args,
                                    max_grad_norm=max_grad_norm,
                                    device=device,
                                )
        opt_metrics["mode"] = "opt"
        opt_metrics["epoch"] = epoch
        logger.log(opt_metrics)


def take_step(  model: torch.nn.Module,
                loss_fn: torch.nn.Module,
                batch: torch_geometric.batch.Batch,
                optimizer: torch.optim.Optimizer,
                ema: Optional[ExponentialMovingAverage],
                output_args: Dict[str, bool],
                max_grad_norm: Optional[float],
                device: torch.device,
            ) -> Tuple[float, Dict[str, Any]]:
    # measure time
    start_time = time.time()

    # send batch to device
    batch = batch.to(device)
    optimizer.zero_grad(set_to_none=True)
    batch_dict = batch.to_dict()

    output = model(
                    batch_dict,
                    training=True,
                    compute_force=output_args["forces"],
                    compute_virials=output_args["virials"],
                    compute_stress=output_args["stress"],
                )
    loss = loss_fn(pred=output, ref=batch)
    loss.backward()

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


def evaluate(
            model: torch.nn.Module,
            loss_fn: torch.nn.Module,
            data_loader: DataLoader,
            output_args: Dict[str, bool],
            device: torch.device,
        ) -> Tuple[float, Dict[str, Any]]:
    
    # freeze the model
    for param in model.parameters():
        param.requires_grad = False
    
    # create metrics
    metrics = MACELoss(loss_fn=loss_fn).to(device)

    # start the timer
    start_time = time.time()

    for batch in data_loader:
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        output = model(
                    batch_dict,
                    training=False,
                    compute_force=output_args["forces"],
                    compute_virials=output_args["virials"],
                    compute_stress=output_args["stress"],
                )
        avg_loss, aux = metrics(batch, output)

    avg_loss, aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

    # unfreeze the model
    for param in model.parameters():
        param.requires_grad = True

    return avg_loss, aux


class MACELoss(Metric):
    def __init__(self, loss_fn: torch.nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_data", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Fs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_fs", default=[], dist_reduce_fx="cat")

    def update(self, batch, output):  # pylint: disable=arguments-differ
        loss = self.loss_fn(pred=output, ref=batch)
        self.total_loss += loss
        self.num_data += batch.num_graphs

        if output.get("energy") is not None and batch.energy is not None:
            self.E_computed += 1.0
            self.delta_es.append(batch.energy - output["energy"])
            self.delta_es_per_atom.append(
                (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
            )
        if output.get("forces") is not None and batch.forces is not None:
            self.Fs_computed += 1.0
            self.fs.append(batch.forces)
            self.delta_fs.append(batch.forces - output["forces"])
        

    def convert(self, delta: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return to_numpy(delta)

    def compute(self):
        aux = {}
        aux["loss"] = to_numpy(self.total_loss / self.num_data).item()
        if self.E_computed:
            delta_es = self.convert(self.delta_es)
            delta_es_per_atom = self.convert(self.delta_es_per_atom)
            aux["mae_e"] = compute_mae(delta_es)
            aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
            aux["rmse_e"] = compute_rmse(delta_es)
            aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
            aux["q95_e"] = compute_q95(delta_es)
        if self.Fs_computed:
            fs = self.convert(self.fs)
            delta_fs = self.convert(self.delta_fs)
            aux["mae_f"] = compute_mae(delta_fs)
            aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
            aux["rmse_f"] = compute_rmse(delta_fs)
            aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
            aux["q95_f"] = compute_q95(delta_fs)
        
        return aux["loss"], aux
