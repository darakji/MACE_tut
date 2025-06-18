###########################################################################################
# Modified version of below code | Modified by Rushikesh

# Training script for MACE
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
import ast
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch.distributed
import torch.nn.functional
from torch.utils.data import ConcatDataset
from torch_ema import ExponentialMovingAverage

import mace
from mace import data, tools
from mace.tools import torch_geometric
from .models import configure_model
from .trainer import train
from mace.tools.multihead_tools import (
    HeadConfig,
    dict_head_to_dataclass,
    prepare_default_head,
)
from mace.tools.scripts_utils import (
    LRScheduler,
    check_path_ase_read,
    dict_to_array,
    get_atomic_energies,
    get_avg_num_neighbors,
    get_config_type_weights,
    get_dataset_from_xyz,
    get_loss_fn,
    get_optimizer,
    get_params_options,
    get_swa,
    remove_pt_head,
)
from mace.tools.tables_utils import create_error_table
from mace.tools.utils import AtomicNumberTable


def main() -> None:
    """
    This script runs the training/fine tuning for mace
    """
    args = tools.build_default_arg_parser().parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    run(args)


def run(args: argparse.Namespace) -> None:
    """
    This script runs the training/fine tuning for mace
    """
    # tag is like model name
    tag = tools.get_tag(name=args.name, seed=args.seed)

    # check all args
    args, input_log_messages = tools.check_args(args)

    # Setup
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir)
    logging.info("===========VERIFYING SETTINGS===========")
    for message, loglevel in input_log_messages:
        logging.log(level=loglevel, msg=message)

    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.debug(f"Configuration: {args}")

    # setup dtype, init device
    tools.set_default_dtype(args.default_dtype)
    device = tools.init_device(args.device)
    
    # see if pretrained model is given and can we use multihead finetuning
    model_foundation: Optional[torch.nn.Module] = None
    if args.foundation_model is not None:
        assert os.path.exists(args.foundation_model), f"Couldn't find the model at path {args.foundation_model}"

        # load the model
        model_foundation = torch.load(args.foundation_model, map_location=args.device)
        logging.info(f"Using foundation model {args.foundation_model} as initial checkpoint.")
        args.r_max = model_foundation.r_max.item()
        # if pretraining file in not provided, can't do multihead finetuning
        if args.pt_train_file is None:
            logging.warning("Using multiheads finetuning with a foundation model that is not a Materials Project model, need to provied a path to a pretraining file with --pt_train_file.")
            args.multiheads_finetuning = False
        
        # if multihead finetuning is selected
        if args.multiheads_finetuning:
            assert args.E0s != "average", "average atomic energies cannot be used for multiheads finetuning"
            # check that the foundation model has a single head, if not, use the first head
            if hasattr(model_foundation, "heads"):
                if len(model_foundation.heads) > 1:
                    logging.warning("Mutlihead finetuning with models with more than one head is not supported, using the first head as foundation head.")
                    model_foundation = remove_pt_head(model_foundation, args.foundation_head)
    else:
        args.multiheads_finetuning = False

    # if head is provided, use, else prepare
    # head is dict(str:things), eg train_file, valid_file, E0 etc
    if args.heads is not None:
        args.heads = ast.literal_eval(args.heads)
    else:
        args.heads = prepare_default_head(args)

    # load input data for each head one by one
    logging.info("===========LOADING INPUT DATA===========")
    heads = list(args.heads.keys())
    logging.info(f"Using heads: {heads}")
    head_configs: List[HeadConfig] = []

    for head, head_args in args.heads.items():
        logging.info(f"=============    Processing head {head}     ===========")
        head_config = dict_head_to_dataclass(head_args, head, args)
        # if statistics file is given, use
        if head_config.statistics_file is not None:
            with open(head_config.statistics_file, "r") as f:  # pylint: disable=W1514
                statistics = json.load(f)
            logging.info("Using statistics json file")
            head_config.r_max = (
                statistics["r_max"] if args.foundation_model is None else args.r_max
            )
            head_config.atomic_numbers = statistics["atomic_numbers"]
            head_config.mean = statistics["mean"]
            head_config.std = statistics["std"]
            head_config.avg_num_neighbors = statistics["avg_num_neighbors"]
            head_config.compute_avg_num_neighbors = False
            if isinstance(statistics["atomic_energies"], str) and statistics["atomic_energies"].endswith(".json"):
                with open(statistics["atomic_energies"], "r", encoding="utf-8") as f:
                    atomic_energies = json.load(f)
                head_config.E0s = atomic_energies
                head_config.atomic_energies_dict = ast.literal_eval(atomic_energies)
            else:
                head_config.E0s = statistics["atomic_energies"]
                head_config.atomic_energies_dict = ast.literal_eval(
                    statistics["atomic_energies"]
                )

        # Data preparation, if train_file is .d5 or .hdf5 or dir that contains it, or empty dir => False
        if check_path_ase_read(head_config.train_file):
            if head_config.valid_file is not None:
                assert check_path_ase_read(head_config.valid_file), "valid_file if given must be same format as train_file"

            config_type_weights = get_config_type_weights(head_config.config_type_weights)
            collections, atomic_energies_dict = get_dataset_from_xyz(
                work_dir=args.work_dir,
                train_path=head_config.train_file,
                valid_path=head_config.valid_file,
                valid_fraction=head_config.valid_fraction,
                config_type_weights=config_type_weights,
                test_path=head_config.test_file,
                seed=args.seed,
                energy_key=head_config.energy_key,
                forces_key=head_config.forces_key,
                stress_key=head_config.stress_key,
                virials_key=head_config.virials_key,
                dipole_key=head_config.dipole_key,
                charges_key=head_config.charges_key,
                head_name=head_config.head_name,
                keep_isolated_atoms=head_config.keep_isolated_atoms,
            )
            head_config.collections = collections
            head_config.atomic_energies_dict = atomic_energies_dict
            logging.info(f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, "
                        f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}],")
        head_configs.append(head_config)

    # check if enough number of samples are there
    if all(check_path_ase_read(head_config.train_file) for head_config in head_configs):
        size_collections_train = sum(len(head_config.collections.train) for head_config in head_configs)
        size_collections_valid = sum(len(head_config.collections.valid) for head_config in head_configs)
        if size_collections_train < args.batch_size:
            logging.error(f"Batch size ({args.batch_size}) is larger than the number of training data ({size_collections_train})")
        if size_collections_valid < args.valid_batch_size:
            logging.warning(f"Validation batch size ({args.valid_batch_size}) is larger than the number of validation data ({size_collections_valid})")

    # if we are usng multihead finetuning, load pretrain data as well
    if args.multiheads_finetuning:
        logging.info("==================Using multiheads finetuning mode==================")
        args.loss = "universal"
        
        logging.info(f"Using foundation model for multiheads finetuning with {args.pt_train_file}")
        # add pretraining head at start
        heads = list(dict.fromkeys(["pt_head"] + heads))
        # load data
        collections, atomic_energies_dict = get_dataset_from_xyz(
            work_dir=args.work_dir,
            train_path=args.pt_train_file,
            valid_path=args.pt_valid_file,
            valid_fraction=args.valid_fraction,
            config_type_weights=None,
            test_path=None,
            seed=args.seed,
            energy_key=args.energy_key,
            forces_key=args.forces_key,
            stress_key=args.stress_key,
            virials_key=args.virials_key,
            dipole_key=args.dipole_key,
            charges_key=args.charges_key,
            head_name="pt_head",
            keep_isolated_atoms=args.keep_isolated_atoms,
        )

        # create pretrain head
        head_config_pt = HeadConfig(
            head_name="pt_head",
            train_file=args.pt_train_file,
            valid_file=args.pt_valid_file,
            E0s="foundation",
            statistics_file=args.statistics_file,
            valid_fraction=args.valid_fraction,
            config_type_weights=None,
            energy_key=args.energy_key,
            forces_key=args.forces_key,
            stress_key=args.stress_key,
            virials_key=args.virials_key,
            dipole_key=args.dipole_key,
            charges_key=args.charges_key,
            keep_isolated_atoms=args.keep_isolated_atoms,
            collections=collections,
            avg_num_neighbors=model_foundation.interactions[0].avg_num_neighbors,
            compute_avg_num_neighbors=False,
        )
        head_config_pt.collections = collections
        head_configs.append(head_config_pt)
        logging.info(f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}")

    # Atomic number table
    # yapf: disable
    for head_config in head_configs:
        # if atomic numbers not given, extract from train and valid datasets
        if head_config.atomic_numbers is None:            
            z_table_head = tools.get_atomic_number_table_from_zs(
                z
                for configs in (head_config.collections.train, head_config.collections.valid)
                for config in configs
                for z in config.atomic_numbers
            )
            head_config.atomic_numbers = z_table_head.zs
            head_config.z_table = z_table_head
        else:
            # if given, but not in stat file, read from command line
            if head_config.statistics_file is None:
                logging.info("Using atomic numbers from command line argument")
            else:
                logging.info("Using atomic numbers from statistics file")
            zs_list = ast.literal_eval(head_config.atomic_numbers)
            assert isinstance(zs_list, list)
            z_table_head = tools.AtomicNumberTable(zs_list)
            head_config.atomic_numbers = zs_list
            head_config.z_table = z_table_head
        # yapf: enable

    #  pool all atomic numbers from all heads
    all_atomic_numbers = set()
    for head_config in head_configs:
        all_atomic_numbers.update(head_config.atomic_numbers)
    z_table = AtomicNumberTable(sorted(list(all_atomic_numbers)))
    logging.info(f"Atomic Numbers used: {z_table.zs}")

    # Atomic energies
    atomic_energies_dict = {}
    for head_config in head_configs:
        # if no atomic energy dict is given
        if head_config.atomic_energies_dict is None or len(head_config.atomic_energies_dict) == 0:
            # E0 can't be none then
            assert head_config.E0s is not None, "Atomic energies must be provided"
            # if not foundation, calculate if train file given
            if check_path_ase_read(head_config.train_file) and head_config.E0s.lower() != "foundation":
                atomic_energies_dict[head_config.head_name] = get_atomic_energies(head_config.E0s,
                                                                                head_config.collections.train, 
                                                                                head_config.z_table)
            # if E0 to be used from foundation
            elif head_config.E0s.lower() == "foundation":
                assert args.foundation_model is not None
                z_table_foundation = AtomicNumberTable([int(z) for z in model_foundation.atomic_numbers])
                foundation_atomic_energies = model_foundation.atomic_energies_fn.atomic_energies

                # if foundation model is multihead
                if foundation_atomic_energies.ndim > 1:
                    foundation_atomic_energies = foundation_atomic_energies.squeeze()
                    if foundation_atomic_energies.ndim == 2:
                        foundation_atomic_energies = foundation_atomic_energies[0]
                        logging.info("Foundation model has multiple heads, using the first head as foundation E0s.")
                atomic_energies_dict[head_config.head_name] = {z: foundation_atomic_energies[z_table_foundation.z_to_index(z)].item() for z in z_table.zs}
            else:
                # if train file not given, may have to read from json
                atomic_energies_dict[head_config.head_name] = get_atomic_energies(head_config.E0s, None, head_config.z_table)
        # use the given atomic energy dict
        else:
            atomic_energies_dict[head_config.head_name] = head_config.atomic_energies_dict

    # Atomic energies for multiheads finetuning, for pretrain head
    if args.multiheads_finetuning:
        assert model_foundation is not None, "Model foundation must be provided for multiheads finetuning"
        z_table_foundation = AtomicNumberTable([int(z) for z in model_foundation.atomic_numbers])
        foundation_atomic_energies = model_foundation.atomic_energies_fn.atomic_energies

        # if the foundation model is itself multihead take first head/which is usully it's pretrained head, as we add pt_head at start
        if foundation_atomic_energies.ndim > 1:
            foundation_atomic_energies = foundation_atomic_energies.squeeze()
            if foundation_atomic_energies.ndim == 2:
                foundation_atomic_energies = foundation_atomic_energies[0]
                logging.info("Foundation model has multiple heads, using the first head as foundation E0s.")
        atomic_energies_dict["pt_head"] = { z: foundation_atomic_energies[z_table_foundation.z_to_index(z)].item()for z in z_table.zs}

    # set the output args
    dipole_only = False
    args.compute_energy = True
    args.compute_dipole = False
    atomic_energies = dict_to_array(atomic_energies_dict, heads)

    # log the atomic energies
    for head_config in head_configs:
        try:
            logging.info(f"Atomic Energies used (z: eV) for head {head_config.head_name}: " + "{" + ", ".join([f"{z}: {atomic_energies_dict[head_config.head_name][z]}" for z in head_config.z_table.zs]) + "}")
        except KeyError as e:
            raise KeyError(f"Atomic number {e} not found in atomic_energies_dict for head {head_config.head_name}, add E0s for this atomic number") from e


    # get val and train sets
    valid_sets = {head: [] for head in heads}
    train_sets = {head: [] for head in heads}

    for head_config in head_configs:
        if check_path_ase_read(head_config.train_file):
            train_sets[head_config.head_name] = [data.AtomicData.from_config( config, z_table=z_table, cutoff=args.r_max, heads=heads)
                                                    for config in head_config.collections.train]
            valid_sets[head_config.head_name] = [data.AtomicData.from_config( config, z_table=z_table, cutoff=args.r_max, heads=heads)
                                                    for config in head_config.collections.valid]

        else:
            raise ValueError(f"Provide file that ends with .xyz instead of {head_config.train_file}")
        
        train_loader_head = torch_geometric.dataloader.DataLoader(
            dataset=train_sets[head_config.head_name],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )
        head_config.train_loader = train_loader_head
        
    # concatenate all the trainsets
    train_set = ConcatDataset([train_sets[head] for head in heads])
    
    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    # valid loaders will be different for each head
    valid_loaders = {heads[i]: None for i in range(len(heads))}
    if not isinstance(valid_sets, dict):
        valid_sets = {"Default": valid_sets}

    for head, valid_set in valid_sets.items():
        valid_loaders[head] = torch_geometric.dataloader.DataLoader(
            dataset=valid_set,
            batch_size=args.valid_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )

    # get loss function and avg num of neighbors
    loss_fn = get_loss_fn(args, dipole_only, args.compute_dipole)
    args.avg_num_neighbors = get_avg_num_neighbors(head_configs, args, train_loader, device)

    # Model
    model, output_args = configure_model(args, train_loader, atomic_energies, model_foundation, heads, z_table)
    model.to(device)

    logging.debug(model)
    logging.info(f"Total number of parameters: {tools.count_parameters(model)}")
    logging.info("")
    logging.info("===========OPTIMIZER INFORMATION===========")
    logging.info(f"Using {args.optimizer.upper()} as parameter optimizer")
    logging.info(f"Batch size: {args.batch_size}")
    if args.ema:
        logging.info(f"Using Exponential Moving Average with decay: {args.ema_decay}")
    logging.info(f"Number of gradient updates: {int(args.max_num_epochs*len(train_set)/args.batch_size)}")
    logging.info(f"Learning rate: {args.lr}, weight decay: {args.weight_decay}")
    logging.info(loss_fn)

    # Optimizer
    param_options = get_params_options(args, model)
    optimizer: torch.optim.Optimizer
    optimizer = get_optimizer(args, param_options)
    logger = tools.MetricsLogger(directory=args.results_dir, tag=tag + "_train")  # pylint: disable=E1123

    # scheduler and swa
    lr_scheduler = LRScheduler(optimizer, args)

    swa: Optional[tools.SWAContainer] = None
    swas = [False]
    if args.swa:
        swa, swas = get_swa(args, model, optimizer, swas, dipole_only)

    # checkpoint handler
    checkpoint_handler = tools.CheckpointHandler(
        directory=args.checkpoints_dir,
        tag=tag,
        keep=args.keep_checkpoints,
        swa_start=args.start_swa,
    )

    start_epoch = 0
    if args.restart_latest:
        try:
            opt_start_epoch = checkpoint_handler.load_latest(state=tools.CheckpointState(model, optimizer, lr_scheduler),
                                                             swa=True,
                                                             device=device,)
        except Exception:  # pylint: disable=W0703
            opt_start_epoch = checkpoint_handler.load_latest( state=tools.CheckpointState(model, optimizer, lr_scheduler),
                                                            swa=False,
                                                            device=device,)
        if opt_start_epoch is not None:
            start_epoch = opt_start_epoch

    # initialize ema
    ema: Optional[ExponentialMovingAverage] = None
    if args.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    else:
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loaders=valid_loaders,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=checkpoint_handler,
        eval_interval=args.eval_interval,
        start_epoch=start_epoch,
        max_num_epochs=args.max_num_epochs,
        logger=logger,
        patience=args.patience,
        output_args=output_args,
        device=device,
        swa=swa,
        ema=ema,
        max_grad_norm=args.clip_grad,
        log_errors=args.error_table,
    )

    logging.info("")
    logging.info("===========RESULTS===========")
    logging.info("Computing metrics for training, validation, and test sets")

    # get train and all valid loaders for all heads
    train_valid_data_loader = {}
    for head_config in head_configs:
        data_loader_name = "train_" + head_config.head_name
        train_valid_data_loader[data_loader_name] = head_config.train_loader
    for head, valid_loader in valid_loaders.items():
        data_load_name = "valid_" + head
        train_valid_data_loader[data_load_name] = valid_loader

    test_sets = {}
    stop_first_test = False # Initialize a flag to determine if testing should stop after the first test
    test_data_loader = {}
    # Check if all heads have the same test file and if it is not None
    if all(head_config.test_file == head_configs[0].test_file for head_config in head_configs) and head_configs[0].test_file is not None:
        stop_first_test = True
    # Check if all heads have the same test directory and if it is not None
    if all(head_config.test_dir == head_configs[0].test_dir for head_config in head_configs) and head_configs[0].test_dir is not None:
        stop_first_test = True

    # load test data head by head, stop if all heads have same test file
    for head_config in head_configs:
        if check_path_ase_read(head_config.train_file):
            for name, subset in head_config.collections.tests:
                test_sets[name] = [data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max, heads=heads)
                                    for config in subset]
    
        for test_name, test_set in test_sets.items():
            test_loader = torch_geometric.dataloader.DataLoader(
                test_set,
                batch_size=args.valid_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
            test_data_loader[test_name] = test_loader
        if stop_first_test:
            break

    for swa_eval in swas:
        epoch = checkpoint_handler.load_latest(state=tools.CheckpointState(model, optimizer, lr_scheduler),swa=swa_eval,device=device,)
        model.to(device)

        if swa_eval:
            logging.info(f"Loaded Stage two model from epoch {epoch} for evaluation")
        else:
            logging.info(f"Loaded Stage one model from epoch {epoch} for evaluation")

        # freeze model params
        for param in model.parameters():
            param.requires_grad = False

        table_train_valid = create_error_table(
                                                table_type=args.error_table,
                                                all_data_loaders=train_valid_data_loader,
                                                model=model,
                                                loss_fn=loss_fn,
                                                output_args=output_args,
                                                log_wandb=False,
                                                device=device,
                                            )
        logging.info("Error-table on TRAIN and VALID:\n" + str(table_train_valid))

        if test_data_loader:
            table_test = create_error_table(
                                                table_type=args.error_table,
                                                all_data_loaders=test_data_loader,
                                                model=model,
                                                loss_fn=loss_fn,
                                                output_args=output_args,
                                                log_wandb=False,
                                                device=device,
                                            )
            logging.info("Error-table on TEST:\n" + str(table_test))

        # Save entire model
        if swa_eval:
            model_path = Path(args.checkpoints_dir) / (tag + "_stagetwo.model")
            torch.save(model, Path(args.model_dir) / (args.name + "_stagetwo.model"))
            logging.info(f"Saved stagetwo model at {Path(args.model_dir) / (args.name + '_stagetwo.model')}")
        else:
            model_path = Path(args.checkpoints_dir) / (tag + ".model")
            torch.save(model, Path(args.model_dir) / (args.name + ".model"))
            logging.info(f"Saved model at {Path(args.model_dir) / (args.name + '.model')}")

        if args.save_cpu:
            model = model.to("cpu")
        torch.save(model, model_path)

        logging.info(f"Saved model to {model_path}")


    logging.info("Done")

if __name__ == "__main__":
    main()
