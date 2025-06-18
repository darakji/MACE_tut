###########################################################################################
# Modified version of below code | Modified by Rushikesh

# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)

# Modifications, Simplified as per my needs and annoted with comments
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.tools.scatter import scatter_sum

from mace.modules.blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from mace.modules.utils import (
    get_edge_vectors_and_lengths,
    get_outputs,
)
# pylint: disable=C0302

# for configure model and other
import ast
import logging

from mace import modules
from mace.tools.finetuning_utils import load_foundations_elements
from mace.tools.scripts_utils import extract_config_mace_model


# Models
@compile_mode("script")
class MACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
    ):
        super().__init__()
        # buffers
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )

        # handle heads
        if heads is None:
            heads = ["default"]
        self.heads = heads

        # handle correlations
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions

        # Embedding irreps
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])

        # node embedding
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )

        # radial embedding/bessel
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )

        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()

        # spherical/angular embedding
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]


        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        # First product block
        node_feats_irreps_out = inter.target_irreps

        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        # first readout
        readout = LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"))
        self.readouts = torch.nn.ModuleList([readout])


        for i in range(num_interactions - 1):
            # for last layer, select only scalers
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            
            # get interaction block
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)

            # get product block
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)

            # get readout
            # for last layer, non linear readout
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{len(heads)}x0e"),
                        len(heads),
                    )
                )
            # for other layers linear layout
            else:
                self.readouts.append(
                    LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"))
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:

        # Setup
        data["node_attrs"].requires_grad_(True) # (num_nodes, num_elements)
        data["positions"].requires_grad_(True)  # (num_nodes, 3)

        num_atoms_arange = torch.arange(data["positions"].shape[0])     # (num_nodes,)
        # ptr = (batch_size+1), [0,12,..], each ptr[i] shows starting index of ith molecule/graph
        num_graphs = data["ptr"].numel() - 1        # int: batch_size=num_graphs

        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )   # (num_nodes) int32, each idx shows which head that node belongs to

        # (batch_size, 3, 3)
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )

        # Atomic energies
        # self.atomic_energies_fn(data["node_attrs"]) => (num_nodes, num_heads)
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]   # nodes_e0 = (num_nodes), selects only the energy of head under which node comes

        # e0 (batch_size) => sums up energy of nodes in each molecule
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        )  
        

        # Embeddings 
        # node embedding/features (num_nodes, num_channels_of_0e in hidden rep)
        node_feats = self.node_embedding(data["node_attrs"])    

        # vectors (num_edges,3), lengths (num_edges,1)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        # edge_attrs=angular_emb = (num_edges, (max_ell+1)^2)
        edge_attrs = self.spherical_harmonics(vectors)
        # edge_feats= radial_emb/bessel (num_edges, num_bessels), with polynomial cutoff applied
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )


        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        node_feats_list = []

        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            # node_feats (num_nodes, num_channels, (max_ell+1)^2)
            # sc (num_nodes, hidden_irrep.dim())
            # RadialMLP is applied inside
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            
            # node_feats (num_nodes, hidden_irrep.dim())
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_feats_list.append(node_feats)

            # readout (num_nodes, num_heads)
            # node_energies (num_nodes), only the energy corresponding to head of config is selected
            node_energies = readout(node_feats, node_heads)[
                num_atoms_arange, node_heads
            ] 
            # energy (batch_size)
            energy = scatter_sum(
                src=node_energies,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )

            energies.append(energy)
            node_energies_list.append(node_energies)

        # Concatenate node features
        # (num_nodes, (num_interactions-1)*hidden_irreps.dim() + scalar_channels_hidden_reps)
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        # in contribution, 1 from atomic_energies and other from each readout of interactions
        contributions = torch.stack(energies, dim=-1)   #(batch_size, 1+num_interactions)
        total_energy = torch.sum(contributions, dim=-1)  # (batch_size)
        node_energy_contributions = torch.stack(node_energies_list, dim=-1) #(batch_size, num_nodes)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # (num_nodes)


        # Outputs
        forces, virials, stress, hessian = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
        )

        return {
            "energy": total_energy, # (batch_size)
            "node_energy": node_energy, #(num_nodes)
            "contributions": contributions, #(batch_size, 1+num_interactions)
            "forces": forces,   #(num_nodes, 3)
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
        }

@compile_mode("script")
class ScaleShiftMACE(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # scale shift module
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        
        # Setup
        data["positions"].requires_grad_(True) # (num_nodes, num_elements)
        data["node_attrs"].requires_grad_(True) # (num_nodes, 3)

        num_atoms_arange = torch.arange(data["positions"].shape[0])     # (num_nodes,)
        # ptr = (batch_size+1), [0,12,..], each ptr[i] shows starting index of ith molecule/graph
        num_graphs = data["ptr"].numel() - 1        # int: batch_size=num_graphs

        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )   # (num_nodes) int32, each idx shows which head that node belongs to

        # (batch_size, 3, 3)
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )


        # Atomic energies
        # self.atomic_energies_fn(data["node_attrs"]) => (num_nodes, num_heads)
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ] # nodes_e0 = (num_nodes), selects only the energy of head under which node comes

        # e0 (batch_size) => sums up energy of nodes in each molecule
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        )  

        # Embeddings 
        # node embedding/features (num_nodes, num_channels_of_0e in hidden rep)
        node_feats = self.node_embedding(data["node_attrs"])    

        # vectors (num_edges,3), lengths (num_edges,1)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        # edge_attrs=angular_emb = (num_edges, (max_ell+1)^2)
        edge_attrs = self.spherical_harmonics(vectors)
        # edge_feats= radial_emb/bessel (num_edges, num_bessels), with polynomial cutoff applied
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        
        # Interactions
        node_es_list = []
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            # node_feats (num_nodes, num_channels, (max_ell+1)^2)
            # sc (num_nodes, hidden_irrep.dim())
            # RadialMLP is applied inside
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )

            # node_feats (num_nodes, hidden_irrep.dim())
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_feats_list.append(node_feats)

            # readout (num_nodes, num_heads)
            # node_energies (num_nodes), only the energy corresponding to head of config is selected
            node_es_list.append(
                readout(node_feats, node_heads)[num_atoms_arange, node_heads]
            )  # [(n_nodes, ), ] -> node_es_list = (num_interactions, n_nodes)


        # Concatenate node features
        # (num_nodes, (num_interactions-1)*hidden_irreps.dim() + scalar_channels_hidden_reps)
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over interactions
        # torch.stack(node_es_list, dim=0) (num_interactions, n_nodes)
        # interaction energies
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)  # (n_nodes, )
        # scale it
        node_inter_es = self.scale_shift(node_inter_es, node_heads)  # (n_nodes,)

        # Sum over nodes in graph
        # (batch_size,)
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
        ) 

        # Add E_0 and (scaled) interaction energy
        total_energy = e0 + inter_e #(batch_size)
        node_energy = node_e0 + node_inter_es   #(num_nodes)

        # outputs
        forces, virials, stress, hessian = get_outputs(
            energy=inter_e,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
        )
        output = {
            "energy": total_energy, # (batch_size)
            "node_energy": node_energy, #(num_nodes)
            "interaction_energy": inter_e, #(batch_size, 1+num_interactions)
            "forces": forces,  #(num_nodes, 3)
            "virials": virials,
            "stress": stress,
            "hessian": hessian,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }
        return output
    

# utils
def configure_model(args, train_loader, atomic_energies, model_foundation=None, heads=None, z_table=None):
    # Selecting outputs
    compute_virials = args.loss in ("stress", "virials", "huber", "universal")
    if compute_virials:
        args.compute_stress = True
        args.error_table = "PerAtomRMSEstressvirials"

    output_args = {
        "energy": args.compute_energy,
        "forces": args.compute_forces,
        "virials": compute_virials,
        "stress": args.compute_stress,
        "dipoles": args.compute_dipole,
    }
    logging.info(
        f"During training the following quantities will be reported: {', '.join([f'{report}' for report, value in output_args.items() if value])}"
    )
    logging.info("===========MODEL DETAILS===========")

    if args.scaling == "no_scaling":
        args.std = 1.0
        logging.info("No scaling selected")
    elif (args.mean is None or args.std is None) and args.model != "AtomicDipolesMACE":
        args.mean, args.std = modules.scaling_classes[args.scaling](
            train_loader, atomic_energies
        )

    # Build model
    if model_foundation is not None and args.model in ["MACE", "ScaleShiftMACE"]:
        logging.info("Loading FOUNDATION model")
        model_config_foundation = extract_config_mace_model(model_foundation)
        model_config_foundation["atomic_energies"] = atomic_energies
        model_config_foundation["atomic_numbers"] = z_table.zs
        model_config_foundation["num_elements"] = len(z_table)
        args.max_L = model_config_foundation["hidden_irreps"].lmax

        if args.model == "MACE" and model_foundation.__class__.__name__ == "MACE":
            model_config_foundation["atomic_inter_shift"] = [0.0] * len(heads)
        else:
            model_config_foundation["atomic_inter_shift"] = (
                _determine_atomic_inter_shift(args.mean, heads)
            )
        model_config_foundation["atomic_inter_scale"] = [1.0] * len(heads)
        args.avg_num_neighbors = model_config_foundation["avg_num_neighbors"]
        args.model = "FoundationMACE"
        model_config_foundation["heads"] = heads
        model_config = model_config_foundation

        logging.info("Model configuration extracted from foundation model")
        logging.info("Using universal loss function for fine-tuning")
        logging.info(
            f"Message passing with hidden irreps {model_config_foundation['hidden_irreps']})"
        )
        logging.info(
            f"{model_config_foundation['num_interactions']} layers, each with correlation order: {model_config_foundation['correlation']} (body order: {model_config_foundation['correlation']+1}) and spherical harmonics up to: l={model_config_foundation['max_ell']}"
        )
        logging.info(
            f"Radial cutoff: {model_config_foundation['r_max']} A (total receptive field for each atom: {model_config_foundation['r_max'] * model_config_foundation['num_interactions']} A)"
        )
        logging.info(
            f"Distance transform for radial basis functions: {model_config_foundation['distance_transform']}"
        )
    else:
        logging.info("Building model")
        logging.info(
            f"Message passing with {args.num_channels} channels and max_L={args.max_L} ({args.hidden_irreps})"
        )
        logging.info(
            f"{args.num_interactions} layers, each with correlation order: {args.correlation} (body order: {args.correlation+1}) and spherical harmonics up to: l={args.max_ell}"
        )
        logging.info(
            f"{args.num_radial_basis} radial and {args.num_cutoff_basis} basis functions"
        )
        logging.info(
            f"Radial cutoff: {args.r_max} A (total receptive field for each atom: {args.r_max * args.num_interactions} A)"
        )
        logging.info(
            f"Distance transform for radial basis functions: {args.distance_transform}"
        )

        assert (
            len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
        ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"

        logging.info(f"Hidden irreps: {args.hidden_irreps}")

        model_config = dict(
            r_max=args.r_max,
            num_bessel=args.num_radial_basis,
            num_polynomial_cutoff=args.num_cutoff_basis,
            max_ell=args.max_ell,
            interaction_cls=modules.interaction_classes[args.interaction],
            num_interactions=args.num_interactions,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps(args.hidden_irreps),
            atomic_energies=atomic_energies,
            avg_num_neighbors=args.avg_num_neighbors,
            atomic_numbers=z_table.zs,
        )
        model_config_foundation = None

    model = _build_model(args, model_config, model_config_foundation, heads)

    if model_foundation is not None:
        model = load_foundations_elements(
            model,
            model_foundation,
            z_table,
            load_readout=args.foundation_filter_elements,
            max_L=args.max_L,
        )

    return model, output_args


def _determine_atomic_inter_shift(mean, heads):
    if isinstance(mean, np.ndarray):
        if mean.size == 1:
            return mean.item()
        if mean.size == len(heads):
            return mean.tolist()
        logging.info("Mean not in correct format, using default value of 0.0")
        return [0.0] * len(heads)
    if isinstance(mean, list) and len(mean) == len(heads):
        return mean
    if isinstance(mean, float):
        return [mean] * len(heads)
    logging.info("Mean not in correct format, using default value of 0.0")
    return [0.0] * len(heads)


def _build_model(args, model_config, model_config_foundation, heads):  # pylint: disable=too-many-return-statements
    if args.model == "MACE":
        return ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=[0.0] * len(heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
        )
    if args.model == "ScaleShiftMACE":
        return ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=args.mean,
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
        )
    if args.model == "FoundationMACE":
        return ScaleShiftMACE(**model_config_foundation)
    
    raise RuntimeError(f"Unknown model: '{args.model}'")
