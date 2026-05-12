import torch
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_geometric.data import Batch, Data


def _try_dig_io():
    """Lazy import of the optional Rust extension. Returns the module or ``None``.

    Callers should fall back to the original Python path when this is ``None``.
    The original implementations are *not* removed.
    """
    try:
        import dig_io  # type: ignore
        if dig_io.is_available():
            return dig_io
    except Exception:
        pass
    return None


# Seed counter for the Rust-backed paths. We do NOT touch ``torch.manual_seed``
# so the Python path stays bit-for-bit identical to before.
_RUST_SEED_COUNTER = 0xC0FFEE1234
_U64_MASK = (1 << 64) - 1


def _next_rust_seed() -> int:
    """LCG over u64 (Knuth's MMIX constants), wrapping with a mask to avoid
    NumPy int64 overflow warnings."""
    global _RUST_SEED_COUNTER
    _RUST_SEED_COUNTER = (
        _RUST_SEED_COUNTER * 6364136223846793005 + 1442695040888963407
    ) & _U64_MASK
    return int(_RUST_SEED_COUNTER)


class UniformSample():
    r"""Uniformly node dropping on the given graph or batched graphs.
    Class objects callable via method :meth:`views_fn`.

    Args:
        ratio (float, optinal): Ratio of nodes to be dropped. (default: :obj:`0.1`)
        impl (str, optional): ``'python'`` (default) uses the original PyTorch
            implementation. ``'rust'`` opt-in: the relabel / index sampling is
            delegated to ``dig_io`` (Rust + ChaCha8 PRNG). Falls back silently
            to ``'python'`` when ``dig_io`` is not built.
    """
    def __init__(self, encoder, ratio, device, impl: str = 'python'):
        self.encoder = encoder
        self.ratio = ratio
        self.device = device
        if impl not in ('python', 'rust'):
            raise ValueError(f"impl must be 'python' or 'rust', got {impl!r}")
        self.impl = impl

    def __call__(self, data):
        return self.views_fn(data)

    def _rust_mod_or_none(self):
        return _try_dig_io() if self.impl == 'rust' else None

    def do_trans(self, data):
        rust = self._rust_mod_or_none()
        if 'gin' in self.encoder or 'gcn' in self.encoder:
            node_num, _ = data.x.size()
            device = data.x.device
            _, edge_num = data.edge_index.size()
            keep_num = int(node_num * (1 - self.ratio))

            if rust is not None and node_num > 0:
                ei = np.ascontiguousarray(
                    data.edge_index.detach().cpu().numpy(), dtype=np.int64
                )
                new_ei_np, keep_np = rust.uniform_sample_subgraph(
                    ei, int(node_num), int(keep_num), _next_rust_seed()
                )
                keep_t = torch.as_tensor(keep_np, dtype=torch.long, device=device)
                new_edge_index = torch.as_tensor(new_ei_np, dtype=torch.long, device=device)
                return Data(x=data.x.index_select(0, keep_t), edge_index=new_edge_index)

            idx_nondrop = torch.randperm(node_num, device=device)[:keep_num]
            mask_nondrop = torch.zeros_like(data.x[:, 0]).scatter_(0, idx_nondrop, 1.0).bool()
            edge_index, _ = subgraph(
                mask_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num
            )
            return Data(x=data.x[mask_nondrop], edge_index=edge_index)

        node_num = data.z.shape[0]
        keep_num = int(node_num * (1 - self.ratio))

        if rust is not None and node_num > 0:
            empty_ei = np.zeros((2, 0), dtype=np.int64)
            _, keep_np = rust.uniform_sample_subgraph(
                empty_ei, int(node_num), int(keep_num), _next_rust_seed()
            )
            keep_t = torch.as_tensor(keep_np, dtype=torch.long, device=data.z.device)
            pos = data.pos.index_select(0, keep_t.to(data.pos.device))
            z = data.z.index_select(0, keep_t)
            return Data(pos=pos, smiles=data.smiles, z=z)

        idx_nondrop = torch.randperm(node_num)[:keep_num]
        mask_nondrop = torch.zeros_like(data.z.cpu()).scatter_(0, idx_nondrop, 1.0).bool()
        pos = data.pos[mask_nondrop]
        z = data.z[mask_nondrop]
        return Data(pos=pos, smiles=data.smiles, z=z)

    #def do_trans(self, data):
        #print(1123123)
        #node_num, _ = data.x.size()
        #device = data.x.device
        #_, edge_num = data.edge_index.size()
        
        #keep_num = int(node_num * (1-self.ratio))
        #idx_nondrop = torch.randperm(node_num, device=device)[:keep_num]
        #mask_nondrop = torch.zeros_like(data.x[:,0]).scatter_(0, idx_nondrop, 1.0).bool()    
            
    
    def views_fn(self, data):
        r"""Method to be called when :class:`UniformSample` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)

        
class RWSample():
    """Subgraph sampling based on random walk on the given graph or batched graphs.
    Class objects callable via method :meth:`views_fn`.

    Args:
        ratio (float, optional): Percentage of nodes to sample from the graph.
            (default: :obj:`0.1`)
        add_self_loop (bool, optional): Set True to add self-loop to edge_index.
            (default: :obj:`False`)
        impl (str, optional): ``'python'`` (default) keeps the original
            PyTorch implementation (with the original quirks). ``'rust'``
            uses ``dig_io.rw_sample_subgraph``; behaviour differs in that the
            neighbor frontier *actually* accumulates (the upstream Python
            uses ``set.union`` and discards the result, so the walk collapses
            to first-hop neighbours of the seed).
    """
    def __init__(self, ratio=0.1, add_self_loop=False, impl: str = 'python'):
        self.ratio = ratio
        self.add_self_loop = add_self_loop
        if impl not in ('python', 'rust'):
            raise ValueError(f"impl must be 'python' or 'rust', got {impl!r}")
        self.impl = impl

    def __call__(self, data):
        return self.views_fn(data)

    def _rust_mod_or_none(self):
        return _try_dig_io() if self.impl == 'rust' else None

    def do_trans(self, data):
        device = data.x.device
        node_num, _ = data.x.size()
        sub_num = int(node_num * self.ratio)

        rust = self._rust_mod_or_none()
        if rust is not None and node_num > 0:
            ei = np.ascontiguousarray(
                data.edge_index.detach().cpu().numpy(), dtype=np.int64
            )
            new_ei_np, keep_np = rust.rw_sample_subgraph(
                ei, int(node_num), int(sub_num), _next_rust_seed(),
                bool(self.add_self_loop),
            )
            keep_t = torch.as_tensor(keep_np, dtype=torch.long, device=device)
            new_edge_index = torch.as_tensor(new_ei_np, dtype=torch.long, device=device)
            return Data(x=data.x.index_select(0, keep_t), edge_index=new_edge_index)

        if self.add_self_loop:
            sl = torch.tensor([[n, n] for n in range(node_num)], device=device).t()
            edge_index = torch.cat((data.edge_index, sl), dim=1)
        else:
            edge_index = data.edge_index

        idx_sub = [torch.randint(node_num, size=(1,), device=device)[0]]
        idx_neigh = set([n.item() for n in edge_index[1][edge_index[0]==idx_sub[0]]])

        count = 0
        while len(idx_sub) <= sub_num:
            count = count + 1
            if count > node_num:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = list(idx_neigh)[torch.randperm(len(idx_neigh), device=device)[0]]
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            idx_neigh.union(set([n.item() for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

        idx_sub = torch.LongTensor(idx_sub, device=device)
        mask_nondrop = torch.zeros_like(data.x[:,0]).scatter_(0, idx_sub, 1.0).bool()
        edge_index, _ = subgraph(mask_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)
        return Data(x=data.x[mask_nondrop], edge_index=edge_index)

    def views_fn(self, data):
        r"""Method to be called when :class:`RWSample` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)
