from utils.register import register_llm_model
from torch.utils.data import Dataset
import pandas as pd
import ml_collections
from featurizer import *
from torch_geometric.data import Batch, Data
import torch
from models.encoder import *

llm_model_dict = register_llm_model()

class GraphDataset(Dataset):
    def __init__(self, data_path, data_config):
        super(GraphDataset, self).__init__()

        self.data = pd.read_csv(data_path)
        self.graph_feat_config = data_config['graph_featurizer']
        self.name_idx = self.data['pdb_id'].to_list()
        self.data_config = data_config

        llm_model_config = data_config['llm_config']
        self.llm_feature_extractor = llm_model_dict[llm_model_config['name']](llm_model_config['model_path'], llm_model_config)

        if data_config.get("latent_compressed"):
            compressed_model_config = data_config["latent_compressed"]
            if compressed_model_config["encoder"]["name"] == "mlp":
                self.encoder = MLPEncoder(**compressed_model_config["encoder"]["config"])
            else:
                raise NotImplementedError(compressed_model_config["encoder"]["name"])
            return_keys = self.encoder.load_state_dict(torch.load(compressed_model_config["model_path"], map_location="cpu")["encoder"], strict=True)
            print("encoder return_keys: ", return_keys)
            self.encoder.to(device=llm_model_config["device"])
        else:
            self.encoder = None

        print(f'dataset size= {len(self.name_idx)}')

    def __len__(self):
        return len(self.name_idx)

    def __getitem__(self, index):
        if index is None:
            return None
        pdb_id = self.name_idx[index]
        ret = get_structure(self.data_config, pdb_id)
        c = ml_collections.ConfigDict(self.graph_feat_config)
        graph_coords = ret['coords']
        graph_seq = ret['seq']
        graph_mask = ret['mask']
        num_posenc = c.num_posenc
        num_rbf = c.num_rbf
        knn_num = c.knn_num

        data = construct_graph_data_single(graph_coords, graph_seq, graph_mask, num_posenc, num_rbf, knn_num)

        embedding = self.llm_feature_extractor(ret['seq'])
        if self.encoder is not None:
            with torch.no_grad():
                embedding = self.encoder(embedding)

        graph = Data(
            x=embedding,
            extra_x=data["extra_x"],
            pos=data["pos"],
            edge_index=data["edge_index"],
            edge_attr=data["edge_attr"],
            node_v=data["node_v"],
            edge_v=data["edge_v"],
            coords=data["coords"],
            coords_mask=ret['coord_mask'],
            seq=data['seq'],
            name=ret['name'],
        )

        return {"graph": graph}

    def collate_fn(self, batch):
        batch_graph_list = []
        for info in batch:
            if info is None:
                continue
            batch_graph_list.append(info['graph'])
        batch_graph = Batch.from_data_list(batch_graph_list)
        return batch_graph
