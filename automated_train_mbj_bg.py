from alignn.data import load_dataset

# from alignn.data import get_train_val_loaders
from alignn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
from alignn.train import train_dgl
from alignn.models.alignn import ALIGNN
import torch
import os

dataset_name = "dft_3d"
k = "mbj_bandgap"
id_tag = "jid"
config_file = "/home/liuke/liuke/CrystalModel/alignn-main/config_mbj_bg.json"
config = loadjson(config_file)


dataset = load_dataset(name=dataset_name, target=k)
size = len(dataset)
batch_size = 32
config["epochs"] = 200
config["batch_size"] = batch_size
config["dataset"] = dataset_name

tconf = TrainingConfig(**config)
print(tconf)
model = ALIGNN()
#model.load_state_dict(torch.load(model_path, map_location=device)["model"])
#model

train_dgl(tconf, model)
"""
import sys
sys.exit()
train_loader, val_loader,  test_loader, prepare_bacth = get_train_val_loaders(
    dataset=dataset_name,
    target=k,
    n_train=size - 2,
    n_test=1,
    n_val=1,
    workers=4,
    id_tag=id_tag,
    batch_size=1,
)
#model.train()

"""
