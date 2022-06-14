from fairseq.file_io import PathManager
from fairseq.checkpoint_utils import torch_persistent_save
import torch
import os


def reset_lang2adapter(model_path, model_name, new_model_name, lang2adapter):
    with open(PathManager.get_local_path(os.path.join(model_path, model_name)), "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    state["model"]["decoder.lang2adapter"] = lang2adapter
    state_dict = {
        "cfg": state["cfg"],
        "args": state["args"],
        "model": state["model"],
        "optimizer_history": state["optimizer_history"],
        "extra_state": state["extra_state"],
        "last_optimizer_state": state["last_optimizer_state"],
    }
    with PathManager.open(os.path.join(model_path, new_model_name), "wb") as f:
        torch_persistent_save(state_dict, f)
        print("Successfully saving to {}/{}".format(model_path, new_model_name))


reset_lang2adapter("/path/to/adapter-step1-10W/", "checkpoint_last.pt", "avg16_20.pt", torch.LongTensor(
    [1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 0, 4, 3, 2, 2, 2, 0, 1, 1, 4, 4, 2, 2,
     4, 2, 3, 1, 2, 1, 4, 4, 2, 1, 2, 4, 1, 1, 0, 3, 1, 3, 2, 2, 1, 0, 3, 2, 4, 2, 0, 2, 1, 3, 2, 3, 4, 4, 2, 0]))
reset_lang2adapter("/path.to/adapter-step1-50W/", "checkpoint_last.pt", "avg16_20.pt", torch.LongTensor(
    [1, 0, 0, 0, 3, 0, 0, 2, 2, 1, 0, 0, 1, 1, 2, 2, 2, 4, 0, 3, 2, 2, 2, 0, 1, 4, 2, 0, 2, 0, 1, 2, 1, 2, 2, 1, 1, 0,
     2, 2, 2, 1, 0, 3, 0, 4, 0, 3, 2, 4, 0]))
