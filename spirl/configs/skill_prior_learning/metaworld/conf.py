import os

from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from spirl.components.logger import Logger
from spirl.utils.general_utils import AttrDict
from spirl.components.evaluator import TopOfNSequenceEvaluator
from spirl.data.metaworld.src.metaworld_data_loader import D4RLSequenceSplitDataset

current_dir = os.path.dirname(os.path.realpath(__file__))

data_spec = AttrDict(
    dataset_class=D4RLSequenceSplitDataset,
    n_actions=4,
    state_dim=39,
    env_name="something",
    res=128,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 280

configuration = {
    'model': ClSPiRLMdl,
    'logger': Logger,
    'data_dir': '.',
    'epoch_cycles_train': 50,
    'num_epochs': 100,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=5e-4,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    cond_decode=True,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1  # flat last action from seq gets cropped
