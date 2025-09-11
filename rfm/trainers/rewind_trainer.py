from rfm.trainers import RFMHeadsTrainer
from transformers import Trainer

# should be the same as RFMHeadsTrainer


class ReWiNDTrainer(RFMHeadsTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
