from .rfm_heads_trainer import RFMHeadsTrainer

# should be the same as RFMHeadsTrainer


class ReWiNDTrainer(RFMHeadsTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
