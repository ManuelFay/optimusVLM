from transformers import Idefics3Model

from optimus_vlm.models.optimus import OptimusModel


class OptimusVLM(Idefics3Model):
    def __init__(self, config):
        super(OptimusVLM, self).__init__(config=config)
        self.model: Idefics3Model = Idefics3Model(config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class OptimusVLMForMLM(OptimusVLM):
    def __init__(self, config):
        super(OptimusVLMForMLM, self).__init__(config=config)
        self.mlm = True
        self.lm_head = self.model.lm_head

    def forward(self, *args, **kwargs):
        hidden_states = self.model(*args, **kwargs)
        return self.lm_head(hidden_states)