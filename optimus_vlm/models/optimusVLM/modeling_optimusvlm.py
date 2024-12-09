from transformers import Idefics3Model, Idefics3ForConditionalGeneration


class OptimusVLM(Idefics3Model):
    def __init__(self, config):
        super(OptimusVLM, self).__init__(config=config)

class OptimusVLMForMLM(Idefics3ForConditionalGeneration):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super(OptimusVLMForMLM, self).__init__(config=config)
