from transformers import Idefics3ForConditionalGeneration, Idefics3Model


class OptimusVLM(Idefics3Model):
    def __init__(self, config):
        super(OptimusVLM, self).__init__(config=config)

class OptimusVLMForMLM(Idefics3ForConditionalGeneration):
    def __init__(self, config):
        super(OptimusVLMForMLM, self).__init__(config=config)
