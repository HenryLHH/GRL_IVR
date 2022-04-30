from torch import nn

class BaseAgent(nn.Module):

    def __init__(self):
        super().__init__()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def save_model(self, model_path):
        raise NotImplementedError

    def load_model(self, model_path):
        raise NotImplementedError
    

    def process_fn(self, batch, buffer, indices):
        return batch

    def post_process_fn(self, batch, buffer, indices):
        if hasattr(buffer, "update_weight") and hasattr(batch, "weight"):
            buffer.update_weight(indices, batch.weight)


    def __call__(self, batch, state=None):
        # return batch of (action, ...)
        raise NotImplementedError


    def learn(self, batch, batch_size=None):
        raise NotImplementedError


    def sync_weights(self):
        pass