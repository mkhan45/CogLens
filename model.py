from mynn.layers.dense import dense
from mynn.initializers.normal import normal


class Model:
    def __init__(self, dim_input, dim_output):
        """ Initialize the model. """
        self.layer = dense(dim_input, dim_output, weight_initializer=normal)

    def __call__(self, x):
        """ Performs a forward pass on the model.

        Parameters
        ----------
        x : np.ndarray, shape=(N, dim_input)
            The training data.

        Returns
        -------
        preds : mg.Tensor, shape=(dim_output)
            The model's predictions.
        """
        return self.layer(x)

    @property
    def parameters(self):
        return self.layer.parameters
