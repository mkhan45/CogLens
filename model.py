from mynn.layers.dense import dense
from mynn.initializers.glorot_normal import glorot_normal


class Model:
    def __init__(self, dim_input, dim_output):
        """ Initialize the model. """
        self.dense = dense(dim_input, dim_output, weight_initializer=glorot_normal)

    def __call__(self, x):
        """ Performs a forward pass on the model.

        Parameters
        ----------
        x : np.ndarray, shape=(N, dim_input)
            The training data.

        Returns
        -------
        preds : np.ndarray, shape=(dim_output)
            The model's predictions.
        """
        return self.dense(x)

    @property
    def parameters(self):
        return self.dense.parameters
