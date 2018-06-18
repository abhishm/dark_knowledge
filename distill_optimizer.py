from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf

class DistillOptimizer(optimizer.Optimizer):
    """Implementation of optimizer used in `Distilling the knowledge in a
    Neural Net paper`.
    See [Hinton et. al., 2014](https://arxiv.org/pdf/1503.02531.pdf)
    @@__init__
    """
    def __init__(self, learning_rate=0.001, mu=0.9, max_norm=15., use_locking=False, name="DistillOptimizer"):
        super(DistillOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._mu = mu
        self._max_norm = max_norm

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._mu_t = None
        self._max_norm_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._mu_t = ops.convert_to_tensor(self._mu, name="mu")
        self._max_norm_t = ops.convert_to_tensor(self._max_norm, name="max_norm")

    def _create_slots(self, var_list):
        # Create slots for the first moment.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        max_norm_t = math_ops.cast(self._max_norm_t, var.dtype.base_dtype)

        m = self.get_slot(var, "m")
        m_t = m.assign(mu_t * m - (1. - mu_t) * lr_t * grad)

        var_update = state_ops.assign(var, tf.clip_by_norm(var + m_t, max_norm_t, axes=0))
        return control_flow_ops.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
