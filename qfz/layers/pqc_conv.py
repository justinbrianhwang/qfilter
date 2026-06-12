"""PQCConv2D: trainable parameterized quantum convolution.

A :class:`~qfz.layers.quanvolution.Quanvolution2D` whose circuit angles
are trained jointly with the rest of the network by backpropagation
through the quantum simulator. Defaults to the hardware-efficient
ansatz, whose parameters are all rotation angles.
"""

from qfz.layers.quanvolution import Quanvolution2D


class PQCConv2D(Quanvolution2D):
    """Trainable quantum convolution layer.

    Identical interface to :class:`Quanvolution2D` but with trainable
    circuit parameters and a variational ansatz by default
    (``circuit="hardware_efficient"``, ``n_layers=2``).

    Example:
        >>> layer = PQCConv2D(in_channels=1, out_channels=4)
        >>> sum(p.numel() for p in layer.parameters() if p.requires_grad)
        16

    See :class:`Quanvolution2D` for the full argument list.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 circuit: str = "hardware_efficient", n_layers: int = 2,
                 trainable: bool = True, **kwargs):
        super().__init__(in_channels, out_channels, circuit=circuit,
                         n_layers=n_layers, trainable=trainable, **kwargs)
