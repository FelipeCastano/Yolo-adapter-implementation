import torch.nn as nn
from ultralytics.nn.modules.block import C2f


class YOLOAdapterBlock(nn.Module):
    """Lightweight bottleneck adapter inserted in parallel with a C2f block.

    Performs a spatial downsampling via :class:`~torch.nn.Conv2d` followed by
    an upsampling via :class:`~torch.nn.ConvTranspose2d`, forcing the adapter
    to learn a compressed representation before being added element-wise to
    the C2f output.

    :param c1: Number of input channels.
    :type c1: int
    :param c2: Number of output channels.
    :type c2: int
    :param r: Channel expansion ratio used to compute the hidden dimension
        as ``max(1, int(c1 * r))``. Assigned dynamically based on the
        parameter count of the parallel C2f block.
    :type r: float
    :param k: Kernel size for both convolutional layers.
    :type k: int
    :param s: Stride for both convolutional layers.
    :type s: int
    :param p: Padding for both convolutional layers.
    :type p: int
    """

    def __init__(self, c1: int, c2: int, r: float = 0.25, k: int = 3, s: int = 2, p: int = 1):
        super().__init__()
        c_hidden = max(1, int(c1 * r))
        self.conv2d = nn.Conv2d(c1, c_hidden, kernel_size=k, stride=s, padding=p, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv_transpose = nn.ConvTranspose2d(c_hidden, c2, kernel_size=k, stride=s, padding=p, bias=True)

    def forward(self, x):
        """Run the adapter forward pass.

        Downsamples the input spatially, applies ReLU, then upsamples back to
        the original spatial dimensions using ``output_size`` to handle inputs
        with odd spatial dimensions.

        :param x: Input feature map of shape ``(B, C_in, H, W)``.
        :type x: torch.Tensor
        :return: Output feature map of shape ``(B, C_out, H, W)``.
        :rtype: torch.Tensor
        """
        return self.conv_transpose(self.relu(self.conv2d(x)), output_size=x.shape[2:])


class C2f_Adapter(C2f):
    """Drop-in replacement for :class:`~ultralytics.nn.modules.block.C2f` that
    adds a parallel :class:`YOLOAdapterBlock` whose output is summed
    element-wise with the original C2f output.

    The expansion ratio ``r`` is assigned dynamically by
    :class:`~adapter_manager.AdapterManager` based on the parameter count of
    the original C2f block:

    * **Heavy** (``N_p > 5M``): ``r = 1.0``
    * **Medium** (``1M < N_p ≤ 5M``): ``r = 0.5``
    * **Light** (``N_p ≤ 1M``): ``r = 0.25``

    :param c1: Number of input channels.
    :type c1: int
    :param c2: Number of output channels.
    :type c2: int
    :param n: Number of bottleneck repetitions.
    :type n: int
    :param shortcut: Whether to use residual shortcuts in bottleneck blocks.
    :type shortcut: bool
    :param g: Number of groups for grouped convolution.
    :type g: int
    :param e: Channel expansion ratio for the hidden dimension inside C2f.
    :type e: float
    :param r: Channel expansion ratio for the adapter hidden dimension.
    :type r: float
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False,
                 g: int = 1, e: float = 0.5, r: float = 0.25):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.adapter = YOLOAdapterBlock(c1, c2, r=r)

    def forward(self, x):
        """Run the C2f forward pass and add the adapter output element-wise.

        :param x: Input feature map of shape ``(B, C_in, H, W)``.
        :type x: torch.Tensor
        :return: Sum of the C2f output and the adapter output,
            both of shape ``(B, C_out, H, W)``.
        :rtype: torch.Tensor
        """
        return super().forward(x) + self.adapter(x)