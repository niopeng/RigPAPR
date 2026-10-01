import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch import autocast
import torch.nn.functional as F
import tinycudann as tcnn
import warnings
import typing as T
from .utils import activation_func


def get_mapping_mlp(args, use_amp=False, amp_dtype=torch.float16):
    return MappingMLP(args.mapping_mlp, inp_dim=args.shading_code_dim, out_dim=args.mapping_mlp.out_dim, use_amp=use_amp, amp_dtype=amp_dtype)


class MLP(nn.Module):
    def __init__(self, inp_dim=2, num_layers=3, num_channels=128, out_dim=2, act_type="relu", last_act_type="none",
                 use_wn=False, a=1., b=1., trainable=False, skip_layers=[], bias=True, half_layers=[], residual_layers=[],
                 residual_dims=[]):
        super(MLP, self).__init__()
        self.skip_layers = skip_layers
        self.residual_layers = residual_layers
        self.residual_dims = residual_dims
        assert len(residual_dims) == len(residual_layers)
        wn = weight_norm if use_wn else lambda x, **kwargs: x
        layers = [nn.Identity()]
        num_layers = num_layers + 1
        for i in range(num_layers):
            cur_inp = inp_dim if i == 0 else num_channels
            cur_out = out_dim if i == num_layers - 1 else num_channels
            if (i+1) in half_layers:
                cur_out = cur_out // 2
            if i in half_layers:
                cur_inp = cur_inp // 2
            if i in self.skip_layers:
                cur_inp += inp_dim
            if i in self.residual_layers:
                cur_inp += self.residual_dims[residual_layers.index(i)]
            layers.append(
                wn(nn.Linear(cur_inp, cur_out, bias=bias), name='weight'))
            layers.append(activation_func(act_type=act_type,
                          num_channels=cur_out, a=a, b=b, trainable=trainable))
        layers[-1] = activation_func(act_type=last_act_type,
                                     num_channels=out_dim, a=a, b=b, trainable=trainable)
        assert len(layers) == 2 * num_layers + 1
        self.model = nn.ModuleList(layers)

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, residuals=[]):
        skip_layers = [i*2+1 for i in self.skip_layers]
        residual_layers = [i*2+1 for i in self.residual_layers]
        assert len(residuals) == len(self.residual_layers)
        # print(skip_layers)
        inp = x
        for i, layer in enumerate(self.model):
            if i in skip_layers:
                x = torch.cat([x, inp], dim=-1)
            if i in residual_layers:
                x = torch.cat([x, residuals[residual_layers.index(i)]], dim=-1)
            x = layer(x)
        return x


class MappingMLP(nn.Module):
    def __init__(self, args, inp_dim=2, out_dim=2, use_amp=False, amp_dtype=torch.float16):
        super(MappingMLP, self).__init__()
        self.args = args
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.model = MLP(inp_dim=inp_dim, num_layers=args.num_layers, num_channels=args.dim, out_dim=out_dim,
                         act_type=args.act, last_act_type=args.last_act, use_wn=args.use_wn)
        print("Mapping MLP:\n", self.model)

    def forward(self, x):

        with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            out = self.model(x)
            return out


class TopkMLP(nn.Module):
    def __init__(self, args, num_feats=3, device='cuda', use_amp=False, amp_dtype=torch.float16):
        super(TopkMLP, self).__init__()
        self.args = args
        self.loss_type = args.loss_type.lower()
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.means = args.means
        self.bias = None
        
        if args.learn_bias:
            self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device), requires_grad=True)
        
        if args.init_weights == []:
            assert args.learn == True
            if args.type == 1:    # including d2r
                self.weights = nn.Parameter(torch.cat([torch.ones(1, device=device), torch.zeros(num_feats - 1, device=device)], dim=0), requires_grad=True)
            elif args.type == 2:    # excluding d2r
                self.weights = nn.Parameter(torch.zeros(num_feats - 1, dtype=torch.float32, device=device), requires_grad=True)
        else:
            assert len(args.init_weights) == num_feats - 1
            if args.type == 1:
                self.weights = nn.Parameter(torch.cat([torch.ones(1, device=device), torch.tensor(args.init_weights, device=device, dtype=torch.float32)], dim=0), requires_grad=args.learn)
            elif args.type == 2:
                self.weights = nn.Parameter(torch.tensor(args.init_weights, device=device, dtype=torch.float32), requires_grad=args.learn)

        self.weight_act = activation_func(act_type=args.weight_act)

    @torch.no_grad()
    def get_features(self, feature_list, x=None):
        with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            if self.means == []:
                self.means = [f.mean().item() for f in feature_list]
            elif self.args.means == []:
                cur_means = [f.mean().item() for f in feature_list]
                self.means = [(m + cm) / 2 for m, cm in zip(self.means, cur_means)]

            if self.args.minus_mean:
                feature_list = [f - m for f, m in zip(feature_list, self.means)]
            
            weights = self.weights.clone()
            weights[self.args.act_idxs] = self.weight_act(weights[self.args.act_idxs])

            if self.args.type == 1:   # including D2R
                features = torch.stack(feature_list, dim=-1)
                scores = torch.sum(features * weights, dim=-1)
            elif self.args.type == 2:   # excluding D2R
                features = torch.stack(feature_list[1:], dim=-1)
                scores = torch.sum(features * weights, dim=-1) + feature_list[0]
            else:
                raise NotImplementedError(f"Invalid type: {self.args.type}")

            return scores
        
    def forward(self, feature_list, attn_scores, x=None, step=-1):
        with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            with torch.autograd.grad_mode.set_grad_enabled(self.args.learn):
                if self.args.minus_mean:
                    feature_list = [f - m for f, m in zip(feature_list, self.means)]

                weights = self.weights.clone()
                weights[self.args.act_idxs] = self.weight_act(weights[self.args.act_idxs])

                if self.args.type == 1:
                    features = torch.stack(feature_list, dim=-1)
                    scores = torch.sum(features * weights, dim=-1)
                elif self.args.type == 2:
                    features = torch.stack(feature_list[1:], dim=-1)
                    scores = torch.sum(features * weights, dim=-1) + feature_list[0]
                else:
                    raise NotImplementedError(f"Invalid type: {self.args.type}")
                
                if step % 200 == 0:
                    print("topk mlp weights: ", self.weights.tolist(), weights.tolist())
                    if self.means is not None:
                        print("topk mlp means: ", self.means)
                    if self.bias is not None:
                        print("topk mlp bias: ", self.bias.item())

                if self.bias is not None:
                    scores += self.bias

                if self.loss_type == "bce":
                    labels = torch.ones_like(attn_scores, requires_grad=False)
                    medians = torch.quantile(attn_scores.float(), q=self.args.quantile, dim=-1, keepdim=True)[0]
                    labels[attn_scores > medians] = 0   # Highers attn scores should have lower score values, since topk selects the lowest values
                    if self.args.quantile != 0.5 and self.args.weight_quantile:
                        # weight the loss
                        weights = torch.ones_like(attn_scores, requires_grad=False)
                        weights[attn_scores > medians] = 0.5 / (1 - self.args.quantile)
                        weights[attn_scores <= medians] = 0.5 / self.args.quantile
                    else:
                        weights = None
                    loss = nn.BCEWithLogitsLoss(weight=weights)(scores * self.args.logit_scale, labels)

                return loss


def init_weight(
    weight: torch.Tensor,
    w_init_gain: str = "linear",
    init_method: str = "xavier_normal",
    lrelu_nslope: float = 0.01,
    kaiming_fan_mode: str = "fan_in",
):
    """
    A helper function to initialize the weights of a linear/convolutional layer.

    Args:
        weight:
            (*), an n-dimensional torch.Tensor to be initialized
        w_init_gain:
            The nonlinearity after the linear layer. Can be chosen from the functions supported by
            torch.nn.init.calculate_gain.
            This includes:
            'linear', 'relu', 'silu', 'leaky_relu', 'relu', 'tanh', 'sigmoid'.
        init_method:
            The initialization method. Can be chosen from:
                'normal':
                    randomly sampeld from a Gaussian distribution
                'uniform':
                    randomly sampeld from a uniform distribution
                'xavier_uniform':
                    Check :py:func:`torch.nn.init.xavier_uniform_`.
                'xavier_normal'
                    Check :py:func:`torch.nn.init.xavier_normal_`.
                'xavier':
                    same as 'xavier_normal'
                'kaiming_uniform':
                    Check :py:func:`torch.nn.init.kaiming_uniform_`.
                'kaiming_normal'
                    Check :py:func:`torch.nn.init.kaiming_normal_`.
                'kaiming':
                    same as 'kaiming_normal'
                'orthogonal':
                    Check :py:func:`torch.nn.init.orthogonal_`.
        lrelu_nslope:
            Negative slope used in the leaky-relu.
        kaiming_fan_mode:
            Fan mode used by kaiming_* init_methods.
    Returns:
        Does not return. The function directly modifies the content of weight.

    Note that the function contains torch.no_grad, so there is no need to wrap it with one.
    """

    # handle silu/swish
    if w_init_gain in {"silu", "swish"}:
        # since silu and relu has similar shape, use the gain for relu
        w_init_gain = "relu"

    # calculate gain
    if w_init_gain == "leaky_relu":
        gain = torch.nn.init.calculate_gain(w_init_gain, lrelu_nslope)
        kaiming_a = lrelu_nslope
    else:
        gain = torch.nn.init.calculate_gain(w_init_gain)
        kaiming_a = 0

    if init_method in {
        "kaiming_uniform",
        "kaiming_normal",
        "kaiming",
    } and w_init_gain not in {"relu", "leaky_relu"}:
        print("using kaiming init method on %s, not recommended" % (w_init_gain))

    if init_method == "normal":
        torch.nn.init.normal_(weight, 0.0, gain)
    if init_method == "uniform":
        torch.nn.init.uniform_(weight, -gain, gain)
    elif init_method == "xavier_uniform":
        torch.nn.init.xavier_uniform_(weight, gain=gain)
    elif init_method == "xavier_normal" or init_method == "xavier":
        torch.nn.init.xavier_normal_(weight, gain=gain)
    elif init_method == "kaiming_uniform":
        torch.nn.init.kaiming_uniform_(weight, a=kaiming_a, mode=kaiming_fan_mode, nonlinearity=w_init_gain)
    elif init_method == "kaiming_normal" or init_method == "kaiming":
        torch.nn.init.kaiming_normal_(weight, a=kaiming_a, mode=kaiming_fan_mode, nonlinearity=w_init_gain)
    elif init_method == "orthogonal":
        torch.nn.init.orthogonal_(weight, gain=gain)
    else:
        raise NotImplementedError("initialization method [%s] is not implemented" % init_method)


class LinearLayer(nn.Linear):
    """
    A helper class that wraps around the typical nn.Linear to help initialize it.
    """

    _FLOAT_MODULE = nn.Linear

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        w_init_gain: str = "linear",
        init_method: str = "xavier_normal",
        lrelu_nslope: float = 0.01,
        kaiming_fan_mode: str = "fan_in",
        bias_init_val: float = 0.0,
        lr_multiplier: float = 1.0,
        fixed_bias: float = None,
    ):
        """
        Create a linear layer whose weights are initialized with a chosen method.
        The usage of the layer is the same as torch.nn.Linear.

        Args:
            in_features:
                Size of the last dimension of the input tensor.
            out_features:
                Size of the last dimension of the output tensor.
            bias:
                Whether to add a learnable bias term, one for each out_features.
            w_init_gain:
                The nonlinearity that is designed to be added after the layer.
                Check :py:func:`cdslib.nn.init_weight` for supported nonlinearities.
                Note that the layer does not add nonlinearity.
            init_method:
                The initialization method. Check :py:func:`cdslib.nn.init_weight`.
            lrelu_nslope:
                The negative slope of leaky-relu if leaky-relu is used. Check :py:func:`cdslib.nn.init_weight`.
            kaiming_fan_mode:
                The fan mode if kaiming_* init method is used. Check :py:func:`cdslib.nn.init_weight`.
            bias_init_val:
                Initialization value of the bias.
            lr_multiplier:
                A scalar to be multiplied with the learning rate.
                For example, if lr_multiplier = 0.1, both the weight and the bias will update at 1/10 of the rate.
            fixed_bias (float):
                A fixed bias value to add to the output. If `None`, nothing is added.
        """
        nn.Linear.__init__(self, in_features, out_features, bias)
        init_weight(
            self.weight,
            w_init_gain=w_init_gain,  # leaky_relu, relu, tanh, sigmoid, ...
            init_method=init_method,
            lrelu_nslope=lrelu_nslope,
            kaiming_fan_mode=kaiming_fan_mode,
        )
        # init bias
        if self.bias is not None:
            nn.init.constant_(self.bias, bias_init_val)

        # learning rate multiplier (make the layer update slower or faster)
        self.lr_multiplier = lr_multiplier

        # add constant at the end
        self.fixed_bias = fixed_bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input:
                :math:`(*, C_{in})`, input tensor.

        Returns:
            :math:`(*, C_{out})`, output tensor.
        """
        if self.bias is not None:
            out = F.linear(input, self.weight * self.lr_multiplier, self.bias * self.lr_multiplier)
        else:
            out = F.linear(input, self.weight * self.lr_multiplier)

        if self.fixed_bias is not None:
            out = out + self.fixed_bias

        return out


class StackedLinearLayers(nn.Module):
    """
    Convenient helper nn.Module to create a stack of linear layers.
    """

    def __init__(
        self,
        num_layers: int,
        dim_input: int,
        dim_output: int,
        dim_features: T.Union[T.Sequence[int], int],
        nonlinearity: str = "leaky_relu",
        add_norm_layer: bool = False,
        norm_fun: T.Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
        output_add_nonlinearity: bool = False,
    ):
        """
        Convenient helper nn.Module to create a stack of linear layers.

        Args:
            num_layers:
                Total number of linear layers to create.
            dim_input:
                Feature dimension of the input tensor, which is :math:`(*, C_{in})`.
            dim_output:
                Feature dimension of the output tensor, which is :math:`(*, C_{out})`.
            dim_features:
                An integer if all layers share the same feature dimension,
                or a list of num_layer-1 integers, one for each layer except the last layer.
            nonlinearity:
                Nonlinearity used after each linear layer (except the last layer if output_add_nonlinearity is False).
                Choose from: `leaky_relu`, `relu`, `tanh`, `sigmoid`, `silu`, `swish`
                Note that silu (swish) is supported in pytorch version >= 1.7.0.
            add_norm_layer:
                Whether to add normalization layers between linear layers
            norm_fun:
                Callable function used to normalize the output of linear layer (before nonlinearity).
                It should be a function that takes dim_feature as input.
                For example, you can pass `nn.LayerNorm`.
                If you want to control additional functionality like the eps and elementwise_affine of nn.LayerNorm,
                you can pass a lambda function:
                lambda dim: torch.nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)
            dropout_prob:
                Dropout probability added after nonlinearity. If 0, no dropout layer is added.
            output_add_nonlinearity:
                Whether to add nonlinearity (norm_layer, and dropout) at the last layer

        Note that the order of the layers is:
            Linear -> normalization (if add_norm_layer) -> nonlinearity -> dropout.

        """
        super().__init__()
        self.dim_input = dim_input
        self.num_layers = num_layers
        self.dim_output = dim_output
        self.add_norm_layer = add_norm_layer
        self.linear_bias = not self.add_norm_layer  # if added normalization layer, no need to learn bias
        self.norm_fun = norm_fun
        self.dropout_prob = dropout_prob
        self.output_add_nonlinearity = output_add_nonlinearity

        # check dim_features
        if isinstance(dim_features, int):
            self.dim_features = [dim_features for _ in range(self.num_layers - 1)]
        elif isinstance(dim_features, T.Sequence):
            if len(dim_features) == 1:
                self.dim_features = [dim_features[0] for _ in range(self.num_layers - 1)]
            else:
                self.dim_features = dim_features
        else:
            raise ValueError("wrong type dim_features")
        if len(self.dim_features) != self.num_layers - 1 and self.num_layers != 1:
            raise ValueError("wrong length of dim_features (should be num_layers-1)")

        # check dropout probability
        if self.dropout_prob > 0 and self.num_layers == 1:
            self.dropout_prob = 0
            warnings.warn(
                "dropout option adds dropout after all but the last layer."
                "so non-zero dropout expects "
                "num_layers greater than 1, but got dropout={} and "
                "num_layers={}".format(self.dropout_prob, self.num_layers)
            )

        # check nonlinearity
        self.nonlinearity = nonlinearity
        if self.nonlinearity == "leaky_relu":
            nonlinearity_fun = lambda: nn.LeakyReLU(negative_slope=0.01, inplace=False)
        elif self.nonlinearity == "relu":
            nonlinearity_fun = lambda: nn.ReLU(inplace=False)
        elif self.nonlinearity == "tanh":
            nonlinearity_fun = nn.Tanh
        elif self.nonlinearity == "sigmoid":
            nonlinearity_fun = nn.Sigmoid
        elif self.nonlinearity in {"silu", "swish"}:
            nonlinearity_fun = lambda: nn.SiLU(inplace=False)
        else:
            raise ValueError("unsupported nonlinearity")

        self.main = nn.ModuleList()
        self.addons = nn.ModuleList()  # normalization, nonlinearity
        # self.dropouts = nn.ModuleList() if self.dropout_prob > 0 else None  # dropout layers

        # batchnorm and instance norm takes (batch, feature, seq_len) instead of (batch, seq_len, feature)
        self.permute_for_norm = False

        dh = self.dim_input
        for layer_idx in range(self.num_layers - 1):
            self.main.append(
                LinearLayer(
                    in_features=dh,
                    out_features=self.dim_features[layer_idx],
                    bias=self.linear_bias,
                    w_init_gain=self.nonlinearity,
                )
            )
            dh = self.dim_features[layer_idx]

            additional_layers = []
            if self.add_norm_layer:
                layer = self.norm_fun(dh)
                additional_layers.append(layer)
                if isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.InstanceNorm1d):
                    self.permute_for_norm = True
                elif isinstance(layer, nn.LayerNorm):
                    pass
                else:
                    warnings.warn(
                        "Unrecognized normalization function {}, "
                        "make sure you have checked we do not "
                        "need to permute the dimensions.".format(layer)
                    )

            additional_layers.append(nonlinearity_fun())
            if self.dropout_prob > 0:
                additional_layers.append(nn.Dropout(p=self.dropout_prob, inplace=False))

            self.addons.append(nn.Sequential(*additional_layers))

        # output linear layer
        self.main.append(
            LinearLayer(
                in_features=dh,
                out_features=self.dim_output,
                bias=True,
                w_init_gain="linear",
            )
        )

        if self.output_add_nonlinearity:
            dh = self.dim_output
            additional_layers = []
            if self.add_norm_layer:
                layer = self.norm_fun(dh)
                additional_layers.append(layer)
                if isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.InstanceNorm1d):
                    self.permute_for_norm = True
                elif isinstance(layer, nn.LayerNorm):
                    pass
                else:
                    warnings.warn(
                        "Unrecognized normalization function {}, "
                        "make sure you have checked we do not "
                        "need to permute the dimensions.".format(layer)
                    )

            additional_layers.append(nonlinearity_fun())
            if self.dropout_prob > 0:
                additional_layers.append(nn.Dropout(p=self.dropout_prob, inplace=False))
            self.addons.append(nn.Sequential(*additional_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x:
                :math:`(*, C_{in})`, input tensor.

        Returns:
            :math:`(*, C_{out})`, output tensor.
        """
        for i in range(len(self.main) - 1):
            x = self.main[i](x)
            if self.permute_for_norm and len(x.shape) == 3:
                x = x.permute(0, 2, 1)
            x = self.addons[i](x)
            if self.permute_for_norm and len(x.shape) == 3:
                x = x.permute(0, 2, 1)
        x = self.main[-1](x)
        if self.output_add_nonlinearity:
            if self.permute_for_norm and len(x.shape) == 3:
                x = x.permute(0, 2, 1)
            x = self.addons[-1](x)
            if self.permute_for_norm and len(x.shape) == 3:
                x = x.permute(0, 2, 1)
        return x