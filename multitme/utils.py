import numpy as np
from inspect import isclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pyro.distributions.util import broadcast_shape


GLASBEY_30_COLORS = [
    "#e51415",
    "#347fba",
    "#4bb049",
    "#9a4da4",
    "#ff8000",
    "#ffff30",
    "#a85523",
    "#f782c0",
    "#9b9b9b",
    "#004100",
    "#00ffff",
    "#000090",
    "#5f0025",
    "#fff5ff",
    "#4d4d50",
    "#b59bff",
    "#6700f9",
    "#4c006f",
    "#00816c",
    "#72ff9c",
    "#6a6c00",
    "#e2c88c",
    "#63c3fe",
    "#da0084",
    "#593200",
    "#40bcab",
    "#b39c00",
    "#f200ff",
    "#930000",
    "#004d74",
]

def remove_outliers(arr, percentile=99.5):
    arr = arr.copy()
    max_value = np.percentile(arr, percentile)
    arr[arr > max_value] = max_value
    return arr

def extract_channel(acquisition, acquisition_arr, selected_channel):
    channel_names = [
        (a if a else b)
        for (a, b) in zip(acquisition.channel_labels, acquisition.channel_names)
    ]

    if selected_channel not in channel_names:
        raise ValueError(f"Channel {selected_channel} not found in {channel_names}")

    for channel_idx, channel in enumerate(channel_names):
        if channel == selected_channel:
            array = remove_outliers(acquisition_arr[channel_idx, ...])
            return array
        
def extract_channels(acquisition, acquisition_arr, selected_channels):
    image_arrays = []
    for channel in selected_channels:
        arr = extract_channel(acquisition, acquisition_arr, channel)
        image_arrays.append(arr)
    combined_array = np.stack(image_arrays, axis=2)
    return combined_array.astype(int)

def extract_maximum_projection_of_channels(acquisition, acquisition_arr, selected_channels):
    arr = extract_channels(acquisition, acquisition_arr, selected_channels)
    return np.max(arr, axis=2)

class DatasetFromArray(Dataset):
    def __init__(self, signal, annotations=None, reference=None, transform=None, target_transform=None):
        self.annotations = annotations
        self.signal = signal
        self.transform = transform
        self.reference = reference
        self.target_transform = target_transform

    def __len__(self):
        return self.signal.shape[0]

    def __getitem__(self, idx):
        signal = self.signal[idx]
        label = self.annotations[idx]
        reference = self.reference[idx]
        if self.transform:
            signal = self.transform(signal)
        if self.target_transform:
            signal = self.target_transform(signal)
        return signal, label, reference


class Exp(nn.Module):
    """
    a custom module for exponentiation of tensors
    """

    def __init__(self):
        super().__init__()

    def forward(self, val):
        return torch.exp(val)


class ConcatModule(nn.Module):
    """
    a custom module for concatenation of tensors
    """

    def __init__(self, allow_broadcast=False):
        self.allow_broadcast = allow_broadcast
        super().__init__()

    def forward(self, *input_args):
        # we have a single object
        if len(input_args) == 1:
            # regardless of type,
            # we don't care about single objects
            # we just index into the object
            input_args = input_args[0]

        # don't concat things that are just single objects
        if torch.is_tensor(input_args):
            return input_args
        else:
            if self.allow_broadcast:
                shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
                input_args = [s.expand(shape) for s in input_args]
            return torch.cat(input_args, dim=-1)


class ListOutModule(nn.ModuleList):
    """
    a custom module for outputting a list of tensors from a list of nn modules
    """

    def __init__(self, modules):
        super().__init__(modules)

    def forward(self, *args, **kwargs):
        # loop over modules in self, apply same args
        return [mm.forward(*args, **kwargs) for mm in self]


def call_nn_op(op):
    """
    a helper function that adds appropriate parameters when calling
    an nn module representing an operation like Softmax

    :param op: the nn.Module operation to instantiate
    :return: instantiation of the op module with appropriate parameters
    """
    if op in [nn.Softmax, nn.LogSoftmax]:
        return op(dim=1)
    else:
        return op()


class MLP(nn.Module):
    def __init__(
        self,
        mlp_sizes,
        activation=nn.ReLU,
        output_activation=None,
        post_layer_fct=lambda layer_ix, total_layers, layer: None,
        post_act_fct=lambda layer_ix, total_layers, layer: None,
        allow_broadcast=False,
        use_cuda=False,
    ):
        # init the module object
        super().__init__()

        assert len(mlp_sizes) >= 2, "Must have input and output layer sizes defined"

        # get our inputs, outputs, and hidden
        input_size, hidden_sizes, output_size = (
            mlp_sizes[0],
            mlp_sizes[1:-1],
            mlp_sizes[-1],
        )

        # assume int or list
        assert isinstance(
            input_size, (int, list, tuple)
        ), "input_size must be int, list, tuple"

        # everything in MLP will be concatted if it's multiple arguments
        last_layer_size = input_size if type(input_size) == int else sum(input_size)

        # everything sent in will be concatted together by default
        all_modules = [ConcatModule(allow_broadcast)]

        # loop over l
        for layer_ix, layer_size in enumerate(hidden_sizes):
            assert type(layer_size) == int, "Hidden layer sizes must be ints"

            # get our nn layer module (in this case nn.Linear by default)
            cur_linear_layer = nn.Linear(last_layer_size, layer_size)

            # for numerical stability -- initialize the layer properly
            cur_linear_layer.weight.data.normal_(0, 0.001)
            cur_linear_layer.bias.data.normal_(0, 0.001)

            # add our linear layer
            all_modules.append(cur_linear_layer)

            # handle post_linear
            post_linear = post_layer_fct(
                layer_ix + 1, len(hidden_sizes), all_modules[-1]
            )

            # if we send something back, add it to sequential
            # here we could return a batch norm for example
            if post_linear is not None:
                all_modules.append(post_linear)

            # handle activation (assumed no params -- deal with that later)
            all_modules.append(activation())

            # now handle after activation
            post_activation = post_act_fct(
                layer_ix + 1, len(hidden_sizes), all_modules[-1]
            )

            # handle post_activation if not null
            # could add batch norm for example
            if post_activation is not None:
                all_modules.append(post_activation)

            # save the layer size we just created
            last_layer_size = layer_size

        # now we have all of our hidden layers
        # we handle outputs
        assert isinstance(
            output_size, (int, list, tuple)
        ), "output_size must be int, list, tuple"

        if type(output_size) == int:
            all_modules.append(nn.Linear(last_layer_size, output_size))
            if output_activation is not None:
                all_modules.append(
                    call_nn_op(output_activation)
                    if isclass(output_activation)
                    else output_activation
                )
        else:
            # we're going to have a bunch of separate layers we can spit out (a tuple of outputs)
            out_layers = []

            # multiple outputs? handle separately
            for out_ix, out_size in enumerate(output_size):
                # for a single output object, we create a linear layer and some weights
                split_layer = []

                # we have an activation function
                split_layer.append(nn.Linear(last_layer_size, out_size))

                # then we get our output activation (either we repeat all or we index into a same sized array)
                act_out_fct = (
                    output_activation
                    if not isinstance(output_activation, (list, tuple))
                    else output_activation[out_ix]
                )

                if act_out_fct:
                    # we check if it's a class. if so, instantiate the object
                    # otherwise, use the object directly (e.g. pre-instaniated)
                    split_layer.append(
                        call_nn_op(act_out_fct) if isclass(act_out_fct) else act_out_fct
                    )

                # our outputs is just a sequential of the two
                out_layers.append(nn.Sequential(*split_layer))

            all_modules.append(ListOutModule(out_layers))

        # now we have all of our modules, we're ready to build our sequential!
        # process mlps in order, pretty standard here
        self.sequential_mlp = nn.Sequential(*all_modules)

    # pass through our sequential for the output!
    def forward(self, *args, **kwargs):
        return self.sequential_mlp.forward(*args, **kwargs)