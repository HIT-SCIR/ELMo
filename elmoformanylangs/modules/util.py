"""
Assorted utilities for working with neural networks in AllenNLP.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple, Callable
import itertools
import math
import torch
from torch.autograd import Variable

def get_lengths_from_binary_sequence_mask(mask: torch.Tensor):
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.
    Parameters
    ----------
    mask : torch.Tensor, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.
    Returns
    -------
    A torch.LongTensor of shape (batch_size,) representing the lengths
    of the sequences in the batch.
    """
    return mask.long().sum(-1)


def sort_batch_by_length(tensor: torch.autograd.Variable,
                         sequence_lengths: torch.autograd.Variable):
    """
    Sort a batch first tensor by some specified lengths.
    Parameters
    ----------
    tensor : Variable(torch.FloatTensor), required.
        A batch first Pytorch tensor.
    sequence_lengths : Variable(torch.LongTensor), required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.
    Returns
    -------
    sorted_tensor : Variable(torch.FloatTensor)
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : Variable(torch.LongTensor)
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : Variable(torch.LongTensor)
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
    permuation_index : Variable(torch.LongTensor)
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.
    """

    if not isinstance(tensor, Variable) or not isinstance(sequence_lengths, Variable):
        raise Exception("Both the tensor and sequence lengths must be torch.autograd.Variables.")

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)

    # This is ugly, but required - we are creating a new variable at runtime, so we
    # must ensure it has the correct CUDA vs non-CUDA type. We do this by cloning and
    # refilling one of the inputs to the function.
    index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths)))
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    index_range = Variable(index_range.long())
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index


def get_final_encoder_states(encoder_outputs: torch.Tensor,
                             mask: torch.Tensor,
                             bidirectional: bool = False) -> torch.Tensor:
    """
    Given the output from a ``Seq2SeqEncoder``, with shape ``(batch_size, sequence_length,
    encoding_dim)``, this method returns the final hidden state for each element of the batch,
    giving a tensor of shape ``(batch_size, encoding_dim)``.  This is not as simple as
    ``encoder_outputs[:, -1]``, because the sequences could have different lengths.  We use the
    mask (which has shape ``(batch_size, sequence_length)``) to find the final state for each batch
    instance.
    Additionally, if ``bidirectional`` is ``True``, we will split the final dimension of the
    ``encoder_outputs`` into two and assume that the first half is for the forward direction of the
    encoder and the second half is for the backward direction.  We will concatenate the last state
    for each encoder dimension, giving ``encoder_outputs[:, -1, :encoding_dim/2]`` concated with
    ``encoder_outputs[:, 0, encoding_dim/2:]``.
    """
    # These are the indices of the last words in the sequences (i.e. length sans padding - 1).  We
    # are assuming sequences are right padded.
    # Shape: (batch_size,)
    last_word_indices = mask.sum(1).long() - 1
    batch_size, _, encoder_output_dim = encoder_outputs.size()
    expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
    # Shape: (batch_size, 1, encoder_output_dim)
    final_encoder_output = encoder_outputs.gather(1, expanded_indices)
    final_encoder_output = final_encoder_output.squeeze(1)  # (batch_size, encoder_output_dim)
    if bidirectional:
        final_forward_output = final_encoder_output[:, :(encoder_output_dim // 2)]
        final_backward_output = encoder_outputs[:, 0, (encoder_output_dim // 2):]
        final_encoder_output = torch.cat([final_forward_output, final_backward_output], dim=-1)
    return final_encoder_output


def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.autograd.Variable):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.
    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Variable, required.
    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    binary_mask = tensor_for_masking.clone()
    binary_mask.data.copy_(torch.rand(tensor_for_masking.size()) > dropout_probability)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask
    
def block_orthogonal(tensor: torch.Tensor,
                     split_sizes: List[int],
                     gain: float = 1.0) -> None:
        """
        An initializer which allows initializing model parameters in "blocks". This is helpful
        in the case of recurrent models which use multiple gates applied to linear projections,
        which can be computed efficiently if they are concatenated together. However, they are
        separate parameters which should be initialized independently.
        Parameters
        ----------
        tensor : ``torch.Tensor``, required.
            A tensor to initialize.
        split_sizes : List[int], required.
            A list of length ``tensor.ndim()`` specifying the size of the
            blocks along that particular dimension. E.g. ``[10, 20]`` would
            result in the tensor being split into chunks of size 10 along the
            first dimension and 20 along the second.
        gain : float, optional (default = 1.0)
            The gain (scaling) applied to the orthogonal initialization.
        """

        if isinstance(tensor, Variable):
        # in pytorch 4.0, Variable equals Tensor
        #    block_orthogonal(tensor.data, split_sizes, gain)
        #else:
            sizes = list(tensor.size())
            if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
                raise ConfigurationError("tensor dimensions must be divisible by their respective "
                                         "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
            indexes = [list(range(0, max_size, split))
                       for max_size, split in zip(sizes, split_sizes)]
            # Iterate over all possible blocks within the tensor.
            for block_start_indices in itertools.product(*indexes):
                # A list of tuples containing the index to start at for this block
                # and the appropriate step size (i.e split_size[i] for dimension i).
                index_and_step_tuples = zip(block_start_indices, split_sizes)
                # This is a tuple of slices corresponding to:
                # tensor[index: index + step_size, ...]. This is
                # required because we could have an arbitrary number
                # of dimensions. The actual slices we need are the
                # start_index: start_index + step for each dimension in the tensor.
                block_slice = tuple([slice(start_index, start_index + step)
                                     for start_index, step in index_and_step_tuples])
                tensor[block_slice] = torch.nn.init.orthogonal_(tensor[block_slice].contiguous(), gain=gain)
