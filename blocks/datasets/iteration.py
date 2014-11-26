import itertools

import six


class SubsetIterator(six.Iterator):
    """A subset iterator.

    Subset iterators are there to support different ways of iterating over
    datasets. A subset iterator provides a dataset-agnostic iteration
    scheme, such as sequential batches, shuffled batches, etc.

    """
    def __iter__(self):
        return self


class InfiniteSubsetIterator(SubsetIterator):
    """Iterators that return a batch size.

    For infinite datasets it doesn't make sense to provide indices to
    examples, but the number of samples per batch can still be given.
    Hence InfiniteSubsetIterator is the base class for subset iterators
    that only provide the number of examples that should be in a batch.

    """
    pass


class FiniteSubsetIterator(SubsetIterator):
    """Iterators that return slices or indices.

    For datasets where the number of examples is known and easily
    accessible (as is the case for most datasets which are small enough
    to be kept in memory, like MNIST) we can provide slices or lists of
    labels to the dataset.

    """
    pass


class ConstantIterator(InfiniteSubsetIterator):
    """Constant batch size iterator.

    This subset iterator simply returns the same constant batch size
    for a given number of times (or else infinitely).

    """
    def __init__(self, batch_size, times=None):
        self.batch_size = batch_size
        self.times = times

    def __iter__(self):
        if self.times is None:
            self.iterator = itertools.repeat(self.batch_size)
        else:
            self.iterator = itertools.repeat(self.batch_size, self.times)
        return self

    def __next__(self):
        return six.next(self.iterator)


class SequentialSubsetIterator(FiniteSubsetIterator):
    """Sequential batches iterator.

    Iterate over the examples sequentially in batches of a given size.

    Notes
    -----
    The batch size isn't enforced, so the last batch could be smaller.

    """
    def __init__(self, num_examples, batch_size):
        self.num_examples = num_examples
        self.batch_size = batch_size

    def __iter__(self):
        self.iterator = (slice(x, min(self.num_examples, x + self.batch_size))
                         for x in range(0, self.num_examples, self.batch_size))
        return self

    def __next__(self):
        return six.next(self.iterator)
