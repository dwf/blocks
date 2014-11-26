from abc import ABCMeta, abstractmethod

import six

from blocks.datasets.iteration import (FiniteSubsetIterator,
                                       InfiniteSubsetIterator)


class Dataset(six.Iterator):
    """A dataset.

    A dataset is responsible for loading data from e.g. a file, and
    allowing the user to iterate over it for training purposes.

    """
    __metaclass__ = ABCMeta
    supported_subset_iterators = tuple()
    unsupported_subset_iterators = tuple()

    def __iter__(self):
        if hasattr(self, 'subset_iterator'):
            # Reset the iterator -> reset the subset iterator
            self.subset_iterator = iter(self.subset_iterator)
        return self

    @abstractmethod
    def __next__(self):
        pass

    def set_subset_iterator(self, subset_iterator):
        """Set the subset iterator.

        Also checks whether the subset iterator used is supported or not.

        """
        if (not isinstance(subset_iterator,
                           self.supported_subset_iterators) or
            isinstance(subset_iterator,
                       self.unsupported_subset_iterators)):
            raise ValueError("Dataset does not support this subset iterator")
        self.subset_iterator = subset_iterator
        return self


class InfiniteDataset(Dataset):
    """Datasets which are infinite e.g. a probability distribution"""
    supported_subset_iterators = InfiniteSubsetIterator


class FiniteDataset(Dataset):
    """Datasets where the number of examples is known."""
    supported_subset_iterators = FiniteSubsetIterator
