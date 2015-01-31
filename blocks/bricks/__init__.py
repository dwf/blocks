"""The interface of bricks and some simple implementations."""
import logging
from itertools import chain
from abc import abstractproperty

import numpy
from six import add_metaclass
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams

from blocks import config
from blocks.bricks.base import application, _Brick, Brick, lazy
from blocks.graph import add_role, ParameterRole
from blocks.utils import pack, shared_floatx_zeros

logger = logging.getLogger(__name__)


class WeightsRole(ParameterRole):
    pass

WEIGHTS = WeightsRole()


class BiasesRole(ParameterRole):
    pass

BIASES = BiasesRole()


class Random(Brick):
    """A mixin class for Bricks which need Theano RNGs.

    Parameters
    ----------
    theano_rng : object
        A ``MRG_RandomStreams`` instance.

    """
    seed_rng = numpy.random.RandomState(config.default_seed)

    def __init__(self, theano_seed=None, **kwargs):
        super(Random, self).__init__(**kwargs)
        self.theano_seed = theano_seed

    @property
    def theano_seed(self):
        if getattr(self, '_theano_seed', None) is not None:
            return self._theano_seed
        else:
            self._theano_seed = self.seed_rng.randint(
                numpy.iinfo(numpy.int32).max)
            return self._theano_seed

    @theano_seed.setter
    def theano_seed(self, value):
        self._theano_seed = value

    @property
    def theano_rng(self):
        """Returns Brick's Theano RNG, or a default one.

        The default seed can be set through ``blocks.config``.

        """
        if getattr(self, '_theano_rng', None) is not None:
            return self._theano_rng
        else:
            return MRG_RandomStreams(self.theano_seed)

    @theano_rng.setter
    def theano_rng(self, theano_rng):
        self._theano_rng = theano_rng


class Initializable(Brick):
    """Base class for bricks which push parameter initialization.

    Many bricks will initialize children which perform a linear
    transformation, often with biases. This brick allows the weights
    and biases initialization to be configured in the parent brick and
    pushed down the hierarchy.

    Parameters
    ----------
    weights_init : object
        A `NdarrayInitialization` instance which will be used by to
        initialize the weight matrix. Required by :meth:`initialize`.
    biases_init : object, optional
        A `NdarrayInitialization` instance that will be used to initialize
        the biases. Required by :meth:`initialize` when `use_bias` is
        `True`. Only supported by bricks for which :attr:`has_biases` is
        ``True``.
    use_bias : bool, optional
        Whether to use a bias. Defaults to `True`. Required by
        :meth:`initialize`. Only supported by bricks for which
        :attr:`has_biases` is ``True``.
    rng : object
        A ``numpy.RandomState`` instance.

    Attributes
    ----------
    has_biases : bool
        ``False`` if the brick does not support biases, and only has
        :attr:`weights_init`.  For an example of this, see
        :class:`Bidirectional`. If this is ``False``, the brick does not
        support the arguments ``biases_init`` or ``use_bias``.

    """
    has_biases = True
    seed_rng = numpy.random.RandomState(config.default_seed)

    @lazy
    def __init__(self, weights_init, biases_init=None, use_bias=True,
                 seed=None, **kwargs):
        super(Initializable, self).__init__(**kwargs)
        self.weights_init = weights_init
        if self.has_biases:
            self.biases_init = biases_init
        elif biases_init is not None or not use_bias:
            raise ValueError("This brick does not support biases config")
        self.use_bias = use_bias
        self.seed = seed

    @property
    def seed(self):
        if getattr(self, '_seed', None) is not None:
            return self._seed
        else:
            self._seed = self.seed_rng.randint(
                numpy.iinfo(numpy.int32).max)
            return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

    @property
    def rng(self):
        if getattr(self, '_rng', None) is not None:
            return self._rng
        else:
            return numpy.random.RandomState(self.seed)

    @rng.setter
    def rng(self, rng):
        self._rng = rng

    def _push_initialization_config(self):
        for child in self.children:
            if isinstance(child, Initializable):
                child.rng = self.rng
                if self.weights_init:
                    child.weights_init = self.weights_init
        if hasattr(self, 'biases_init') and self.biases_init:
            for child in self.children:
                if (isinstance(child, Initializable) and
                        hasattr(child, 'biases_init')):
                    child.biases_init = self.biases_init


class Feedforward(Brick):
    """Declares an interface for bricks with one input and one output.

    Many bricks have just one input and just one output (activations,
    :class:`Linear`, :class:`MLP`). To make such bricks interchangable
    in most contexts they should share an interface for configuring
    their input and output dimensions. This brick provides such an
    interface.

    """

    @abstractproperty
    def input_dim():
        """The input dimension of the brick."""
        pass

    @abstractproperty
    def output_dim():
        """The output dimension of the brick."""
        pass


def override_with_attribute(property_, attribute=None):
    """Make a property behave like an attribute.

    Parameters
    ----------
    property_ : property
        The property to be overriden.
    attribute : str, optional
        The attribute to store the actual value. If ``None``,
        the property's name prepended with an underscore is used.

    """

    if not attribute:
        attribute = "_" + property_.fget.__name__

    def getter(self):
        return getattr(self, attribute)

    def setter(self, value):
        setattr(self, attribute, value)

    def deleter(self):
        delattr(self, attribute)

    overrider = property(getter, setter, deleter)
    return overrider


class Linear(Initializable, Feedforward):
    r"""A linear transformation with optional bias.

    Linear brick which applies a linear (affine) transformation by
    multiplying the input with a weight matrix. Optionally a bias is added.

    Parameters
    ----------
    input_dim : int
        The dimension of the input. Required by :meth:`allocate`.
    output_dim : int
        The dimension of the output. Required by :meth:`allocate`.

    Notes
    -----
    See :class:`Initializable` for initialization parameters.

    A linear transformation with bias is a matrix multiplication followed
    by a vector summation.

    .. math:: f(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}

    """
    input_dim = override_with_attribute(Feedforward.input_dim)
    output_dim = override_with_attribute(Feedforward.output_dim)

    @lazy
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _allocate(self):
        W = shared_floatx_zeros((self.input_dim, self.output_dim), name='W')
        add_role(W, WEIGHTS)
        self.params.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        if self.use_bias:
            b = shared_floatx_zeros((self.output_dim,), name='b')
            add_role(b, BIASES)
            self.params.append(b)
            self.add_auxiliary_variable(b.norm(2), name='b_norm')

    def _initialize(self):
        if self.use_bias:
            W, b = self.params
            self.biases_init.initialize(b, self.rng)
        else:
            W, = self.params
        self.weights_init.initialize(W, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the linear transformation.

        Parameters
        ----------
        input_ : Theano variable
            The input on which to apply the transformation

        Returns
        -------
        output : Theano variable
            The transformed input plus optional bias

        """
        if self.use_bias:
            W, b = self.params
        else:
            W, = self.params
        output = tensor.dot(input_, W)
        if self.use_bias:
            output += b
        return output


class Maxout(Brick):
    """Maxout pooling transformation.

    A brick that does max pooling over groups of input units. If you use
    this code in a research project, please cite [GWFM13]_.

    .. [GWFM13] Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
       Courville, and Yoshua Bengio, *Maxout networks*, ICML (2013), pp.
       1319-1327.

    Parameters
    ----------
    num_pieces : int
        The size of the groups the maximum is taken over.

    Notes
    -----
    Maxout applies a set of linear transformations to a vector and selects
    for each output dimension the result with the highest value.

    """
    @lazy
    def __init__(self, num_pieces, **kwargs):
        super(Maxout, self).__init__(**kwargs)
        self.num_pieces = num_pieces

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the maxout transformation.

        Parameters
        ----------
        input_ : Theano variable
            The input on which to apply the transformation

        Returns
        -------
        output : Theano variable
            The transformed input

        """
        last_dim = input_.shape[-1]
        output_dim = last_dim // self.num_pieces
        new_shape = ([input_.shape[i] for i in range(input_.ndim - 1)]
                     + [output_dim, self.num_pieces])
        output = tensor.max(input_.reshape(new_shape, ndim=input_.ndim + 1),
                            axis=input_.ndim)
        return output


class LinearMaxout(Initializable):
    """Maxout pooling following a linear transformation.

    This code combines the :class:`Linear` brick with a :class:`Maxout`
    brick.

    Parameters
    ----------
    input_dim : int
        The dimension of the input. Required by :meth:`allocate`.
    output_dim : int
        The dimension of the output. Required by :meth:`allocate`.
    num_pieces : int
        The number of linear functions. Required by :meth:`allocate`.

    Notes
    -----
    See :class:`Initializable` for initialization parameters.

    .. todo:: Name of :attr:`linear_transformation` shouldn't be hardcoded.

    """
    @lazy
    def __init__(self, input_dim, output_dim, num_pieces, **kwargs):
        super(LinearMaxout, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_pieces = num_pieces

        self.linear_transformation = Linear(
            name=self.name + '_linear_to_maxout', input_dim=input_dim,
            output_dim=output_dim * num_pieces, weights_init=self.weights_init,
            biases_init=self.biases_init, use_bias=self.use_bias)
        self.maxout_transformation = Maxout(name=self.name + '_maxout',
                                            num_pieces=num_pieces)
        self.children = [self.linear_transformation,
                         self.maxout_transformation]

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the linear transformation followed by maxout.

        Parameters
        ----------
        input_ : Theano variable
            The input on which to apply the transformations

        Returns
        -------
        output : Theano variable
            The transformed input

        """
        pre_activation = self.linear_transformation.apply(input_)
        output = self.maxout_transformation.apply(pre_activation)
        return output


class ActivationDocumentation(_Brick):
    """Dynamically adds documentation to activations.

    Notes
    -----
    See http://bugs.python.org/issue12773.

    """
    def __new__(cls, name, bases, classdict):
        classdict['__doc__'] = \
            """Elementwise application of {0} function.""".format(name.lower())
        if 'apply' in classdict:
            classdict['apply'].__doc__ = \
                """Apply the {0} function elementwise.

                Parameters
                ----------
                input_ : Theano variable
                    Theano variable to apply {0} to, elementwise.

                Returns
                -------
                output : Theano variable
                    The input with the activation function applied.

                """.format(name.lower())
        return super(ActivationDocumentation, cls).__new__(cls, name, bases,
                                                           classdict)


@add_metaclass(ActivationDocumentation)
class Activation(Brick):
    """A base class for simple, elementwise activation functions.

    This base class ensures that activation functions are automatically
    documented using the :class:`ActivationDocumentation` metaclass.

    """
    pass


class Identity(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return input_


class Tanh(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.tanh(input_)


class Sigmoid(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.nnet.sigmoid(input_)


class Rectifier(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.switch(input_ > 0, input_, 0)


class Softmax(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.nnet.softmax(input_)


class Sequence(Brick):
    """A sequence of bricks.

    This brick applies a sequence of bricks, assuming that their in- and
    outputs are compatible.

    Parameters
    ----------
    application_methods : list of application methods to apply

    """
    def __init__(self, application_methods, **kwargs):
        super(Sequence, self).__init__(**kwargs)
        self.application_methods = application_methods

        seen = set()
        self.children = [app.brick for app in application_methods
                         if not (app.brick in seen or seen.add(app.brick))]

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        child_input = input_
        for _, application_method in zip(self.children,
                                         self.application_methods):
            output = application_method(*pack(child_input))
            child_input = output
        return output


class MLP(Sequence, Initializable, Feedforward):
    """A simple multi-layer perceptron.

    Parameters
    ----------
    activations : bricks or ``None``
        A list of activations to apply after each linear transformation.
        Give ``None`` to not apply any activation. Required for
        :meth:`__init__`.
    dims : list of ints
        A list of input dimensions, as well as the output dimension of the
        last layer. Required for :meth:`allocate`.

    Notes
    -----
    See :class:`Initializable` for initialization parameters.

    Note that the ``weights_init``, ``biases_init`` and ``use_bias``
    configurations will overwrite those of the layers each time the
    :class:`MLP` is re-initialized. For more fine-grained control, push the
    configuration to the child layers manually before initialization.

    >>> from blocks.initialization import IsotropicGaussian, Constant
    >>> Brick.lazy = True
    >>> mlp = MLP(activations=[Tanh(), None], dims=[30, 20, 10],
    ...           weights_init=IsotropicGaussian(),
    ...           biases_init=Constant(1))
    >>> mlp.push_initialization_config()  # Configure children
    >>> mlp.children[0].weights_init = IsotropicGaussian(0.1)
    >>> mlp.initialize()

    """
    @lazy
    def __init__(self, activations, dims, **kwargs):
        self.activations = activations

        self.linear_transformations = [Linear(name='linear_{}'.format(i))
                                       for i in range(len(activations))]
        # Interleave the transformations and activations
        application_methods = [brick.apply for brick in list(chain(*zip(
            self.linear_transformations, activations))) if brick is not None]
        if not dims:
            dims = [None] * (len(activations) + 1)
        self.dims = dims
        super(MLP, self).__init__(application_methods, **kwargs)


    @property
    def input_dim(self):
        return self.dims[0]

    @input_dim.setter
    def input_dim(self, value):
        self.dims[0] = value

    @property
    def output_dim(self):
        return self.dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.dims[-1] = value

    def _push_allocation_config(self):
        if not len(self.dims) - 1 == len(self.linear_transformations):
            raise ValueError
        for input_dim, output_dim, layer in zip(self.dims[:-1], self.dims[1:],
                                                self.linear_transformations):
            layer.input_dim = input_dim
            layer.output_dim = output_dim
            layer.use_bias = self.use_bias
