# Borrowed from Keras

import tensorflow as tf
_GRAPH_LEARNING_PHASES = {}
_GRAPH_BN_LEARNING_PHASES = {}


def learning_phase():
    """Returns the learning phase flag.

    The learning phase flag is a bool tensor (0 = test, 1 = train)
    to be passed as input to any Keras function
    that uses a different behavior at train time and test time.

    # Returns
        Learning phase (scalar integer tensor or Python integer).
    """
    graph = tf.get_default_graph()
    if graph not in _GRAPH_LEARNING_PHASES:
        phase = tf.placeholder(dtype='bool',
                               name='tf_plus_learning_phase')
        _GRAPH_LEARNING_PHASES[graph] = phase
    return _GRAPH_LEARNING_PHASES[graph]

def batchnorm_learning_phase():
    """Returns the learning phase flag.

    The learning phase flag is a bool tensor (0 = test, 1 = train)
    to be passed as input to any Keras function
    that uses a different behavior at train time and test time.

    # Returns
        Learning phase (scalar integer tensor or Python integer).
    """
    graph = tf.get_default_graph()
    if graph not in _GRAPH_BN_LEARNING_PHASES:
        phase = tf.placeholder(dtype='bool',
                               name='tf_plus_learning_phase')
        _GRAPH_BN_LEARNING_PHASES[graph] = phase
    return _GRAPH_BN_LEARNING_PHASES[graph]
