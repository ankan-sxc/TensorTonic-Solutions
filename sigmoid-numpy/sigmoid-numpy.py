import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid activation function.

    Parameters
    ----------
    x : scalar, list, or nested list

    Returns
    -------
    float or np.ndarray
        Float for scalar input, NumPy array otherwise.
    """
    if np.isscalar(x):
        return float(1.0 / (1.0 + np.exp(-x)))

    x = np.array(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


# -----------------------
# Tests
# -----------------------

# Example tests
assert np.allclose(sigmoid([0, 2, -2]),
                   [0.5, 0.88079708, 0.11920292])

assert sigmoid(0) == 0.5

assert np.allclose(
    sigmoid([[-1, 0], [1, 2]]),
    [[0.26894142, 0.5],
     [0.73105858, 0.88079708]]
)

# Scalar edge cases
assert isinstance(sigmoid(0), float)
assert sigmoid(100) > 0.999999999999
assert sigmoid(-100) < 1e-12

# Float input
assert np.isclose(sigmoid(0.5), 0.62245933)

# Empty list
result = sigmoid([])
assert isinstance(result, np.ndarray)
assert result.shape == (0,)

# Nested empty list
result = sigmoid([[]])
assert result.shape == (1, 0)

# Preserve shape
x = [[1, 2, 3], [4, 5, 6]]
y = sigmoid(x)
assert y.shape == (2, 3)

# Integer inputs become float outputs
y = sigmoid([1, 2, 3])
assert y.dtype == float

# NumPy array input
x = np.array([-1, 0, 1])
assert np.allclose(
    sigmoid(x),
    [0.26894142, 0.5, 0.73105858]
)

# Large positive/negative values
assert np.isclose(sigmoid(1000), 1.0)

# For very negative values, NumPy may emit an overflow warning,
# but the mathematical result is still effectively 0.
assert np.isclose(sigmoid(-1000), 0.0)

print("All tests passed!")
