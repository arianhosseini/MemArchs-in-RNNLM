import numpy as np

def sample_unit_simplex(shape, axis=1, sampling_function=np.random.rand,
                        dtype=np.float32):
    shape = np.asarray(shape, dtype=np.int32)
    shape[0], shape[axis] = shape[axis], shape[0]
    raw_values = sampling_function(*shape).astype(dtype)
    normalized_values = np.swapaxes(raw_values / np.sum(raw_values, axis=0),
                                    0, axis)

    return normalized_values.copy()
