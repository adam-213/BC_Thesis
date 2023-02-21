return (image - mean[:, None, None]) / std[:, None, None]
RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 0
 - 