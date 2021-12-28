# scipy.linalg.blas.sgemm
import scipy.linalg

def _gram(self, layer):
    """
    Compute gram matrix; just the dot product of the layer and its
    transform
    """
    gram = blas.sgemm(1.0, layer, layer.T)
    return gram
