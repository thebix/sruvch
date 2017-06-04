class SingleLayerPerceptron(object):
    """A single layer perceptron. Have the following properties:

    Attributes:
        w: A float representing the w.
        b: A float representing the b.
    """

    def __init__(self, w, b=0.0):
        """Return a SingleLayerPerceptron object whose w is *w* and b is *b*."""
        self.w = w
        self.b = b

    def props():
        """Return the props."""
        return self.w, self.b
