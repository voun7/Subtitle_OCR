class DetMetric:
    # This metric will be implemented when an algorithms that requires it is added.
    def __init__(self, post_processor):
        self.post_processor = post_processor

    def __call__(self, predictions, batch, validation):
        pass
