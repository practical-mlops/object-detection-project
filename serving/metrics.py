from bentoml.metrics import Histogram

confidence_histogram = Histogram(
    name="confidence_score",
    documentation="The confidence score of the prediction",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
)
