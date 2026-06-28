"""Inference layer: load a trained model and diagnose a recording."""
from cardiag.inference.classifier import Classifier
from cardiag.inference.triage import TriageClassifier

__all__ = ["Classifier", "TriageClassifier"]
