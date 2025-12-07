"""
Пакет с классификаторами для классификации изображений достопримечательностей
"""

from .base_classifier import BaseClassifier
from .bow_classifier import BOWClassifier
from .vit_classifier import ViTClassifier

__all__ = ['BaseClassifier', 'BOWClassifier', 'ViTClassifier']

