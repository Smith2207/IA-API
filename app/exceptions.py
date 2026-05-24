class PetNotDetectedError(Exception):
    """La imagen no contiene un perro o gato detectable."""


class ImageProcessingError(Exception):
    """Error genérico de procesamiento de imagen."""
