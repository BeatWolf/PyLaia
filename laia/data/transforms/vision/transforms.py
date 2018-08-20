import torchvision
from PIL import ImageOps, Image


class Invert(object):
    """Invert the colors of a PIL image with the given probability."""

    def __call__(self, img):
        # type: (Image) -> Image
        return ImageOps.invert(img)

    def __repr__(self):
        return "vision.{}()".format(self.__class__.__name__)


class Convert(object):
    """Convert a PIL image to Greyscale, RGB or RGBA."""

    def __init__(self, mode):
        # type: (str) -> None
        assert mode in ("L", "RGB", "RGBA")
        self.mode = mode

    def __call__(self, img):
        # type: (Image) -> Image
        return img.convert(self.mode)

    def __repr__(self):
        format_string = "vision." + self.__class__.__name__ + "("
        if self.mode is not None:
            format_string += "mode={}".format(self.mode)
        format_string += ")"
        return format_string


class ToPreparedTensor(object):
    def __init__(
        self,
        invert=True,
        mode="L",
        fixed_height=None,
        fixed_width=None,
        min_height=None,
        min_width=None,
        pad_color=0,
    ):
        self._convert_transform = Convert(mode)
        self._invert_transform = Invert() if invert else lambda x: x
        self._resize_transform = (
            Resize((fixed_width, fixed_height))
            if fixed_width or fixed_height
            else lambda x: x
        )
        self._pad_transform = (
            Pad((min_width, min_height), fill=pad_color)
            if min_width or min_height
            else lambda x: x
        )
        self._tensor_transform = ToTensor()

    def __call__(self, img):
        assert isinstance(img, Image.Image)
        img = self._convert_transform(img)
        img = self._invert_transform(img)
        img = self._resize_transform(img)
        img = self._pad_transform(img)
        img = self._tensor_transform(img)
        return img


ToTensor = torchvision.transforms.transforms.ToTensor

Resize = torchvision.transforms.transforms.Resize

Pad = torchvision.transforms.transforms.Pad
