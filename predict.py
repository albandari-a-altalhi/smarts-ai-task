import numpy as np
from pathlib import Path
from tf_keras.layers import DepthwiseConv2D
from tf_keras.models import load_model
from tf_keras.preprocessing import image


class CompatibleDepthwiseConv2D(DepthwiseConv2D):
    """Load models exported by older Keras versions."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


folder = Path(__file__).resolve().parent
model = load_model(
    folder / "keras_model.h5",
    compile=False,
    custom_objects={"DepthwiseConv2D": CompatibleDepthwiseConv2D},
)

class_names = ["Camels", "Lion", "monkey"]

image_path = input("Enter image path: ")

img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = (img_array / 127.5) - 1

predictions = model.predict(img_array, verbose=0)
index = np.argmax(predictions)
class_name = class_names[index]
confidence = predictions[0][index] * 100

print("Predicted class:", class_name)
print("Confidence:", str(round(confidence, 2)) + "%")