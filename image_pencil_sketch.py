import gradio as gr

import numpy as np
import cv2
from PIL import Image


def image_pencil_sketch(image):
   gray_image = Image.fromarray(image).convert("L")
   inverted_image = 255 - np.array(gray_image)
   blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
   inverted_blurred = 255 - blurred
   pencil_sketch = cv2.divide(np.array(gray_image), inverted_blurred, scale=256.0)
   return pencil_sketch

def process_image(image):
   results = [image]

   pencil_sketch = image_pencil_sketch(image)
   results.append(pencil_sketch)

   return results[1]

# Gradio Interface
iface = gr.Interface(
   title="Building an Image Process Engine",
   description="by Jeff",
   fn=process_image,
   inputs=gr.Image(label="Input Image"),
   # outputs=gr.Gallery(label="Result Images"),
   outputs=gr.Image(label="Result Images"),
)
iface.launch()
