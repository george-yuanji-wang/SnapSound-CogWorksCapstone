from database import Database
from facenet_pytorch import MTCNN, InceptionResnetV1 
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
import numpy as np
import io
import json
import os


pipe = pipeline("image-classification", model="motheecreator/vit-Facial-Expression-Recognition")
streaming = False
threshold = 0.5
mtcnn = MTCNN(keep_all=True, post_process=False)
resnet = InceptionResnetV1(pretrained="vggface2").eval()

##################################################

### Basic   #################################
##################################################



def input(image, thresh):
    image = Image.open(image)
    image = image.convert("RGB")
    boxes, probabilities = mtcnn.detect(image)
    draw = ImageDraw.Draw(image)
    font_path = "arial.ttf"  # Path to a .ttf file
    font_size = 30  # Specify the font size
    font = ImageFont.truetype(font_path, font_size)

    max_index = int(list(probabilities).index(max(list(probabilities))))
    
    if boxes is not None:
        box = boxes[max_index]
        cropped_image = image.crop(box)
        image_resized = cropped_image.resize((224, 224))
        prediction = pipe(image_resized)
        if prediction[0]['label'] == "neutral":
            prediction_ = prediction[1]['label']
        else:
            prediction_ = prediction[0]['label']

        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="red", width=3)
        left, top, right, bottom = draw.textbbox((box[0], box[1] - 10), text = prediction_, font=font)
        draw.rectangle((left-5, top-5, right+5, bottom+5), fill="white")
        draw.text((box[0], box[1] - 10), text=prediction_, font=font, fill="black")

        return image, prediction_
    
    else:
        return "No Boxes"


##################################################

### UI + STYLE   #################################
##################################################




with gr.Blocks(theme='Taithrah/Minimal') as demo:
    with gr.Tabs():
        with gr.TabItem("Emotion Detect"):
            with gr.Row():
                input_image = gr.Image(type="filepath", label="Input Image")
                output_image = gr.Image(type="pil", label="Processed Image")

            prediction = gr.Textbox()
 
            threshold_slider = gr.Slider(minimum=0.1, maximum=1, value=0.35, step=0.05, label="Threshold value")
            submit_button_classify = gr.Button("Classify Emotion")
            if input_image is not None:
                submit_button_classify.click(input, inputs=[input_image, threshold_slider], outputs=[output_image, prediction])
            else:
                print("This will not be printed since 'value' exists")



demo.launch(share=True)


