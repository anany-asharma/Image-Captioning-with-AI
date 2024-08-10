from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
from PIL import Image 


# initializing the processor and model from hugging face
processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# creating model
def create_model(image):
    inputs=processor(image, return_tensors="pt")
    outputs=model.generate(**inputs)
    caption=processor.decode(outputs[0],skip_special_tokens=True,max_new_tokens=20)
    return caption
def generate_caption(image):
    try:
        caption=create_model(image)
        return caption
    except Exception as e:
        return "An error occurred"
    

# creating interface
iface=gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image captioning with BLIP",
    description="Upload an image to generate a caption"
)
iface.launch()