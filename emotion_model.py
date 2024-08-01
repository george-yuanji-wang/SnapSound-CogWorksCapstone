from transformers import pipeline
from torchvision import transforms
from PIL import Image

pipe = pipeline("image-classification", model="motheecreator/vit-Facial-Expression-Recognition")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the model's expected input size
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize pixel values
])

image = Image.open("sad.jpg")
image_resized = image.resize((224, 224))

#image = preprocess(image).unsqueeze(0) 


# Perform the classification
results = pipe(image)
print(results)

