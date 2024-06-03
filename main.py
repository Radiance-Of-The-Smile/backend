from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64, os
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf # type: ignore
from ultralytics import YOLO # type: ignore
from tensorflow.keras.models import load_model # type: ignore

app = FastAPI()

# Add CORS middleware to allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins if needed
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Reloading Saved Models
model = YOLO("models/quadenum.pt") # object detection for quadrant and enumeration
cnn_model = load_model("models/densenet121.h5") # disease classifier

def preprocess_image(image: Image.Image, path: str) -> tuple[Image.Image, list[str]]:

  results = model(image)  # Make Bounding Boxes for all Teeth

  plt.figure(figsize=(12, 8))
  plt.axis('off')
  plt.imshow(image)

  anomalies = []
  count = 0
  for i in results[0].boxes:
    sum = int(i.cls)
    quad = sum // 8
    label = 'Q=' + str(quad) + '\nE=' + str((sum - quad * 8)) + '\nD='

    x_min, y_min, x_max, y_max = i.xyxy[0]
    cropped_image = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

    # Resize the cropped image to 128x128
    cropped_image = cropped_image.resize((128, 128))

    # Convert the cropped image to a numpy array
    img_array = np.array(cropped_image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values

    # Make predictions
    predictions = cnn_model.predict(img_array) # Predict disease using DenseNet
    top_class = np.argsort(predictions[0])[::-1][0]  # Get top predicted classes

    if top_class == 0: continue
    elif top_class == 1: label += 'caries'
    elif top_class == 2: label += 'deep-caries'
    elif top_class == 3: label += 'impacted'
    else: label += 'periapical-lesion'

    anomalies.extend([label.replace('\n', ', ') + '\n'])

    # Plot the bounding box
    plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min])

    # Display the box number
    plt.text(x_max, y_max, label, fontsize=8, ha='right', va='bottom')

    count += 1

  plt.savefig(path) # Saving Processed Image
  image = Image.open(path)
  os.remove(path) # Free Memory
  return image, anomalies

@app.get("/")
async def health_check():
    return "The health check is successful!"

@app.post("/preprocess")
async def preprocess_image_endpoint(image: UploadFile = File(...)):

    # Read image content
    content = await image.read()

    # Load image from bytes
    image_obj = Image.open(BytesIO(content))

    filename, file_extension = os.path.splitext(image.filename)
    # image_obj.save('static/'+filename+file_extension) # Saving Uploaded Image
    processed_image_path = f"{filename}_processed{file_extension}"
    
    # Preprocess image
    preprocessed_image, anomalies = preprocess_image(image_obj, processed_image_path)

    # Convert preprocessed image back to bytes for response
    preprocessed_bytes = BytesIO()
    preprocessed_image.save(preprocessed_bytes, format="PNG")
    preprocessed_bytes_array = preprocessed_bytes.getvalue()

    # Encode preprocessed image to base64 string
    preprocessed_bytes_base64 = base64.b64encode(preprocessed_bytes_array).decode("utf-8")

    # # Restoring Processed Image from Bytes (Have to do this in frontend)
    # decoded_data = base64.b64decode(preprocessed_bytes_base64)
    # image = Image.open(BytesIO(decoded_data))
    # image.show()

    return {
        "processed_image": preprocessed_bytes_base64,
        "anomalies": anomalies
    }
