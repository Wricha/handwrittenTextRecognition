from django.shortcuts import render

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import io

# Load your CNN model
model_path = "C:\\Users\\DELL\\prescriptionRecognition\\handwritten_text_50.keras" # Update with your CNN model path
cnn_model = load_model(model_path)

# Define a function to preprocess the image
def preprocess_image(image):
    """
    Preprocess the uploaded image to match the input requirements of the CNN model.
    """
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((32,128))  # Resize to match model input
    
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)
    
    image_array = np.expand_dims(image_array, axis=0)  # Add batch and channel dimensions
    return image_array

# Define a function to decode the CNN model predictions
def decode_predictions(predictions):
    """
    Convert the model's output into readable text.
    The decoding logic depends on your specific model setup.
    For example, if your model uses an index-to-character mapping, implement it here.
    """
    # Placeholder: Replace with actual decoding logic
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    max_index = len(characters) - 1
    
    # Convert predictions to readable text
    output = ""
    for pred in np.argmax(predictions, axis=2)[0]:  # Assuming predictions shape (1, seq_len, num_classes)
        if pred <= max_index:  # Check to avoid out-of-range index
            output += characters[pred]
        else:
            output += "?"

    return output

# Django view for image upload
@csrf_exempt
def upload_prescription(request):
    if request.method == "POST" and request.FILES.get("file"):
        try:
            # Get the uploaded image
            uploaded_file = request.FILES["file"]
            image = Image.open(io.BytesIO(uploaded_file.read()))

            # Preprocess the image for the model
            preprocessed_image = preprocess_image(image)

            # Predict using the CNN model
            predictions = cnn_model.predict(preprocessed_image)

            # Decode predictions to text
            extracted_text = decode_predictions(predictions)

            return JsonResponse({"extracted_text": extracted_text})

        except Exception as e:
            return JsonResponse({"error": str(e)})
    #else:
    #    return JsonResponse({"error": "Invalid request method or no file uploaded."})
    
    return render(request, "core/index.html")