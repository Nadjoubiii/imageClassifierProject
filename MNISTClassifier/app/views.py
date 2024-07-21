import base64
from io import BytesIO
from .models import UploadedImage
from .serializers import UploadedImageSerializer
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import io
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


# class ImageUploadView(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def post(self, request, *args, **kwargs):
#         file_serializer = UploadedImageSerializer(data=request.data)
#         if file_serializer.is_valid():
#             file_serializer.save()

#             # Load the image
#             image = Image.open(file_serializer.instance.image)
#             image = image.resize((28, 28)).convert('L')
#             image = np.array(image)
#             image = image / 255.0
#             image = image.reshape.flatten(1, 784)

#             # Load the model and predict
#             model = load_model('app/classifier.h5')
#             prediction = model.predict(image)
#             predicted_digit = str(np.argmax(prediction))

#             # Update the predicted digit in the database
#             file_serializer.instance.predicted_digit = predicted_digit
#             file_serializer.instance.save()

#             return Response(file_serializer.data, status=201)
#         else:
#             return Response(file_serializer.errors, status=400)

model = load_model('app/classifier.h5')
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        
        # Original Image
        original_img = Image.open(image).convert('L')
        
        # Preprocess the image
        img = original_img.resize((28, 28))
        img = ImageOps.invert(img)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(2)
        img_array = np.array(img).astype('float32') / 255.0
        img_array = img_array.flatten().reshape(1, 784)
        
        # Predict the digit
        predicted_digit = np.argmax(model.predict(img_array))
        
        # Convert images to base64 for display
        buffered_original = BytesIO()
        original_img.save(buffered_original, format="PNG")
        original_img_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        buffered_processed = BytesIO()
        img.save(buffered_processed, format="PNG")
        processed_img_base64 = base64.b64encode(buffered_processed.getvalue()).decode()
        
        return JsonResponse({
            'predicted_digit': int(predicted_digit),
            'original_image': f'data:image/png;base64,{original_img_base64}',
            'processed_image': f'data:image/png;base64,{processed_img_base64}'
        })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)