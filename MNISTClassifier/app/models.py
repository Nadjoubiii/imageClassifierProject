from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='images/')
    predicted_digit = models.CharField(max_length=2, blank=True)
