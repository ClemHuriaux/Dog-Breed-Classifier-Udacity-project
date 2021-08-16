from django.db import models


class Image(models.Model):
    Image = models.ImageField(upload_to='images/')
