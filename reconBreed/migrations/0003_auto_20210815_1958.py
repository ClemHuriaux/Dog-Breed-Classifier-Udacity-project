# Generated by Django 2.2.5 on 2021-08-15 17:58

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('reconBreed', '0002_remove_dog_name'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Dog',
            new_name='Image',
        ),
        migrations.RenameField(
            model_name='image',
            old_name='Dog_Image',
            new_name='Image',
        ),
    ]
