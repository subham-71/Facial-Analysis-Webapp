from django import forms
from app.models import FaceReconition

class FaceRecognitionform(forms.ModelForm):

    class Meta:
        model=FaceReconition
        fields=['image']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['image'].widget.attrs.update({'class':'form-control'})
        


