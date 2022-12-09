import magic
import img2pdf
from django.core.files import File

from basic.models import DocumentType, Document


class ImageHandler:

    @staticmethod
    def handle_image(image_obj):
        type = ImageHandler.identify_type(image_obj.file)
        if type != image_obj.file.file.content_type:
            raise Exception()
        if 'image' in type:
            ImageHandler.convert_file_from_image(image_obj)
        return image_obj

    @staticmethod
    def classify_image(image_obj: Document) -> DocumentType:
        return DocumentType.Unknown

    @staticmethod
    def identify_type(image_obj):
        type = magic.from_buffer(image_obj.read(), mime=True)
        image_obj.seek(0)
        return type

    @staticmethod
    def convert_file_from_image(image_obj):
        dpix = dpiy = 300
        layout_fun = img2pdf.get_fixed_dpi_layout_fun((dpix, dpiy))
        pdf_as_bytes = img2pdf.convert(image_obj.file.read(), layout_fun=layout_fun)
        new_file = File(pdf_as_bytes, name=f'adsfasdf')
        image_obj.file = new_file
        print()
