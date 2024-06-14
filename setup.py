from setuptools import setup

setup(
    name='subtitle_ocr',
    version='1.0',
    packages=[
        'sub_ocr', 'sub_ocr.models', 'sub_ocr.models.detection', 'sub_ocr.models.detection.db',
        'sub_ocr.models.detection.db.backbones', 'sub_ocr.models.recognition', 'sub_ocr.models.recognition.alphabets',
        'sub_ocr.utilities'
    ],
    include_package_data=True,
    package_data={"sub_ocr.models.recognition": ["**/*.txt"]},
    install_requires=['torchvision', 'opencv-python', 'shapely', 'pyclipper'],
    url='https://github.com/voun-github/Subtitle_OCR',
    license='',
    author='Victor N',
    author_email='nwaeze_victor@yahoo.com',
    description='Using deep learning with PyTorch for a specialized subtitle text detection and recognition.'
)
