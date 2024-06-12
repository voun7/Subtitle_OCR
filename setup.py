from setuptools import setup

setup(
    name='Subtitle_OCR',
    version='1.0',
    packages=[
        'models.detection', 'models.detection.db', 'models.detection.db.backbones', 'models.recognition',
        'models.recognition.alphabets', 'utilities'
    ],
    py_modules=['subtitle_ocr'],
    include_package_data=True,
    package_data={"models.recognition": ["**/*.txt"]},
    install_requires=['opencv-python', 'shapely', 'pyclipper'],
    url='https://github.com/voun-github/Subtitle_OCR',
    license='',
    author='Victor N',
    author_email='nwaeze_victor@yahoo.com',
    description='Using deep learning with PyTorch for a specialized subtitle text recognition.'
)
