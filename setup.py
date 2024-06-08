from setuptools import setup

setup(
    name='Subtitle_OCR',
    version='1.0',
    packages=[
        'models.detection.db', 'models.detection.db.backbones', 'models.recognition', 'models.recognition.crnn',
        'utilities'
    ],
    py_modules=['subtitle_ocr'],
    include_package_data=True,
    install_requires=['opencv-python', 'shapely', 'pyclipper'],
    url='https://github.com/voun-github/Subtitle_OCR',
    license='',
    author='Victor N',
    author_email='nwaeze_victor@yahoo.com',
    description='Using deep learning with PyTorch for a specialized subtitle text recognition.'
)

# run 'python setup.py sdist' to build
# todo: finish implementation of module and test distribution builds for errors.
