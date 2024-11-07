from setuptools import setup

setup(
    name="subtitle_ocr",
    version="1.4",
    packages=['sub_ocr', 'sub_ocr.postprocess', "sub_ocr.alphabets"],
    include_package_data=True,
    package_data={"sub_ocr.alphabets": ["*.txt"]},
    install_requires=[
        "torch@https://download.pytorch.org/whl/cu124/"
        "torch-2.5.1%2Bcu124-cp312-cp312-win_amd64.whl ;platform_system=='Windows'",
        "torch;platform_system!='Windows'", "opencv-python", "shapely", "pyclipper", "requests"
    ],
    url="https://github.com/voun7/Subtitle_OCR",
    license="",
    author="Victor N",
    author_email="nwaeze_victor@yahoo.com",
    description="Using deep learning with PyTorch for a specialized subtitle text detection and recognition."
)
