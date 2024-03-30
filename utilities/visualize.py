import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def cv_draw_bboxes(image_path: str, bbox_path: str) -> None:
    image = cv2.imread(image_path)  # Load the image
    with open(bbox_path, 'r') as file:  # Read the text file containing bounding box coordinates
        lines = file.readlines()

    for line in lines:  # Iterate over the lines and draw bounding boxes
        line = line.split()
        x1, y1, x2, y2 = map(int, line)  # Change value type from string to int and parse values to variables.
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the bounding box on the image

    # Display the image with bounding boxes
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plt_draw_bboxes(img, bbs=None, labels=None):
    fig, ax = plt.subplots()
    ax.imshow(img)
    for bb, label in zip(bbs, labels):
        bbox = Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.annotate(label, (bb[0], bb[1]), color='red', weight='bold', fontsize=10)
        ax.add_patch(bbox)
    plt.show()


def visualize_datasource():
    pass
