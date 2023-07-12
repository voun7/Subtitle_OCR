# A file dedicated to helpful utility functions.
import cv2


def draw_bounding_boxes(image_path: str, bbox_path: str) -> None:
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


if __name__ == '__main__':
    image_file = 'training_data/chinese_data/trdg_synthetic_images/0.jpg'
    bbox_file = 'training_data/chinese_data/trdg_synthetic_images/0_boxes.txt'
    draw_bounding_boxes(image_file, bbox_file)
