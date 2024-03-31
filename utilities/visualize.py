import cv2 as cv


def rescale(scale: float, frame=None, bbox: tuple = None):
    """
    Method to rescale any image frame, bbox.
    """
    if frame is not None:
        width, height = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
        return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

    if bbox:
        x1, y1, x2, y2 = bbox
        x1 = x1 * scale
        y1 = y1 * scale
        x2 = x2 * scale
        y2 = y2 * scale
        return int(x1), int(y1), int(x2), int(y2)


def get_scale_factor(img_frame, img_target_height: int = 700):
    img_height = img_frame.shape[0]
    if img_height > img_target_height:
        rescale_factor = img_target_height / img_height
        return rescale_factor


def visualize_datasource(image_path: str, bboxes: list = None) -> None:
    img_frame = cv.imread(image_path)  # Load the image

    if scale := get_scale_factor(img_frame):
        img_frame = rescale(scale, img_frame)
    if scale and bboxes:
        bboxes = [rescale(scale, bbox=bbox) for bbox in bboxes]

    if bboxes is not None:
        for box in bboxes:  # Iterate over the lines and draw bounding boxes
            x1, y1, x2, y2 = map(int, box)  # Change value type from string to int and parse values to variables.
            cv.rectangle(img_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the bounding box on the image

    # Display the image with bounding boxes
    cv.imshow(f"Image Rescale Value: {round(scale, 6) if scale else scale}", img_frame)
    cv.waitKey()
    cv.destroyAllWindows()
