import cv2 as cv


def rescale(scale: float, frame=None, bbox: tuple = None):
    """
    Method to rescale any image frame or bbox using scale.
    Bbox is returned as an integer. This function should be used only for visualization.
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


def get_scale_factor(img_frame, img_target_height: int = 700) -> float:
    img_height = img_frame.shape[0]
    if img_height > img_target_height:
        rescale_factor = img_target_height / img_height
        return rescale_factor


def visualize_data(image, bboxes: list[list | tuple] = None) -> None:
    if type(image) is str:
        image = cv.imread(image)  # Load the image
        if scale := get_scale_factor(image):
            image = rescale(scale, image)
        if scale and bboxes:
            bboxes = [rescale(scale, bbox=bbox) for bbox in bboxes]
        win_name = f"Image Rescale Value: {round(scale, 6) if scale else scale}"
    else:
        # Convert PyTorch tensor to NumPy array
        image = image.permute(1, 2, 0).numpy()  # Convert from CHW to HWC
        image = (image * 255).astype("uint8")  # Convert to 8-bit integer
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Change channel order from RGB to BGR
        win_name = "PyTorch tensor to Image"

    if bboxes is not None:
        for box in bboxes:  # Iterate over the lines and draw bounding boxes
            x_min, y_min, x_max, y_max = map(int, box)  # Change value type to int and parse values to variables.
            cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Draw the bounding box on the image

    # Display the image
    cv.imshow(win_name, image)
    cv.waitKey()
    cv.destroyAllWindows()
