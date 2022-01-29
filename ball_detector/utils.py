import cv2


def get_focal_length(distance_between, real_width, width_in_image):  # Borrowed from Sami
    """
    Finds focal length of camera given the dimensions of a reference object in real life and an image.
    @distance_between: distance between camera and reference object (cm)
    @real_width: real-world width of reference object (cm)
    @width_in_image: width of the object in the image (px)
    """
    return (width_in_image * distance_between)/real_width


def distance_to_object(focal_length, real_width, width_in_image):
    """
    Finds distance to image
    @focal_length: focal length of the camera (see get_focal_length())
    @real_width: real-world width of reference object (cm)
    @width_in_image: width of the object in the image (px)
    """
    return (real_width * focal_length)/width_in_image


def draw_text_box(img, text, pos, font_kwargs):
    text_size, _ = cv2.getTextSize(text, thickness=1, **font_kwargs)
    cv2.rectangle(img, pos, (pos[0] + text_size[0], pos[1] - text_size[1]), (0, 0, 0), thickness=cv2.FILLED)
    cv2.putText(img, text, pos, color=(255, 255, 255), **font_kwargs)
