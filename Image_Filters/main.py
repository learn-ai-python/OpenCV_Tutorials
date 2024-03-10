import cv2
import numpy as np


def black_and_white(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def negative(image):
    return cv2.bitwise_not(image)


def sepia(image):
    image = np.array(image, dtype=np.float64)

    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])

    sepia_image = cv2.transform(image, sepia_filter)

    sepia_image[np.where(sepia_image > 255)] = 255

    return sepia_image.astype(np.uint8)


def sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)
    return sobel_combined


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(gray, 100, 200)
    return canny_edges


def cartoon(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


def pencil(image):
    _, pencil_image = cv2.pencilSketch(image, sigma_s=50, sigma_r=0.09, shade_factor=0.03)
    return pencil_image


def emboss(image):
    kernel = np.array([[0, -1, -1],
                       [1, 0, -1],
                       [1, 1, 0]])
    embossed_image = cv2.filter2D(image, -1, kernel)
    return embossed_image


def duo_tone(image, channel):
    duotone_image = image.copy()
    for i in range(3):
        if i != channel:
            duotone_image[:, :, i] = 0
    return duotone_image


if __name__ == "__main__":
    image = cv2.imread('tree.jpg')

    # Apply filters
    bw_image = black_and_white(image)
    negative_image = negative(image)
    sepia_image = sepia(image)
    sobel_image = sobel(image)
    canny_image = canny(image)
    cartoon_image = cartoon(image)
    pencil_image = pencil(image)
    emboss_image = emboss(image)
    duo_tone_image0 = duo_tone(image, 0)
    duo_tone_image1 = duo_tone(image, 1)
    duo_tone_image2 = duo_tone(image, 2)

    # Display the results
    cv2.imshow('Black and White', bw_image)
    cv2.imshow('Negative', negative_image)
    cv2.imshow('Sepia', sepia_image)
    cv2.imshow('Sobel', sobel_image)
    cv2.imshow('Canny', canny_image)
    cv2.imshow('Cartoon', cartoon_image)
    cv2.imshow('Pencil', pencil_image)
    cv2.imshow('Emboss', emboss_image)
    cv2.imshow('Duo Tone (B)', duo_tone_image0)
    cv2.imshow('Duo Tone (G)', duo_tone_image1)
    cv2.imshow('Duo Tone (R)', duo_tone_image2)

    # Save images
    cv2.imwrite('Black_and_White.jpg', bw_image)
    cv2.imwrite('Negative.jpg', negative_image)
    cv2.imwrite('Sepia.jpg', sepia_image)
    cv2.imwrite('Sobel.jpg', sobel_image)
    cv2.imwrite('Canny.jpg', canny_image)
    cv2.imwrite('Cartoon.jpg', cartoon_image)
    cv2.imwrite('Pencil.jpg', pencil_image)
    cv2.imwrite('Emboss.jpg', emboss_image)
    cv2.imwrite('Duo_Tone_B.jpg', duo_tone_image0)
    cv2.imwrite('Duo_Tone_G.jpg', duo_tone_image1)
    cv2.imwrite('Duo_Tone_R.jpg', duo_tone_image2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
