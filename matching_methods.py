import cv2

if __name__ == '__main__':

    # 0 = grayscale
    # template_image = cv2.imread('waldo.png', 0)
    # source_image = cv2.imread('find_waldo.jpg')

    template_image = cv2.imread('lettuce_stem.png', 0)
    source_image = cv2.imread('cam00.tiff')


    template_height, template_width = template_image.shape

    source_image_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

    matching_methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in matching_methods:
        img = source_image.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(source_image_gray, template_image, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

        cv2.rectangle(img, top_left, bottom_right, 255, 2)

        cv2.imshow(f"{meth}-Detected Image", img)
        cv2.imshow(f"{meth}-Matching", res)

        cv2.waitKey(0)