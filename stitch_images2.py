import sys
import cv2
import numpy as np
import imutils

# Using keypoints to stitch the images

def get_stitched_image(img1, img2, M):

    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)

    img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

    # Resulting dimensions

    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)

    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    transform_dist = [-x_min, -y_min]

    transform_array = np.array([[1, 0, transform_dist[0]],

                                [0, 1, transform_dist[1]],

                                [0, 0, 1]])

    result_img = cv2.warpPerspective(img2, transform_array.dot(M),

                                     (x_max - x_min, y_max - y_min))

    result_img[transform_dist[1]:w1 + transform_dist[1],

    transform_dist[0]:h1 + transform_dist[0]] = img1

    return result_img


# Find SIFT and return Homography Matrix

def get_sift_homography(img1, img2):

    sift = cv2.xfeatures2d.SIFT_create()
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

#BruteForceMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)

    verify_ratio = 0.8 

    verified_matches = []

    for m1, m2 in matches:
        if m1.distance < verify_ratio * m2.distance:
            verified_matches.append(m1)


    min_matches = 8

    if len(verified_matches) > min_matches:

        img1_pts = []

        img2_pts = []

        for match in verified_matches:
            img1_pts.append(k1[match.queryIdx].pt)

            img2_pts.append(k2[match.trainIdx].pt)

        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)

        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)

        return M

    else:
        print ('Error: Not enough matches')
        exit()


def equalize_histogram_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img

def main():
    # Get input set of images

    img1 = cv2.imread('C:\opencv_test\img\im1.JPG')
    #cv2.imshow('Ip1', img1)
    #cv2.waitKey() & 0xFF


    img2 = cv2.imread('C:\opencv_test\img\im2.JPG')
    #cv2.imshow('Ip2', img2)
    #cv2.waitKey() & 0xFF


    img1 = imutils.resize(img1, width=550)
    img2 = imutils.resize(img2, width=550)
    img1 = imutils.resize(img1, height=450)
    img2 = imutils.resize(img2, height=450)


    ##################################################

    # Equalize histogram (Use in case of errors/not enough matching keypoints)

    #img1 = equalize_histogram_color(img1)

    #img2 = equalize_histogram_color(img2)


    ###################################################

    # Show input images

    #cv2.imshow('I/P - 1',  img1)
    #cv2.imshow('I/P - 2', img2)

    M = get_sift_homography(img1, img2)
    result_image = get_stitched_image(img2, img1, M)

    result_image_name = 'results.png'

    cv2.imwrite(result_image_name, result_image)

    cv2.imshow('Result', result_image)

    cv2.waitKey(0)


# Call main function

if __name__ == '__main__':
    main()
