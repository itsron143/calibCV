import numpy as np
import argparse
import imutils
import cv2

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours


# Variables for each extruder
TRUE_DIST = [2, 4, 6, 8, 10, 12, 14]

REF_CENTER_COORDINATE = []
PROJECTED_COORDINATES = []

COLORS = (
    (0, 0, 255),
    (0, 255, 128),
    (255, 255, 0),
    (255, 0, 0),
    (255, 0, 255),
    (0, 0, 0),
    (255, 255, 255),
    (0, 255, 255),
)
COLORS_NAMES = [
    "Red",
    "Green",
    "Light Blue",
    "Dark Blue",
    "Purple",
    "Black",
    "White",
    "Yellow",
]


# calculate midpoint
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# calculate distance in cm based on reference object width
def dist_cm(ptA, ptB, refObj):
    return dist.euclidean(ptA, ptB) / refObj[2]


# pre-process the image to extract the squares
def pre_process(image):
    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    return edged


# projection coordinates of square center on the reference square line
def true_dist_from_ref_center(image, center_coordinates, REF_CENTER_COORDINATE):
    center_x, center_y = REF_CENTER_COORDINATE
    for coord in center_coordinates:
        temp_x, temp_y = coord
        PROJECTED_COORDINATES.append((temp_x, center_y))
    return image


# calculate offsets
def get_offsets(center_coordinates, refObj, REF_CENTER_COORDINATE):
    offsets = []
    y_offsets = []
    x_offsets = []

    # X Offsets
    for i in range(len(TRUE_DIST)):
        true_x, true_y = PROJECTED_COORDINATES[i]
        dist_from_ref = dist.euclidean(REF_CENTER_COORDINATE, (true_x, true_y)) / refObj[2]
        temp_offset = TRUE_DIST[i] - dist_from_ref
        if temp_offset >= 0:
            x_offsets.append(str(abs(temp_offset)))
        else:
            x_offsets.append("-" + str(abs(temp_offset)))

    # Y Offsets
    for i in range(len(center_coordinates)):
        true_x, true_y = PROJECTED_COORDINATES[i]
        cen_x, cen_y = center_coordinates[i]
        if cen_y >= true_y:
            offset_dist = dist_cm((true_x, true_y), (true_x, cen_y), refObj)
            y_offsets.append(str(abs(offset_dist)))
        else:
            offset_dist = dist_cm((true_x, cen_y), (true_x, true_y), refObj)
            y_offsets.append("-" + str(abs(offset_dist)))

    offsets = [x_offsets, y_offsets]

    return offsets


def calc_offsets(image, ref_obj_width):
    # Preprocess the Image
    global COLORS
    global COLORS_NAMES
    global TRUE_DIST
    edged = pre_process(image)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # sort the contours from left-to-right and, then initialize the
    # distance colors and reference object
    (cnts, _) = contours.sort_contours(cnts)

    refObj = None
    center_coordinates = []
    i = 0
    offsets = []

    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue

        M = cv2.moments(c)

        # get coordinates of the center
        conX = int(M["m10"] / M["m00"])
        conY = int(M["m01"] / M["m00"])

        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        # compute the center of the bounding box
        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])

        if refObj is not None:
            center_coordinates.append((cX, cY))
            cv2.circle(image, (cX, cY), 3, COLORS[i], 4)
            # print("Square", i+1, "-", COLORS_NAMES[i], "::", "Center -", [cX, cY])
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(c)
            cv2.rectangle(
                image, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 255, 0), 2
            )
            i += 1

        # if this is the first contour we are examining (i.e.,
        # the left-most contour), we presume this is the
        # reference object
        if refObj is None:
            # unpack the ordered bounding box, then compute the
            # midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-right and
            # bottom-right
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(c)
            cv2.rectangle(
                image, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 255, 0), 1
            )
            (tl, tr, br, bl) = box
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            # compute the Euclidean distance between the midpoints,
            # then construct the reference object
            D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            refObj = (box, (cX, cY), D / ref_obj_width)
            continue

    REF_CENTER_COORDINATE = tuple(center_coordinates[0])
    center_coordinates = center_coordinates[1:]
    TRUE_DIST = TRUE_DIST[:len(center_coordinates)]
    COLORS = COLORS[:len(center_coordinates)]
    COLORS_NAMES = COLORS_NAMES[:len(center_coordinates)]
    image = true_dist_from_ref_center(image, center_coordinates, REF_CENTER_COORDINATE)
    offsets = get_offsets(center_coordinates, refObj, REF_CENTER_COORDINATE)
    return offsets, image
