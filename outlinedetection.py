# outlinedetection.py (updated)
import cv2
from camera_config import init_video_capture, release_video_capture
import numpy as np

def safe_div(x,y):
    if y==0: return 0
    return x/y

def nothing(x):
    pass

def rescale_frame(frame, percent=25):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

class perspective:
    @staticmethod
    def order_points(points):
        # sort the points by their x-coordinates
        points = points[np.argsort(np.ravel(points[:,0]))]

        # find the top-left and bottom-right coordinates of the contour
        tl = np.min(points[:, 0], axis=0)
        br = np.max(points[:, 0], axis=0)

        # calculate the center coordinates of the rectangle
        cX = (tl[0] + br[0]) / 2.0
        cY = (tl[1] + br[1]) / 2.0

        # define the dimensions and shape of the rectangle
        (x, y) = tl
        (A, B) = (cX, cY)

        return [[(int(x), int(y))],
                [(int(A), int(B))],
                [(int(br[0]), int(br[1]))],
                [(tl[0], br[1])]]

class dist:
    @staticmethod
    def euclidean(pointA, pointB):
        return np.sqrt((pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2)

def main():
    videocapture = init_video_capture()

    if not videocapture.isOpened():
        print("can't open camera")
        exit()

    windowName="Webcam Live video feed"

    cv2.namedWindow(windowName)

    # Sliders to adjust image
    cv2.createTrackbar("threshold", windowName, 75, 255, nothing)
    cv2.createTrackbar("kernel", windowName, 5, 30, nothing)
    cv2.createTrackbar("iterations", windowName, 1, 10, nothing)

    showLive=True
    while(showLive):
        ret, frame=videocapture.read()
        if not ret:
            print("cannot capture the frame")
            exit()

        thresh= cv2.getTrackbarPos("threshold", windowName)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(gray,thresh,255,cv2.THRESH_BINARY)

        kern=cv2.getTrackbarPos("kernel", windowName)
        kernel = np.ones((kern,kern),np.uint8)

        itera=cv2.getTrackbarPos("iterations", windowName)
        dilation =   cv2.dilate(thresh1, kernel, iterations=itera)
        erosion = cv2.erode(dilation,kernel,iterations = itera)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # focus on only the largest outline by area
        areas = []
        for contour in contours:
          ar = cv2.contourArea(contour)
          areas.append(ar)

        max_area = max(areas)
        max_area_index = areas.index(max_area)
        cnt = contours[max_area_index]

        # compute the rotated bounding box of the contour
        orig = frame.copy()
        box = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)

        # loop hover the original points and draw them
        for (x, y) in box:
          cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 1)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 1)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # draw the object sizes on the image
        cv2.putText(orig, "{:.1f}mm".format(dA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}mm".format(dB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # compute the center of the contour
        M = cv2.moments(cnt)
        cX = int(safe_div(M["m10"],M["m00"]))
        cY = int(safe_div(M["m01"],M["m00"]))

        # draw the contour and center of the shape on the image
        cv2.circle(orig, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(orig, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # show the image
        cv2.imshow(windowName, orig)

        if cv2.waitKey(30)>=0:
            showLive=False

    release_video_capture(videocapture)
    cv2.destroyAllWindows()

def midpoint(pointA, pointB):
    return ((pointA[0] + pointB[0]) / 2.0, (pointA[1] + pointB[1]) / 2.0)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

