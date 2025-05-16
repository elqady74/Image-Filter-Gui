from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from io import BytesIO
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def apply_filter(img, filter_name):
    try:
        if filter_name == "Add noise":
            noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
            return cv2.add(img, noise)
        elif filter_name == "Remove noise":
            return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        elif filter_name == "Mean filter":
            return cv2.blur(img, (5, 5))
        elif filter_name == "Median filter":
            return cv2.medianBlur(img, 5)
        elif filter_name == "Gaussian filter":
            return cv2.GaussianBlur(img, (5, 5), 0)
        elif filter_name == "Gaussian noise":
            gauss = np.random.normal(0, 20, img.shape).astype(np.uint8)
            return cv2.add(img, gauss)
        elif filter_name == "Erosion":
            kernel = np.ones((5, 5), np.uint8)
            return cv2.erode(img, kernel, iterations=1)
        elif filter_name == "Dilation":
            kernel = np.ones((5, 5), np.uint8)
            return cv2.dilate(img, kernel, iterations=1)
        elif filter_name == "Opening":
            kernel = np.ones((5, 5), np.uint8)
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        elif filter_name == "Closing":
            kernel = np.ones((5, 5), np.uint8)
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        elif filter_name == "Boundary extraction":
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(img, kernel, iterations=1)
            return cv2.absdiff(img, eroded)
        elif filter_name == "Region filling":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            h, w = th.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)
            filled = th.copy()
            cv2.floodFill(filled, mask, (0, 0), 255)
            inv = cv2.bitwise_not(filled)
            result = cv2.bitwise_or(th, inv)
            return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif filter_name == "Global threshold":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)
        elif filter_name == "Adaptive threshold":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
            return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        elif filter_name == "Otsu threshold":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        elif filter_name == "Hough":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=30, maxLineGap=10)
            hough_img = img.copy()
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(hough_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return hough_img
        elif filter_name == "Watershed":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            img_copy = img.copy()
            markers = cv2.watershed(img_copy, markers)
            img_copy[markers == -1] = [255, 0, 0]
            return img_copy
        else:
            return img
    except Exception as e:
        raise Exception(f"Error in filter {filter_name}: {str(e)}")

@app.post("/apply_filter")
async def apply_filter_endpoint(file: UploadFile = File(...), filter_name: str = Form(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        filtered_img = apply_filter(img, filter_name)
        _, buffer = cv2.imencode('.png', filtered_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return {"filtered_image": f"data:image/png;base64,{img_base64}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))