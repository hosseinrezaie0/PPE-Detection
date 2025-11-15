from ultralytics import YOLO
import cv2
import cvzone

data_path = "videos/sample_video.mp4"
model_path = "model/fine_tuned_model.pt"

classes = ["boost", "gloves", "helmet", "human", "vest"]
ppe_classes = ["vest", "helmet"]

cap = cv2.VideoCapture(data_path)
model = YOLO(model_path)

def iou(boxA, boxB):
#     box = [x1, y1, x2, y2]
    x1_iou = max(boxA[0], boxB[0])
    y1_iou = max(boxA[1], boxB[1])
    x2_iou = max(boxA[2], boxB[2])
    y2_iou = max(boxA[3], boxB[3])
    # the width of overlap
    w_iou = max(0, x2_iou - x1_iou)
    # the height of overlap
    h_iou = max(0, y2_iou - y1_iou)
    # the total area of overlap
    area = w_iou * h_iou

    if area == 0:
        return 0
    boxA_area = ((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = ((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return area / (boxA_area + boxB_area - area)


def center_inside(human_box, other_box):
    """
    Check if the center of other box is inside the human box
    """
    cx = (other_box[2] - other_box[0]) / 2
    cy = (other_box[3] - other_box[1]) / 2
    return (human_box[0] <= cx <= human_box[2]) and (human_box[1] <= cy <= human_box[3])


try:
    print(f"Model class names: {model.names}")
except Exception:
    print(f"The error {Exception} occurred")



while True:
    success, img = cap.read()
    if not success:
        print("No video frame found or end of file - exiting...")
        break

    # img = cv2.resize(img, (640, 640))
    results = model(img, stream=True)


    # Collect all detections for this frame
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "cls": cls,
                "conf": conf
            })

    for det in detections:
        cls_name = classes[det["cls"]]
        x1, y1, x2, y2 = det["bbox"]

        if cls_name == "human":
            human_box = det["bbox"]
            found_ppe = set()
            for other in detections:
                other_name = classes[other["cls"]]
                other_box = other["bbox"]
                if other is det:
                    pass
                if other_name in ppe_classes:
                    # Consider overlap or center inside as evidence
                    if iou(human_box, other_box) >= 0.5 or center_inside(human_box, other_box):
                        found_ppe.add(other_name)
            if found_ppe:
                # Green rectangle if ppe present
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)
                label = "PPE " + ",".join(sorted((found_ppe)))
                cvzone.putTextRect(img, f"{label}", (max(0, x1), max(0, y1)), thickness=1, scale=1)
            else:
                # Red rectangle = missing ppe
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cvzone.putTextRect(img, "No PPE Detected!", (x1, y1-10), thickness=1, scale=1)
        else:
            pass

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()






