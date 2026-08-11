from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime
import requests
import torch
import torchvision


class YOLOv8Face:
    """
    YOLOv8-Face ONNX inference class.
    
    Args:
        model_path (str): Path to the ONNX model file.
        conf_thres (float, optional): Confidence threshold for object detection. Defaults to 0.25.
        iou_thres (float, optional): Intersection over Union (IoU) threshold for non-maximum suppression. Defaults to 0.45.
        max_det (int, optional): Maximum number of detections to return. Defaults to 300.
        nms_mode (str, optional): NMS mode for non-maximum suppression. Defaults to "torchvision".
    """

    def __init__(
        self,
        model_path: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        nms_mode: str = "torchvision",
    ) -> None:
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.nms_mode = nms_mode

        # Initialize model (sets self.img_size from ONNX input shape)
        self._initialize_model(model_path)

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the model on the given image and return predictions.

        Args:
            image (np.ndarray): Input image as a numpy array.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - boxes (np.ndarray): Bounding boxes for detected faces.
                - scores (np.ndarray): Confidence scores for detected faces.
                - landmarks (np.ndarray): Landmarks for detected faces.
        """
        if not isinstance(image, np.ndarray) or len(image.shape) != 3:
            raise ValueError("Input image must be a numpy array with 3 dimensions (H, W, C).")

        detections = self.detect(image)

        if len(detections) == 0:
            return np.array([]), np.array([]), np.array([])

        boxes = detections[:, :4]
        scores = detections[:, 4]
        landmarks = detections[:, 5:]

        return boxes, scores, landmarks

    def _initialize_model(self, model_path: str) -> None:
        """
        Initialize the model from the given path.

        Args:
            model_path (str): Path to the ONNX model file.

        Raises:
            RuntimeError: If the model fails to load.

        Returns:
            None
        """
        try:
            self.session = onnxruntime.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            # Get model info
            self.output_names = [x.name for x in self.session.get_outputs()]
            self.input_names = [x.name for x in self.session.get_inputs()]

            # Get input shape from model (e.g., [1, 3, 640, 640])
            input_shape = self.session.get_inputs()[0].shape
            self.img_size = (input_shape[2], input_shape[3])  # (H, W)

            # Get model metadata
            meta = self.session.get_modelmeta()
            if meta.custom_metadata_map:
                self.stride = int(meta.custom_metadata_map.get("stride", 32))
            else:
                self.stride = 32

        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {e}") from e

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            img (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Preprocessed image as a numpy array.
        """
        # Convert BGR to RGB
        img = img[:, :, ::-1]

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Transpose to CHW format and add batch dimension
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)

        return img

    def postprocess(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Postprocess model predictions.

        Args:
            predictions (List[np.ndarray]): List of model predictions.

        Returns:
            np.ndarray: Postprocessed predictions as a numpy array.
        """
        boxes_list = []
        scores_list = []
        landmarks_list = []

        strides = [8, 16, 32]  # YOLOv8 strides for 640x640 input

        for pred, stride in zip(predictions, strides):
            batch_size, channels, height, width = pred.shape

            # Reshape: (1, 80, H, W) -> (1, 80, H*W) -> (1, H*W, 80) -> (H*W, 80)
            pred = pred.reshape(batch_size, channels, -1).transpose(0, 2, 1)[0]

            # Create grid with 0.5 offset (matching PyTorch's make_anchors)
            grid_y, grid_x = np.meshgrid(np.arange(height) + 0.5, np.arange(width) + 0.5, indexing="ij")
            grid_x = grid_x.flatten()
            grid_y = grid_y.flatten()

            # Extract components
            bbox_pred = pred[:, :64]  # DFL bbox prediction
            cls_conf = pred[:, 64]  # Class confidence
            kpt_pred = pred[:, 65:]  # 15 keypoint values (5 points * 3: x, y, visibility)

            # Decode bounding boxes from DFL
            bbox_pred = bbox_pred.reshape(-1, 4, 16)
            bbox_dist = self.softmax(bbox_pred, axis=-1) @ np.arange(16)

            # Convert distances to xyxy format
            x1 = (grid_x - bbox_dist[:, 0]) * stride
            y1 = (grid_y - bbox_dist[:, 1]) * stride
            x2 = (grid_x + bbox_dist[:, 2]) * stride
            y2 = (grid_y + bbox_dist[:, 3]) * stride
            boxes = np.stack([x1, y1, x2, y2], axis=-1)

            # Decode keypoints: kpt = (kpt * 2.0 + grid) * stride
            kpt_grid_y, kpt_grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            kpt_grid_x = kpt_grid_x.flatten()
            kpt_grid_y = kpt_grid_y.flatten()

            kpt_pred = kpt_pred.reshape(-1, 5, 3)
            kpt_x = (kpt_pred[:, :, 0] * 2.0 + kpt_grid_x[:, None]) * stride
            kpt_y = (kpt_pred[:, :, 1] * 2.0 + kpt_grid_y[:, None]) * stride
            landmarks = np.stack([kpt_x, kpt_y], axis=-1).reshape(-1, 10)

            # Apply sigmoid to class confidence
            scores = 1 / (1 + np.exp(-cls_conf))

            boxes_list.append(boxes)
            scores_list.append(scores)
            landmarks_list.append(landmarks)

        # Concatenate all predictions
        boxes = np.concatenate(boxes_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
        landmarks = np.concatenate(landmarks_list, axis=0)

        # Filter by confidence
        mask = scores >= self.conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        landmarks = landmarks[mask]

        if len(boxes) == 0:
            return np.array([])

        # Apply NMS
        if self.nms_mode == "torchvision":
            indices = torchvision.ops.nms(
                torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(scores, dtype=torch.float32),
                self.iou_thres,
            ).numpy()
        else:
            indices = self.nms(boxes, scores, self.iou_thres)

        if len(indices) == 0:
            return np.array([])

        # Filter detections and limit to max_det
        indices = indices[: self.max_det]
        boxes = boxes[indices]
        scores = scores[indices]
        landmarks = landmarks[indices]

        # Combine results
        detections = np.concatenate([boxes, scores[:, None], landmarks], axis=1)

        return detections

    @staticmethod
    def softmax(x, axis=-1):
        """
        Compute softmax values for array x.

        Args:
            x (np.ndarray): Input array.
            axis (int, optional): Axis along which to compute softmax. Default is -1.

        Returns:
            np.ndarray: Softmax values.
        """
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Non-Maximum Suppression (NumPy implementation).

        Args:
            boxes (np.ndarray): Bounding boxes with shape (N, 4).
            scores (np.ndarray): Confidence scores with shape (N,).
            iou_threshold (float): Intersection over Union threshold for NMS.

        Returns:
            List[int]: Indices of the selected bounding boxes.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, img: np.ndarray) -> np.ndarray:
        """
        Run face detection on image.

        Args:
            img (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Detected faces as a numpy array.
        """
        input_tensor = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        detections = self.postprocess(outputs)
        return detections


def letterbox(image, target_shape=(640, 640), color=(114, 114, 114)):
    """
    Resizes and pads image to target_shape while keeping aspect ratio.

    Args:
        image (np.ndarray): Input image as a numpy array.
        target_shape (tuple, optional): Target shape as a tuple (height, width). Default is (640, 640).
        color (tuple, optional): Color to use for padding. Default is (114, 114, 114).

    Returns:
        np.ndarray: Padded and resized image as a numpy array.
    """
    height, width = image.shape[:2]
    scale = min(target_shape[0] / height, target_shape[1] / width)
    new_size = (int(width * scale), int(height * scale))
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    dw, dh = (target_shape[1] - new_size[0]) / 2, (target_shape[0] - new_size[1]) / 2
    top, bottom = int(dh), int(target_shape[0] - new_size[1] - int(dh))
    left, right = int(dw), int(target_shape[1] - new_size[0] - int(dw))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, scale, (dw, dh)


def scale_back_detections(boxes, landmarks, scale, dw, dh, orig_shape):
    """
    Rescales boxes and landmarks back to original image dimensions.

    Args:
        boxes (np.ndarray): Bounding boxes as a numpy array.
        landmarks (np.ndarray): Landmarks as a numpy array.
        scale (float): Scale factor.
        dw (float): Width padding.
        dh (float): Height padding.
        orig_shape (tuple): Original image shape as a tuple (height, width).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - boxes (np.ndarray): Rescaled bounding boxes.
            - landmarks (np.ndarray): Rescaled landmarks.
    """
    orig_h, orig_w = orig_shape

    # Rescale boxes
    if len(boxes) > 0:
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes[:, :4] /= scale
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

    # Rescale landmarks
    if len(landmarks) > 0:
        landmarks[:, 0::2] = (landmarks[:, 0::2] - dw) / scale
        landmarks[:, 1::2] = (landmarks[:, 1::2] - dh) / scale

    return boxes, landmarks


def draw_detections(image: np.ndarray, box: np.ndarray, score: float, landmarks: np.ndarray) -> None:
    """
    Draw face bounding box and landmarks on image.

    Args:
        image (np.ndarray): Input image as a numpy array.
        box (np.ndarray): Bounding box coordinates as a numpy array.
        score (float): Confidence score for the face.
        landmarks (np.ndarray): Landmarks for the face as a numpy array.

    Returns:
        None
    """
    x1, y1, x2, y2 = map(int, box)
    color = (0, 255, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    label = f"{score:.2f}"
    cv2.putText(image, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, lineType=cv2.LINE_AA)

    if landmarks is not None and len(landmarks) >= 10:
        landmarks = landmarks.reshape(5, 2).astype(int)
        for i, (lx, ly) in enumerate(landmarks):
            landmark_color = (0, 0, 255) if i < 2 else ((255, 0, 0) if i == 2 else (0, 255, 0))
            cv2.circle(image, (lx, ly), 3, landmark_color, -1)


def load_image_from_url(url):
    """
    Load an image from a URL and return it as a numpy array.

    Args:
        url (str): The URL of the image to load.

    Returns:
        np.ndarray: The image as a numpy array.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    else:
        print(f"Failed to download image. Status code: {response.status_code}")
        return None


def main():
    model_path = "yolov8n-face.onnx" # Replace with the path to your YOLOv8-Face ONNX model
    detector = YOLOv8Face(model_path, conf_thres=0.35, nms_mode="torchvision") # Initialize the YOLOv8-Face detector
    
    image_url = "https://cdn.mos.cms.futurecdn.net/GA98TY8kmqu5WtSx4m9Ha7-1200-80.jpg" # Replace with the URL of the image
    print("Downloading image from URL...")
    frame = load_image_from_url(image_url)
    
    if frame is not None:
        orig_h, orig_w = frame.shape[:2]
        print(f"Processing image successfully. Resolution: {orig_w}x{orig_h}")
        
        # Apply letterbox padding before running inference
        letterboxed_img, scale, (dw, dh) = letterbox(frame, (640, 640))
        
        # Run detector
        boxes, scores, landmarks = detector(letterboxed_img)
        
        # Scale boxes and landmarks back to original frame sizes
        boxes, landmarks = scale_back_detections(boxes, landmarks, scale, dw, dh, (orig_h, orig_w))
        
        print(f"Detection complete! Found {len(boxes)} faces.")
        print("Drawing bounding boxes and landmarks...")
        
        for box, score, lms in zip(boxes, scores, landmarks):
            draw_detections(frame, box, score, lms)
            
        # Resize output down if too big for window display
        max_dim = 1200
        display_frame = frame
        if orig_w > max_dim or orig_h > max_dim:
            disp_scale = max_dim / max(orig_w, orig_h)
            display_frame = cv2.resize(frame, (int(orig_w * disp_scale), int(orig_h * disp_scale)))

        cv2.imshow("YOLOv8-Face URL Static Test", display_frame)
        print("Press any key on the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()