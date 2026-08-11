export const MODELS = {
    "yolov8n": {
        name: "YOLOv8n-Face",
        path: "/models/yolov8n-face.onnx",
        sizeMB: 12,
        inputSize: 640,
    },

    "yolov8-lite-s": {
        name: "YOLOv8-Lite-S-Face",
        path: "/models/yolov8-lite-s.onnx",
        sizeMB: 7.4,
        inputSize: 640,
    }
};

export const MODEL_ALIASES = {
    standard: "yolov8n",
    lite: "yolov8-lite-s",
};