import * as ort from "onnxruntime-web";

/**
 * Resize an image using Python/OpenCV-compatible letterboxing.
 *
 * @param {HTMLImageElement|HTMLCanvasElement|ImageBitmap} image
 * @param {number} targetWidth
 * @param {number} targetHeight
 * @returns {{
 *   canvas: HTMLCanvasElement,
 *   scale: number,
 *   dw: number,
 *   dh: number
 * }}
 */
export function letterbox(
    image,
    targetWidth = 640,
    targetHeight = 640
) {
    const origW = image.width;
    const origH = image.height;

    const scale = Math.min(
        targetWidth / origW,
        targetHeight / origH
    );

    const newW = Math.floor(origW * scale);
    const newH = Math.floor(origH * scale);

    const dw = (targetWidth - newW) / 2;
    const dh = (targetHeight - newH) / 2;

    const left = Math.floor(dw);
    const top = Math.floor(dh);

    const canvas = document.createElement("canvas");

    canvas.width = targetWidth;
    canvas.height = targetHeight;

    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "rgb(114, 114, 114)";

    ctx.fillRect(
        0,
        0,
        targetWidth,
        targetHeight
    );

    ctx.drawImage(
        image,
        left,
        top,
        newW,
        newH
    );

    return {
        canvas,
        scale,
        dw,
        dh
    };
}


/**
 * Convert a letterboxed canvas into an ONNX tensor.
 *
 * Output shape:
 * [1, 3, 640, 640]
 *
 * @param {HTMLCanvasElement} canvas
 * @returns {ort.Tensor}
 */
export function canvasToTensor(canvas) {
    const width = canvas.width;
    const height = canvas.height;

    const ctx = canvas.getContext("2d");

    const imageData = ctx.getImageData(
        0,
        0,
        width,
        height
    );

    const pixels = imageData.data;

    const area = width * height;

    const values = new Float32Array(
        3 * area
    );

    for (let i = 0; i < area; i++) {
        const pixelOffset = i * 4;

        values[i] =
            pixels[pixelOffset] / 255.0;

        values[area + i] =
            pixels[pixelOffset + 1] / 255.0;

        values[2 * area + i] =
            pixels[pixelOffset + 2] / 255.0;
    }

    return new ort.Tensor(
        "float32",
        values,
        [1, 3, height, width]
    );
}


/**
 * Prepare an image for YOLO inference.
 *
 * @param {HTMLImageElement|HTMLCanvasElement|ImageBitmap} image
 * @param {number} targetSize
 */
export function preprocess(
    image,
    targetSize = 640
) {
    const result = letterbox(
        image,
        targetSize,
        targetSize
    );

    const tensor = canvasToTensor(
        result.canvas
    );

    return {
        tensor,
        scale: result.scale,
        dw: result.dw,
        dh: result.dh
    };
}