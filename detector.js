import * as ort from "onnxruntime-web";

let session = null;

//Adding options models soon!!
const modelPath = "src/models/yolov8n-face.onnx";

const imageUpload = document.getElementById("imageUpload");
const runBtn = document.getElementById("runBtn");
const statusDiv = document.getElementById("status");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");

let currentImage = null;


// ============================================================
// MODEL
// ============================================================

/**
 * Loads the model
 * 
 * @returns {Promise<void>}
 * 
 */
async function loadModel() {
    try {
        statusDiv.innerText = "Loading YOLOv8-Face ONNX model...";
        statusDiv.style.color = "#fbbf24";

        // Keep WASM files from the SAME ORT build as ort.min.js
        ort.env.wasm.wasmPaths =
            "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

        session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ["webgpu"]
        });

        console.log("WebGPU available:", !!navigator.gpu);

        if (navigator.gpu) {
            const adapter = await navigator.gpu.requestAdapter();

            console.log("WebGPU adapter:", adapter);

            if (adapter) {
                console.log(
                    "WebGPU adapter info:",
                    await adapter.requestAdapterInfo?.()
                );
            }
        }

        console.log("Model loaded");
        console.log("Inputs:", session.inputNames);
        console.log("Outputs:", session.outputNames);

        console.log("Input metadata:", session.inputMetadata);
        console.log("Output metadata:", session.outputMetadata);

        statusDiv.innerText =
            "Model loaded successfully! Please choose an image.";
        statusDiv.style.color = "#34d399";

    } catch (e) {
        console.error("MODEL LOAD ERROR:", e);

        statusDiv.innerText =
            "Failed to load model. Check console.";
        statusDiv.style.color = "#f87171";
    }
}

loadModel(); // Load the model


// ============================================================
// IMAGE UPLOAD
// ============================================================

imageUpload.addEventListener("change", (e) => {

    const file = e.target.files[0];

    if (!file) return;

    const reader = new FileReader();

    reader.onload = function(event) {

        const img = new Image();

        img.onload = function() {

            currentImage = img;

            canvas.width = img.width;
            canvas.height = img.height;

            ctx.drawImage(img, 0, 0);

            runBtn.disabled = false;

            statusDiv.innerText =
                `Image loaded (${img.width}x${img.height}). Ready for inference.`;
            statusDiv.style.color = "#34d399";
        };

        img.src = event.target.result;
    };

    reader.readAsDataURL(file);
});


// ============================================================
// RUN DETECTION
// ============================================================

runBtn.addEventListener("click", async () => {

    if (!session || !currentImage) {
        return;
    }

    runBtn.disabled = true;

    statusDiv.innerText =
        "Preprocessing and running inference...";
    statusDiv.style.color = "#fbbf24";

    try {

        const results = await runInference(currentImage);

        drawResults(currentImage, results);

        statusDiv.innerText =
            `Detection complete! Found ${results.boxes.length} face(s).`;

        statusDiv.style.color = "#34d399";

    } catch (err) {

        console.error("INFERENCE ERROR:", err);

        statusDiv.innerText =
            "Error during inference execution.";

        statusDiv.style.color = "#f87171";

    } finally {

        runBtn.disabled = false;
    }
});


// ============================================================
// LETTERBOX
// ============================================================

/**
 * Draw a Letterbox around the image
 * @param {HTMLImageElement} img 
 * @param {number} targetWidth 
 * @param {number} targetHeight 
 * @returns {HTMLCanvasElement}
 */
function letterbox(img, targetWidth = 640, targetHeight = 640) {

    const origW = img.width;
    const origH = img.height;

    // Same calculation as Python
    const scale = Math.min(
        targetWidth / origW,
        targetHeight / origH
    );

    // IMPORTANT:
    // Python uses int(), which truncates.
    const newW = Math.floor(origW * scale);
    const newH = Math.floor(origH * scale);

    // Same as Python
    const dw = (targetWidth - newW) / 2;
    const dh = (targetHeight - newH) / 2;

    // IMPORTANT:
    // OpenCV uses int(dh) / int(dw) for actual placement.
    const left = Math.floor(dw);
    const top = Math.floor(dh);

    const tempCanvas = document.createElement("canvas");

    tempCanvas.width = targetWidth;
    tempCanvas.height = targetHeight;

    const tCtx = tempCanvas.getContext("2d");

    // Same padding color as Python
    tCtx.fillStyle = "rgb(114, 114, 114)";

    tCtx.fillRect(
        0,
        0,
        targetWidth,
        targetHeight
    );

    // Match OpenCV's integer pixel placement.
    tCtx.drawImage(
        img,
        left,
        top,
        newW,
        newH
    );

    return {
        canvas: tempCanvas,

        // Keep the original fractional values because
        // Python uses dw/dh for scale_back_detections()
        scale,
        dw,
        dh
    };
}


// ============================================================
// INFERENCE
// ============================================================

/**
 * Run inference on the image
 * @param {HTMLImageElement} img
 * @returns {Promise<Array>}
 */
async function runInference(img) {

    const targetSize = 640;

    const {
        canvas: lbCanvas,
        scale,
        dw,
        dh
    } = letterbox(
        img,
        targetSize,
        targetSize
    );

    const lbCtx = lbCanvas.getContext("2d");

    const imageData = lbCtx.getImageData(
        0,
        0,
        targetSize,
        targetSize
    );

    const pixels = imageData.data;

    // [1, 3, 640, 640]
    const inputTensorValues =
        new Float32Array(
            3 * targetSize * targetSize
        );

    const area = targetSize * targetSize;

    // Canvas gives RGBA.
    //
    // Python does:
    //
    // BGR -> RGB
    // /255
    // HWC -> CHW
    //
    // Canvas is already RGB, so we only need
    // RGB -> CHW and normalization.

    for (let i = 0; i < area; i++) {

        const pixelOffset = i * 4;

        inputTensorValues[i] =
            pixels[pixelOffset] / 255.0;

        inputTensorValues[area + i] =
            pixels[pixelOffset + 1] / 255.0;

        inputTensorValues[2 * area + i] =
            pixels[pixelOffset + 2] / 255.0;
    }

    const inputTensor = new ort.Tensor(
        "float32",
        inputTensorValues,
        [1, 3, targetSize, targetSize]
    );

    const feeds = {
        [session.inputNames[0]]: inputTensor
    };

    // Run model
    const output = await session.run(feeds);

    // Benchmark
    const times = [];

    for (let i = 0; i < 10; i++) {
        const start = performance.now();

        await session.run(feeds);

        const elapsed = performance.now() - start;

        times.push(elapsed);
    }

    console.log(
        "Inference times:",
        times.map(t => `${t.toFixed(2)} ms`)
    );

    const average =
        times.reduce((a, b) => a + b, 0) / times.length;

    console.log(
        `Average inference: ${average.toFixed(2)} ms`
    );

    console.log("Total inference:", average * 10, "ms");

    console.log("Raw model output:");

    for (const name of session.outputNames) {

        const tensor = output[name];

        console.log(
            name,
            "dims:",
            tensor.dims,
            "length:",
            tensor.data.length
        );
    }

    // ========================================================
    // YOLOv8 FACE POSTPROCESSING
    // ========================================================

    const boxes = [];
    const scores = [];
    const landmarks = [];

    const strides = [8, 16, 32];

    const confThreshold = 0.35;
    const iouThreshold = 0.45;

    for (
        let outputIndex = 0;
        outputIndex < session.outputNames.length;
        outputIndex++
    ) {

        const outputName =
            session.outputNames[outputIndex];

        const tensor = output[outputName];

        const dims = tensor.dims;
        const data = tensor.data;

        console.log(
            `Processing ${outputName}:`,
            dims
        );

        /*
         * Expected:
         *
         * [1, 80, 80, 80]
         * [1, 80, 40, 40]
         * [1, 80, 20, 20]
         */

        if (dims.length !== 4) {
            console.warn(
                "Unexpected output dimensions:",
                dims
            );
            continue;
        }

        const batch = dims[0];
        const channels = dims[1];
        const height = dims[2];
        const width = dims[3];

        if (batch !== 1 || channels !== 80) {
            console.warn(
                "Unexpected YOLOv8-Face output:",
                dims
            );
            continue;
        }

        const stride =
            strides[outputIndex];

        const numCells =
            height * width;

        /*
         * Python:
         *
         * pred.reshape(batch_size, channels, -1)
         *     .transpose(0, 2, 1)[0]
         *
         * Result:
         *
         * [height * width, 80]
         */

        for (let cell = 0; cell < numCells; cell++) {

            // ------------------------------------------------
            // DFL BBOX
            // ------------------------------------------------

            const distances = new Float32Array(4);

            for (let side = 0; side < 4; side++) {

                const logits =
                    new Float32Array(16);

                for (let bin = 0; bin < 16; bin++) {

                    const index =
                        ((side * 16 + bin) * numCells)
                        + cell;

                    logits[bin] = data[index];
                }

                const probs =
                    softmax(logits);

                let distance = 0;

                for (let bin = 0; bin < 16; bin++) {

                    distance +=
                        probs[bin] * bin;
                }

                distances[side] = distance;
            }


            // ------------------------------------------------
            // GRID
            // ------------------------------------------------

            const gridX =
                (cell % width) + 0.5;

            const gridY =
                Math.floor(cell / width) + 0.5;


            // ------------------------------------------------
            // BOX
            // ------------------------------------------------

            const x1 =
                (gridX - distances[0]) * stride;

            const y1 =
                (gridY - distances[1]) * stride;

            const x2 =
                (gridX + distances[2]) * stride;

            const y2 =
                (gridY + distances[3]) * stride;


            // ------------------------------------------------
            // CONFIDENCE
            // ------------------------------------------------

            const confIndex =
                (64 * numCells) + cell;

            const rawConfidence =
                data[confIndex];

            const score =
                sigmoid(rawConfidence);

            if (score < confThreshold) {
                continue;
            }


            // ------------------------------------------------
            // LANDMARKS
            // ------------------------------------------------

            const landmarkValues = [];

            const gridKptX =
                cell % width;

            const gridKptY =
                Math.floor(cell / width);

            for (let point = 0; point < 5; point++) {

                const xIndex =
                    ((65 + point * 3) * numCells)
                    + cell;

                const yIndex =
                    ((65 + point * 3 + 1) * numCells)
                    + cell;

                const rawX =
                    data[xIndex];

                const rawY =
                    data[yIndex];

                const landmarkX =
                    (rawX * 2.0 + gridKptX)
                    * stride;

                const landmarkY =
                    (rawY * 2.0 + gridKptY)
                    * stride;

                landmarkValues.push(
                    landmarkX,
                    landmarkY
                );
            }


            // ------------------------------------------------
            // MAP BACK TO ORIGINAL IMAGE
            // ------------------------------------------------

            const originalX1 =
                (x1 - dw) / scale;

            const originalY1 =
                (y1 - dh) / scale;

            const originalX2 =
                (x2 - dw) / scale;

            const originalY2 =
                (y2 - dh) / scale;

            const mappedLandmarks = [];

            for (let i = 0; i < landmarkValues.length; i += 2) {

                mappedLandmarks.push(
                    (landmarkValues[i] - dw) / scale,
                    (landmarkValues[i + 1] - dh) / scale
                );
            }


            boxes.push([
                originalX1,
                originalY1,
                originalX2,
                originalY2
            ]);

            scores.push(score);

            landmarks.push(mappedLandmarks);
        }
    }


    // ========================================================
    // NMS
    // ========================================================

    const selectedIndices =
        nms(
            boxes,
            scores,
            iouThreshold
        );
    
    console.log("================================");
    console.log("YOLO DEBUG");
    console.log("Candidates:", boxes.length);
    console.log("After NMS:", selectedIndices.length);
    console.log("================================");

    const finalBoxes = [];
    const finalScores = [];
    const finalLandmarks = [];

    for (const index of selectedIndices) {

        finalBoxes.push(
            boxes[index]
        );

        finalScores.push(
            scores[index]
        );

        finalLandmarks.push(
            landmarks[index]
        );
    }

    return {
        boxes: finalBoxes,
        scores: finalScores,
        landmarks: finalLandmarks
    };
}


// ============================================================
// SIGMOID
// ============================================================

/**
 *  Sigmoid maps any real number into a value between 0 and 1 using the formula 1 / (1 + e^(-x)).
 *  performs the same thing as tf.nn.sigmoid.
 * @param {*} x 
 * @returns 
 */
function sigmoid(x) {

    return 1 / (
        1 + Math.exp(-x)
    );
}


// ============================================================
// SOFTMAX
// ============================================================

/**
 *  Softmax maps any real number into a value between 0 and 1 using the formula e^x / sum(e^x).
 *  performs the same thing as tf.nn.softmax.
 * @param {*} values 
 * @returns {Float32Array}
 */
function softmax(values) {

    let maxValue = -Infinity;

    for (const value of values) {
        if (value > maxValue) {
            maxValue = value;
        }
    }

    const expValues =
        new Float32Array(values.length);

    let sum = 0;

    for (let i = 0; i < values.length; i++) {

        const value =
            Math.exp(values[i] - maxValue);

        expValues[i] = value;
        sum += value;
    }

    for (let i = 0; i < expValues.length; i++) {

        expValues[i] /= sum;
    }

    return expValues;
}


// ============================================================
// NMS
// ============================================================

/**
 *  Non-Maximum Suppression (NMS) performs the same thing as tf.image.non_max_suppression.
 * @param {*} boxes 
 * @param {*} scores 
 * @param {*} iouThreshold 
 * @returns {number[]}
 */
function nms(boxes, scores, iouThreshold) {

    const order = Array.from(
        { length: boxes.length },
        (_, i) => i
    );

    // Same as Python argsort()[::-1]
    order.sort(
        (a, b) => scores[b] - scores[a]
    );

    const keep = [];

    while (order.length > 0) {

        const current = order[0];

        keep.push(current);

        const remaining = [];

        for (let j = 1; j < order.length; j++) {

            const target = order[j];

            const iou = calculateIoU(
                boxes[current],
                boxes[target]
            );

            if (iou <= iouThreshold) {
                remaining.push(target);
            }
        }

        order.length = 0;

        order.push(...remaining);
    }

    return keep;
}

// ============================================================
// IOU
// ============================================================

/**
 *  Intersection over Union (IoU) performs the same thing as tf.image.non_max_suppression.
 * @param {*} boxA 
 * @param {*} boxB 
 * @returns {number}
 */
function calculateIoU(boxA, boxB) {

    const [
        ax1,
        ay1,
        ax2,
        ay2
    ] = boxA;

    const [
        bx1,
        by1,
        bx2,
        by2
    ] = boxB;

    const interX1 =
        Math.max(ax1, bx1);

    const interY1 =
        Math.max(ay1, by1);

    const interX2 =
        Math.min(ax2, bx2);

    const interY2 =
        Math.min(ay2, by2);

    const interW =
        Math.max(
            0,
            interX2 - interX1
        );

    const interH =
        Math.max(
            0,
            interY2 - interY1
        );

    const interArea =
        interW * interH;

    const areaA =
        Math.max(
            0,
            ax2 - ax1
        ) *
        Math.max(
            0,
            ay2 - ay1
        );

    const areaB =
        Math.max(
            0,
            bx2 - bx1
        ) *
        Math.max(
            0,
            by2 - by1
        );

    const union =
        areaA + areaB - interArea;

    if (union <= 0) {
        return 0;
    }

    return interArea / union;
}


// ============================================================
// DRAW RESULTS
// ============================================================

/**
 *  Draw results performs the same thing as tf.image.draw_bounding_boxes.
 * @param {HTMLImageElement} img 
 * @param {Object} results 
 */
function drawResults(
    img,
    results
) {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    ctx.drawImage(
        img,
        0,
        0
    );

    ctx.strokeStyle = "#22c55e";

    ctx.fillStyle = "#22c55e";

    ctx.lineWidth =
        Math.max(
            2,
            img.width / 400
        );

    ctx.font =
        `${Math.max(
            10,
            img.width / 60
        )}px sans-serif`;


    for (
        let i = 0;
        i < results.boxes.length;
        i++
    ) {

        const [
            x1,
            y1,
            x2,
            y2
        ] = results.boxes[i];

        const score =
            (
                results.scores[i] * 100
            ).toFixed(0);

        const width =
            x2 - x1;

        const height =
            y2 - y1;


        // Bounding box
        ctx.strokeRect(
            x1,
            y1,
            width,
            height
        );


        // Confidence background
        ctx.fillStyle =
            "#22c55e";

        ctx.fillRect(
            x1,
            Math.max(0, y1 - 20),
            50,
            20
        );


        // Confidence text
        ctx.fillStyle =
            "#0f172a";

        ctx.fillText(
            `${score}%`,
            x1 + 3,
            Math.max(14, y1 - 5)
        );


        // ----------------------------------------------------
        // LANDMARKS
        // ----------------------------------------------------

        const lms =
            results.landmarks[i];

        if (lms && lms.length >= 10) {

            for (
                let point = 0;
                point < 5;
                point++
            ) {

                const lx =
                    lms[point * 2];

                const ly =
                    lms[point * 2 + 1];

                ctx.beginPath();

                ctx.arc(
                    lx,
                    ly,
                    Math.max(
                        2,
                        img.width / 500
                    ),
                    0,
                    Math.PI * 2
                );

                ctx.fillStyle =
                    "#ef4444";

                ctx.fill();
            }
        }

        ctx.fillStyle =
            "#22c55e";
    }
}