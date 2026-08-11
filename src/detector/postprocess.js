import { nms } from "./nms.js";


function sigmoid(x) {
    return 1 / (
        1 + Math.exp(-x)
    );
}


function softmax(values) {
    let maxValue = -Infinity;

    for (const value of values) {
        if (value > maxValue) {
            maxValue = value;
        }
    }

    const expValues =
        new Float32Array(
            values.length
        );

    let sum = 0;

    for (let i = 0; i < values.length; i++) {
        const value =
            Math.exp(
                values[i] - maxValue
            );

        expValues[i] = value;

        sum += value;
    }

    for (
        let i = 0;
        i < expValues.length;
        i++
    ) {
        expValues[i] /= sum;
    }

    return expValues;
}


/**
 * Decode YOLOv8-Face outputs.
 *
 * @param {Object} output
 * @param {string[]} outputNames
 * @param {number} scale
 * @param {number} dw
 * @param {number} dh
 * @param {Object} options
 */
export function postprocess(
    output,
    outputNames,
    {
        scale,
        dw,
        dh
    },
    {
        confThreshold = 0.35,
        iouThreshold = 0.45
    } = {}
) {
    const boxes = [];
    const scores = [];
    const landmarks = [];

    const strides = [8, 16, 32];

    for (
        let outputIndex = 0;
        outputIndex < outputNames.length;
        outputIndex++
    ) {
        const outputName =
            outputNames[outputIndex];

        const tensor =
            output[outputName];

        const dims =
            tensor.dims;

        const data =
            tensor.data;

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

        if (
            batch !== 1 ||
            channels !== 80
        ) {
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

        for (
            let cell = 0;
            cell < numCells;
            cell++
        ) {
            // ================================================
            // DFL BBOX
            // ================================================

            const distances =
                new Float32Array(4);

            for (
                let side = 0;
                side < 4;
                side++
            ) {
                const logits =
                    new Float32Array(16);

                for (
                    let bin = 0;
                    bin < 16;
                    bin++
                ) {
                    const index =
                        (
                            (side * 16 + bin)
                            * numCells
                        ) + cell;

                    logits[bin] =
                        data[index];
                }

                const probs =
                    softmax(logits);

                let distance = 0;

                for (
                    let bin = 0;
                    bin < 16;
                    bin++
                ) {
                    distance +=
                        probs[bin] * bin;
                }

                distances[side] =
                    distance;
            }


            // ================================================
            // GRID
            // ================================================

            const gridX =
                (cell % width) + 0.5;

            const gridY =
                Math.floor(
                    cell / width
                ) + 0.5;


            // ================================================
            // BOX
            // ================================================

            const x1 =
                (gridX - distances[0])
                * stride;

            const y1 =
                (gridY - distances[1])
                * stride;

            const x2 =
                (gridX + distances[2])
                * stride;

            const y2 =
                (gridY + distances[3])
                * stride;


            // ================================================
            // CONFIDENCE
            // ================================================

            const confIndex =
                (64 * numCells) + cell;

            const rawConfidence =
                data[confIndex];

            const score =
                sigmoid(rawConfidence);

            if (
                score < confThreshold
            ) {
                continue;
            }


            // ================================================
            // LANDMARKS
            // ================================================

            const landmarkValues = [];

            const gridKptX =
                cell % width;

            const gridKptY =
                Math.floor(
                    cell / width
                );

            for (
                let point = 0;
                point < 5;
                point++
            ) {
                const xIndex =
                    (
                        (65 + point * 3)
                        * numCells
                    ) + cell;

                const yIndex =
                    (
                        (65 + point * 3 + 1)
                        * numCells
                    ) + cell;

                const rawX =
                    data[xIndex];

                const rawY =
                    data[yIndex];

                const landmarkX =
                    (
                        rawX * 2.0 +
                        gridKptX
                    ) * stride;

                const landmarkY =
                    (
                        rawY * 2.0 +
                        gridKptY
                    ) * stride;

                landmarkValues.push(
                    landmarkX,
                    landmarkY
                );
            }


            // ================================================
            // MAP BACK TO ORIGINAL IMAGE
            // ================================================

            const originalX1 =
                (x1 - dw) / scale;

            const originalY1 =
                (y1 - dh) / scale;

            const originalX2 =
                (x2 - dw) / scale;

            const originalY2 =
                (y2 - dh) / scale;

            const mappedLandmarks = [];

            for (
                let i = 0;
                i < landmarkValues.length;
                i += 2
            ) {
                mappedLandmarks.push(
                    (landmarkValues[i] - dw)
                    / scale,

                    (landmarkValues[i + 1] - dh)
                    / scale
                );
            }


            boxes.push([
                originalX1,
                originalY1,
                originalX2,
                originalY2
            ]);

            scores.push(score);

            landmarks.push(
                mappedLandmarks
            );
        }
    }


    // ================================================
    // NMS
    // ================================================

    const selectedIndices =
        nms(
            boxes,
            scores,
            iouThreshold
        );

    console.log(
        "================================"
    );

    console.log(
        "YOLO DEBUG"
    );

    console.log(
        "Candidates:",
        boxes.length
    );

    console.log(
        "After NMS:",
        selectedIndices.length
    );

    console.log(
        "================================"
    );


    const finalBoxes = [];
    const finalScores = [];
    const finalLandmarks = [];

    for (
        const index of selectedIndices
    ) {
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