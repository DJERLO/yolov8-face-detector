/**
 * Calculate Intersection over Union.
 *
 * @param {number[]} boxA
 * @param {number[]} boxB
 * @returns {number}
 */
export function calculateIoU(boxA, boxB) {
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
        areaA +
        areaB -
        interArea;

    if (union <= 0) {
        return 0;
    }

    return interArea / union;
}


/**
 * Non-Maximum Suppression.
 *
 * @param {number[][]} boxes
 * @param {number[]} scores
 * @param {number} iouThreshold
 * @returns {number[]}
 */
export function nms(
    boxes,
    scores,
    iouThreshold = 0.45
) {
    const order = Array.from(
        { length: boxes.length },
        (_, i) => i
    );

    order.sort(
        (a, b) =>
            scores[b] - scores[a]
    );

    const keep = [];

    while (order.length > 0) {
        const current = order[0];

        keep.push(current);

        const remaining = [];

        for (
            let j = 1;
            j < order.length;
            j++
        ) {
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