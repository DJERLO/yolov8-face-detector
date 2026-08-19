/**
 * Benchmark a FaceDetector instance.
 *
 * @param {FaceDetector} detector
 * @param {HTMLImageElement} image
 * @param {number} iterations
 */
export async function benchmarkDetector(
    detector,
    image,
    iterations = 10
) {
    const times = [];

    // Warmup
    await detector.detect(image);

    for (let i = 0; i < iterations; i++) {
        const start = performance.now();

        const results = await detector.detect(image);

        const elapsed =
            performance.now() - start;

        times.push({
            elapsed,
            detections: results.boxes.length
        });
    }

    const values = times.map(
        result => result.elapsed
    );

    const average =
        values.reduce((a, b) => a + b, 0) /
        values.length;

    const min =
        Math.min(...values);

    const max =
        Math.max(...values);

    return {
        iterations,
        average,
        min,
        max,
        results: times
    };
}