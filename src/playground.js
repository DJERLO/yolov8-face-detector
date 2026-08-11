import { FaceDetector } from "./index.js";
import { benchmarkDetector } from "./benchmark/benchmark.js";

const benchmarkBtn =
    document.getElementById("benchmarkBtn");

const benchmarkOutput =
    document.getElementById("benchmarkOutput");

benchmarkBtn.addEventListener("click", async () => {
    if (!currentImage) {
        return;
    }

    benchmarkBtn.disabled = true;

    try {
        benchmarkOutput.textContent =
            "Benchmarking...";

        const result =
            await benchmarkDetector(
                detector,
                currentImage,
                50 // iterations
            );

        benchmarkOutput.textContent = `
            Iterations: ${result.iterations}
            Average: ${result.average.toFixed(2)} ms
            Min: ${result.min.toFixed(2)} ms
            Max: ${result.max.toFixed(2)} ms
        `.trim();

    } catch (error) {
        console.error(
            "BENCHMARK ERROR:",
            error
        );

        benchmarkOutput.textContent =
            "Benchmark failed.";
    } finally {
        benchmarkBtn.disabled = false;
    }
});

const imageUpload =
    document.getElementById("imageUpload");

const runBtn =
    document.getElementById("runBtn");

const statusDiv =
    document.getElementById("status");

const canvas =
    document.getElementById("outputCanvas");

const ctx =
    canvas.getContext("2d");

let currentImage = null;

const detector =
    new FaceDetector({
        model: "standard",
        executionProviders: ["webgpu"],
    });


async function initialize() {
    try {
        statusDiv.textContent =
            "Loading model...";

        await detector.load();

        statusDiv.textContent =
            "Model loaded successfully.";

    } catch (error) {
        console.error("MODEL LOAD ERROR:", error);

        statusDiv.textContent =
            "Failed to load model.";
    }
}


imageUpload.addEventListener(
    "change",
    (event) => {
        const file =
            event.target.files[0];

        if (!file) {
            return;
        }

        const reader =
            new FileReader();

        reader.onload =
            (event) => {
                const image =
                    new Image();

                image.onload =
                    () => {
                        currentImage =
                            image;

                        canvas.width =
                            image.width;

                        canvas.height =
                            image.height;

                        ctx.drawImage(
                            image,
                            0,
                            0
                        );

                        runBtn.disabled =
                            false;

                        statusDiv.textContent =
                            `Image loaded (${image.width}x${image.height}).`;
                    };

                image.src =
                    event.target.result;
            };

        reader.readAsDataURL(file);
    }
);


runBtn.addEventListener(
    "click",
    async () => {
        if (!currentImage) {
            return;
        }

        runBtn.disabled =
            true;

        try {
            statusDiv.textContent =
                "Running detection...";

            const results =
                await detector.detect(
                    currentImage
                );

            drawResults(
                currentImage,
                results
            );

            statusDiv.textContent =
                `Detection complete! Found ${results.boxes.length} face(s).`;

        } catch (error) {
            console.error(
                "INFERENCE ERROR:",
                error
            );

            statusDiv.textContent =
                "Detection failed.";

        } finally {
            runBtn.disabled =
                false;
        }
    }
);


function drawResults(
    image,
    results
) {
    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    ctx.drawImage(
        image,
        0,
        0
    );

    ctx.strokeStyle =
        "#22c55e";

    ctx.lineWidth =
        Math.max(
            2,
            image.width / 400
        );

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

        ctx.strokeRect(
            x1,
            y1,
            x2 - x1,
            y2 - y1
        );

        const landmarks =
            results.landmarks[i];

        if (
            landmarks &&
            landmarks.length >= 10
        ) {
            ctx.fillStyle =
                "#ef4444";

            for (
                let point = 0;
                point < 5;
                point++
            ) {
                const x =
                    landmarks[
                        point * 2
                    ];

                const y =
                    landmarks[
                        point * 2 + 1
                    ];

                ctx.beginPath();

                ctx.arc(
                    x,
                    y,
                    Math.max(
                        2,
                        image.width / 500
                    ),
                    0,
                    Math.PI * 2
                );

                ctx.fill();
            }
        }
    }
}


initialize();