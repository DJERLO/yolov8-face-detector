import * as ort from "onnxruntime-web";

ort.env.wasm.wasmPaths ="/";

import { preprocess } from "./preprocess.js";
import { postprocess } from "./postprocess.js";

import {
    MODELS,
    MODEL_ALIASES
} from "../models/model-registry.js";


export class FaceDetector {

    constructor({
        model = "standard",
        publicPath = null,
        executionProviders = ["webgpu", "wasm"],
        targetSize = 640,
        confThreshold = 0.35,
        iouThreshold = 0.45
    } = {}) {

        // Custom model
        if (publicPath) {

            this.model = "custom";

            this.modelConfig = {
                name: "Custom Model",
                path: publicPath
            };

            this.modelPath =
                publicPath;
        }

        // Built-in model
        else {

            const modelKey =
                MODEL_ALIASES[model] ?? model;

            const modelConfig =
                MODELS[modelKey];

            if (!modelConfig) {
                throw new Error(
                    `Unknown face detection model: "${model}"`
                );
            }

            this.model =
                modelKey;

            this.modelConfig =
                modelConfig;

            this.modelPath =
                modelConfig.path;
        }

        this.executionProviders =
            executionProviders;

        this.targetSize =
            targetSize;

        this.confThreshold =
            confThreshold;

        this.iouThreshold =
            iouThreshold;

        this.session = null;
    }


    /**
     * Load the ONNX model.
     *
     * @returns {Promise<void>}
     */
    async load() {

        if (this.session) {
            return;
        }

        console.log(
            `Loading ${this.modelConfig.name}...`
        );

        this.session =
            await ort.InferenceSession.create(
                this.modelPath,
                {
                    executionProviders:
                        this.executionProviders
                }
            );

        console.log(
            "Model:",
            this.modelConfig.name
        );

        console.log(
            "Model path:",
            this.modelPath
        );

        console.log(
            "Execution providers:",
            this.executionProviders
        );

        console.log(
            "WebGPU available:",
            !!navigator.gpu
        );

        if (navigator.gpu) {

            const adapter =
                await navigator.gpu.requestAdapter();

            console.log(
                "WebGPU adapter:",
                adapter
            );

            if (
                adapter &&
                adapter.requestAdapterInfo
            ) {
                console.log(
                    "WebGPU adapter info:",
                    await adapter.requestAdapterInfo()
                );
            }
        }

        console.log(
            "Model loaded"
        );

        console.log(
            "Inputs:",
            this.session.inputNames
        );

        console.log(
            "Outputs:",
            this.session.outputNames
        );

        console.log(
            "Input metadata:",
            this.session.inputMetadata
        );

        console.log(
            "Output metadata:",
            this.session.outputMetadata
        );
    }


    /**
     * Run face detection.
     *
     * @param {HTMLImageElement|HTMLCanvasElement|ImageBitmap} image
     * @returns {Promise<{
     *   boxes: number[][],
     *   scores: number[],
     *   landmarks: number[][]
     * }>}
     */
    async detect(image) {

        if (!this.session) {
            throw new Error(
                "FaceDetector model is not loaded. Call load() first."
            );
        }

        const prepared =
            preprocess(
                image,
                this.targetSize
            );

        const feeds = {
            [this.session.inputNames[0]]:
                prepared.tensor
        };

        const output =
            await this.session.run(
                feeds
            );

        return postprocess(
            output,
            this.session.outputNames,
            prepared,
            {
                confThreshold:
                    this.confThreshold,

                iouThreshold:
                    this.iouThreshold
            }
        );
    }


    /**
     * Benchmark warm inference.
     *
     * Development/debugging utility.
     *
     * @param {HTMLImageElement|HTMLCanvasElement|ImageBitmap} image
     * @param {number} iterations
     */
    async benchmark(
        image,
        iterations = 10
    ) {

        if (!this.session) {
            throw new Error(
                "FaceDetector model is not loaded."
            );
        }

        const prepared =
            preprocess(
                image,
                this.targetSize
            );

        const feeds = {
            [this.session.inputNames[0]]:
                prepared.tensor
        };

        // Warm-up inference
        await this.session.run(
            feeds
        );

        const times = [];

        for (
            let i = 0;
            i < iterations;
            i++
        ) {

            const start =
                performance.now();

            await this.session.run(
                feeds
            );

            times.push(
                performance.now() - start
            );
        }

        const average =
            times.reduce(
                (sum, time) =>
                    sum + time,
                0
            ) / times.length;

        return {
            times,
            average,
            total:
                average * iterations
        };
    }
}