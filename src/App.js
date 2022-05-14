import "./App.css";
import React, { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";

const videoConstraints = {
    height: 1080,
    width: 1920,
    maxWidth: "100vw",
    facingMode: "environment",
};
const yolov5s_model_url = "yolov5s/model.json";
const yolov5n_model_url = "yolov5n/model.json";
const names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

function App() {
    const webcamRef = useRef(null);
    const [model, setModel] = useState(null);
    const [videoWidth, setVideoWidth] = useState(960);
    const [videoHeight, setVideoHeight] = useState(640);

    const loadModel = async () => {
        /** @type {tf.GraphModel} */
        const loadedModel = await tf.loadGraphModel(yolov5n_model_url);
        setModel(loadedModel);
        console.log("Model loaded.");
        return loadedModel;
    };

    const onCapture = async (model) => {
        // console.log("Capturing");
        // Check data is available
        if (typeof webcamRef.current !== "undefined" && webcamRef.current !== null && webcamRef.current.video.readyState === 4 && model !== null) {
            // Get Video Properties
            /** @type {HTMLVideoElement} */
            const video = webcamRef.current.video;
            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;

            // Set video width
            webcamRef.current.video.width = videoWidth;
            webcamRef.current.video.height = videoHeight;

            const cnvs = document.getElementById("myCanvas");
            cnvs.width = videoWidth;
            cnvs.height = videoHeight;
            // cnvs.style.position = "absolute";

            const ctx = cnvs.getContext("2d");
            ctx.clearRect(0, 0, webcamRef.current.video.videoWidth, webcamRef.current.video.videoHeight);

            /** @type {tf.Tensor3D} */
            const rawImgTensor = tf.browser.fromPixels(video);
            // console.log(`rawImgTensor shape: ${rawImgTensor.shape}`);

            /** @type {tf.GraphModel} */
            const [inputTensorWidth, inputTensorHeight] = model.inputs[0].shape.slice(1, 3); // [640, 640]
            const inputTensor = tf.tidy(() => {
                return tf.image.resizeBilinear(rawImgTensor, [inputTensorWidth, inputTensorHeight]).div(255.0).expandDims(0);
            });
            // console.log(`inputTensor shape: ${inputTensor.shape}`);
            let startTime = performance.now();
            model
                .executeAsync(inputTensor)
                .then((res) => {
                    // Font options.
                    const font = "16px sans-serif";
                    ctx.font = font;
                    ctx.textBaseline = "top";

                    const [boxes, scores, classes, valid_detections] = res;
                    const boxes_data = boxes.dataSync();
                    const scores_data = scores.dataSync();
                    const classes_data = classes.dataSync();
                    const valid_detections_data = valid_detections.dataSync()[0];

                    let i;
                    for (i = 0; i < valid_detections_data; ++i) {
                        let [x1, y1, x2, y2] = boxes_data.slice(i * 4, (i + 1) * 4);
                        x1 *= videoWidth;
                        x2 *= videoWidth;
                        y1 *= videoHeight;
                        y2 *= videoHeight;
                        const width = x2 - x1;
                        const height = y2 - y1;
                        const klass = names[classes_data[i]];
                        const score = scores_data[i].toFixed(2);

                        // Draw the bounding box.
                        ctx.strokeStyle = "#00FFFF";
                        ctx.lineWidth = 4;
                        ctx.strokeRect(x1, y1, width, height);

                        // Draw the label background.
                        ctx.fillStyle = "#00FFFF";
                        const textWidth = ctx.measureText(klass + ":" + score).width;
                        const textHeight = parseInt(font, 10); // base 10
                        ctx.fillRect(x1, y1, textWidth + 4, textHeight + 4);
                    }
                    for (i = 0; i < valid_detections_data; ++i) {
                        let [x1, y1, ,] = boxes_data.slice(i * 4, (i + 1) * 4);
                        x1 *= videoWidth;
                        y1 *= videoHeight;
                        const klass = names[classes_data[i]];
                        const score = scores_data[i].toFixed(2);

                        // Draw the text last to ensure it's on top.
                        ctx.fillStyle = "#000000";
                        ctx.fillText(klass + ":" + score, x1, y1);
                    }

                    let endTime = performance.now();
                    // console.log(`Took ${endTime - startTime} milliseconds`);
                    return res;
                })
                .then((res) => {
                    let i = 0;
                    const len = res.length;
                    while (i < len) {
                        tf.dispose(res[i]);
                        i++;
                    }
                })
                .finally(() => {
                    tf.dispose(rawImgTensor);
                    tf.dispose(inputTensor);
                });
            // console.dir(`numTensors: ${tf.memory().numTensors}`);
        }
    };

    /* 
    Run only once
     */
    useEffect(() => {
        // console.log(tfgl.version_webgl);
        // console.log(tf.getBackend());
        // tfgl.webgl.forceHalfFloat();
        // var maxSize = tfgl.webgl_util.getWebGLMaxTextureSize(tfgl.version_webgl);
        // console.log(maxSize);
        tf.ready()
            .then((_) => {
                tf.enableProdMode();
                console.log("tfjs is ready");
            })
            .then(loadModel)
            .then((loadedModel) => {
                console.log("Test model is loaded");
                setInterval(onCapture, 333, loadedModel);
            });
    }, []);

    // let supportedConstraints = navigator.mediaDevices.getSupportedConstraints();
    // console.log(supportedConstraints);
    return (
        <div className="App">
            <div style={{ position: "absolute", top: "0px", zIndex: "9999" }}>
                <canvas id="myCanvas" width={videoWidth} height={videoHeight} style={{ backgroundColor: "transparent" }} />
            </div>
            <div style={{ position: "absolute", top: "0px" }}>
                <Webcam
                    audio={false}
                    id="img"
                    ref={webcamRef}
                    // width={640}
                    screenshotQuality={1}
                    screenshotFormat="image/jpeg"
                    videoConstraints={videoConstraints}
                />
            </div>
        </div>
    );
}

export default App;
