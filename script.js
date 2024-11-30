// Copyright 2023 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//      http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;
const demosSection = document.getElementById("demos");
const imageBlendShapes = document.getElementById("image-blend-shapes");
const videoBlendShapes = document.getElementById("video-blend-shapes");
let faceLandmarker;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoWidth = 480;
// Before we can use FaceLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
async function createFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      delegate: "GPU"
    },
    outputFaceBlendshapes: true,
    runningMode,
    numFaces: 1
  });
  demosSection.classList.remove("invisible");
}
createFaceLandmarker();
/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
// Check if webcam access is supported.
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}
// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}
const IRIS_DIAMETER_MM = 11.7;
var trackingLaterality = "OS";
var focalLengthPx = NaN;
var calibrateFlag = false;
// Load default values from HTML attribute
var calibrationDistanceMm = 10 * parseFloat(document.getElementById('calibrationDistanceCm').value);
var warningDistanceMm = 10 * parseFloat(document.getElementById('warningDistanceCm').value);
function switchLaterality() {
  trackingLaterality = trackingLaterality === "OS" ? "OD" : "OS";
  focalLengthPx = NaN;
}
function enableCalibration() {
  focalLengthPx = NaN;
  if (webcamRunning !== true) {
    alert("WEBCAM NOT RUNNING!");
    return;
  }
  calibrateFlag = true;
}
document.getElementById("switchLateralityButton").addEventListener("click", switchLaterality);
document.getElementById("calibrateButton").addEventListener("click", enableCalibration);
document.getElementById("calibrateButton").style.backgroundColor = "#dd0000";
document.getElementById("calibrationDistanceCm").addEventListener("change", function(event){
  let valid = event.target.value !== '' && !isNaN(event.target.value);
  event.target.style.backgroundColor = valid ? null : "#ffcccc";
  if (valid) {
    calibrationDistanceMm = 10 * parseFloat(event.target.value);
  } else {
    calibrationDistanceMm = NaN;
  }
});
document.getElementById("warningDistanceCm").addEventListener("change", function(event){
  let valid = event.target.value !== '' && !isNaN(event.target.value);
  event.target.style.backgroundColor = valid ? null : "#ffcccc";
  if (valid) {
    warningDistanceMm = 10 * parseFloat(event.target.value);
  } else {
    warningDistanceMm = NaN;
  }
});

// Enable the live webcam view and start detection.
function enableCam(event) {
  if (!faceLandmarker) {
    alert("Wait! faceLandmarker not loaded yet.");
    return;
  }
  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE PREDICTIONS";
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE PREDICTIONS";
  }
  // getUsermedia parameters.
  const constraints = {
    video: {
      width: { ideal: 3840 },
      height: { ideal: 2160 }
    }
  };
  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}
// Depth vars init here
// Iris landmark indices
let IRIS_LANDMARK_IDXS = {
  LEFT_CENTER: 473,
  LEFT_LATERAL: 474,
  LEFT_SUPERIOR: 475,
  LEFT_MEDIAL: 476,
  LEFT_INFERIOR: 477,
  RIGHT_CENTER: 468,
  RIGHT_LATERAL: 469,
  RIGHT_SUPERIOR: 470,
  RIGHT_MEDIAL: 471,
  RIGHT_INFERIOR: 472
};
function calculateIrisSizePx(imgW, imgH, ptA, ptB, ptC, ptD) {
  let d1x = (ptA.x - ptB.x) * imgW;
  let d1y = (ptA.y - ptB.y) * imgH;
  let meas1 = Math.sqrt(d1x * d1x + d1y * d1y);
  let d2x = (ptC.x - ptD.x) * imgW;
  let d2y = (ptC.y - ptD.y) * imgH;
  let meas2 = Math.sqrt(d2x * d2x + d2y * d2y);
  let irisSizePx = (meas1 + meas2) / 2.0;
  return irisSizePx;
}
let lastVideoTime = -1;
let results = undefined;
const drawingUtils = new DrawingUtils(canvasCtx);

let frameCount = 0;
let lastTime = performance.now();

async function predictWebcam() {
  const ratio = video.videoHeight / video.videoWidth;
  video.style.width = videoWidth + "px";
  video.style.height = videoWidth * ratio + "px";
  canvasElement.style.width = videoWidth + "px";
  canvasElement.style.height = videoWidth * ratio + "px";
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  // Now let's start detecting the stream.
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await faceLandmarker.setOptions({ runningMode: runningMode });
  }
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    results = faceLandmarker.detectForVideo(video, startTimeMs);
  }
  if (results.faceLandmarks) {
    for (const landmarks of results.faceLandmarks) {
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_TESSELATION,
        { color: "#C0C0C070", lineWidth: 1 }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
        { color: "#E0E0E0" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
        { color: "#E0E0E0" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
        { color: "#E0E0E0" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
        { color: "#E0E0E0" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
        { color: "#E0E0E0" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LIPS,
        { color: "#E0E0E0" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
        { color: trackingLaterality === "OD" ? "#FF3030" : "#E0E0E0" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
        { color: trackingLaterality === "OS" ? "#FF3030" : "#E0E0E0" }
      );
      // Depth calculation here
      let leftIrisSizePx = calculateIrisSizePx(
        video.videoWidth,
        video.videoHeight,
        landmarks[IRIS_LANDMARK_IDXS.LEFT_MEDIAL],
        landmarks[IRIS_LANDMARK_IDXS.LEFT_LATERAL],
        landmarks[IRIS_LANDMARK_IDXS.LEFT_SUPERIOR],
        landmarks[IRIS_LANDMARK_IDXS.LEFT_INFERIOR]
      );
      let rightIrisSizePx = calculateIrisSizePx(
        video.videoWidth,
        video.videoHeight,
        landmarks[IRIS_LANDMARK_IDXS.RIGHT_MEDIAL],
        landmarks[IRIS_LANDMARK_IDXS.RIGHT_LATERAL],
        landmarks[IRIS_LANDMARK_IDXS.RIGHT_SUPERIOR],
        landmarks[IRIS_LANDMARK_IDXS.RIGHT_INFERIOR]
      );
      let irisSizePx =
        trackingLaterality === "OS" ? leftIrisSizePx : rightIrisSizePx;

      // Run calibration if needed
      if (calibrateFlag) {
        focalLengthPx = irisSizePx * calibrationDistanceMm / IRIS_DIAMETER_MM;
        calibrateFlag = false;
      }

      let currentDepthMm = focalLengthPx * IRIS_DIAMETER_MM / irisSizePx;

      document.getElementById("irisWidthPx").innerText =
        "Iris size: " + irisSizePx.toFixed(2) + " px";
      document.getElementById("videoRes").innerText =
        "Video resolution: " + video.videoWidth + "x" + video.videoHeight + " px";
      document.getElementById("focalLengthPx").innerText =
        "Focal Length: " + focalLengthPx.toFixed(1) + " px";
      document.getElementById("currentDepthCm").innerText =
        "Depth: " + (currentDepthMm / 10).toFixed(1) + " cm";
      document.getElementById("calibrationSizePx").innerText =
        "Iris size @ calibration: " + (focalLengthPx * IRIS_DIAMETER_MM / calibrationDistanceMm).toFixed(2) + " px";

      if (isNaN(focalLengthPx)) {
        document.body.style.backgroundColor = null;
      } else {
        if (currentDepthMm < warningDistanceMm) {
          document.body.style.backgroundColor = "#faa";
        } else {
          document.body.style.backgroundColor = null;
        }
      }

      // Increment frame count
      frameCount++;

      // Update FPS calculation every second
      const currentTime = performance.now();
      if (currentTime - lastTime >= 2000) {

        // Update rolling average FPS
        const avgFps = frameCount / ((currentTime - lastTime) / 1000.0);

        // Display averaged FPS on the webpage
        document.getElementById("fpsDisplay").innerText =
          "Avg. FPS: " + avgFps.toFixed(2);

        // Reset metrics
        frameCount = 0;
        lastTime = currentTime;
      }
    }
  }
  // drawBlendShapes(videoBlendShapes, results.faceBlendshapes);
  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}
function drawBlendShapes(el, blendShapes) {
  if (!blendShapes.length) {
    return;
  }
  let htmlMaker = "";
  blendShapes[0].categories.map((shape) => {
    htmlMaker += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">${
          shape.displayName || shape.categoryName
        }</span>
        <span class="blend-shapes-value" style="width: calc(${
          +shape.score * 100
        }% - 120px)">${(+shape.score).toFixed(4)}</span>
      </li>
    `;
  });
  el.innerHTML = htmlMaker;
}
