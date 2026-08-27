import { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  
  
  const [inferenceTime, setInferenceTime] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedModel, setSelectedModel] = useState("efficientnet");
  const totalPredictions = predictionHistory.length;
  const webcamLastPredictionTimeRef = useRef(0);
  const [isPredicting, setIsPredicting] = useState(false);

const averageConfidence =
  totalPredictions > 0
    ? predictionHistory.reduce(
        (sum, item) => sum + Number(item.confidence),
        0
      ) / totalPredictions
    : 0;

const averageInference =
  totalPredictions > 0
    ? predictionHistory.reduce(
        (sum, item) => sum + Number(item.inferenceTime),
        0
      ) / totalPredictions
    : 0;


  // Webcam
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);
  const predictionIntervalRef = useRef(null);
  const [cameraOn, setCameraOn] = useState(false);
  const predictionHistoryRef = useRef([]);
  const webcamProcessingRef = useRef(false);
  useEffect(() => {
  if (cameraOn && videoRef.current && streamRef.current) {
    videoRef.current.srcObject = streamRef.current;

    videoRef.current.play().catch((error) => {
      console.error("Video playback error:", error);
    });
  }
}, [cameraOn]);
useEffect(() => {
  if (cameraOn) {
    predictionIntervalRef.current = setInterval(() => {
      predictWebcamFrame();
    }, 1000);
  }

  return () => {
    if (predictionIntervalRef.current) {
      clearInterval(predictionIntervalRef.current);
      predictionIntervalRef.current = null;
    }
  };
}, [cameraOn]);

  // -----------------------------
  // IMAGE UPLOAD
  // -----------------------------
  const handleImageChange = (event) => {
    const file = event.target.files[0];

    if (!file) return;

    setSelectedImage(file);
    setPreview(URL.createObjectURL(file));
    setPrediction(null);
    setConfidence(null);
    setError("");
  };

  // -----------------------------
  // IMAGE PREDICTION
  // -----------------------------
  const handlePredict = async () => {
  if (!selectedImage) {
    setError("Please select an image first.");
    return;
  }

  setLoading(true);
  setError("");

  const formData = new FormData();

  formData.append("file", selectedImage);
  formData.append("model", selectedModel);
  setIsPredicting(true);

  try {
    const response = await fetch(
      "http://127.0.0.1:8000/predict",
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
  throw new Error("Prediction request failed.");
}

const data = await response.json();

console.log("PREDICTION DATA:", data);
setPrediction(data);

const newPrediction = {
  prediction: data.prediction,
  confidence: data.confidence,
  model: selectedModel,
  inferenceTime: data.inference_time,
  top_3: data.top_3,
  time: new Date().toLocaleTimeString(),
};

    predictionHistoryRef.current.unshift(newPrediction);

    if (predictionHistoryRef.current.length > 5) {
      predictionHistoryRef.current.pop();
    }

    setPredictionHistory([...predictionHistoryRef.current]);

    // Count how many times each sign appears
    const counts = {};

    predictionHistoryRef.current.forEach((item) => {
      counts[item.prediction] =
        (counts[item.prediction] || 0) + 1;
    });

    // Find the most frequent prediction
    const stablePrediction = Object.keys(counts).reduce(
      (a, b) => (counts[a] > counts[b] ? a : b)
    );

    // Get confidence values for the stable prediction
    const matchingResults =
      predictionHistoryRef.current.filter(
        (item) => item.prediction === stablePrediction
      );

    const averageConfidence =
      matchingResults.reduce(
        (sum, item) => sum + Number(item.confidence),
        0

        
      ) / matchingResults.length;

    // Display stable prediction
    setPrediction(stablePrediction);
    setConfidence(averageConfidence);

  } catch (error) {
    console.error("Prediction error:", error);

    setError(
      "Unable to connect to the prediction server. Make sure FastAPI is running."
    );
  } finally {
    setLoading(false);
  }
};

  // -----------------------------
  // START CAMERA
  // -----------------------------
  const predictWebcamFrame = async () => {

  if (webcamProcessingRef.current) {
    return;
  }
  if (
  webcamLastPredictionTimeRef.current &&
  Date.now() - webcamLastPredictionTimeRef.current < 1000
) {
  return;
}
  if (!videoRef.current || !canvasRef.current) {
    return;
  }

  const video = videoRef.current;

if (video.readyState !== 4) {
  return;
}

webcamProcessingRef.current = true;

const canvas = canvasRef.current;

canvas.width = video.videoWidth;
canvas.height = video.videoHeight;

const context = canvas.getContext("2d");

  context.drawImage(
    video,
    0,
    0,
    canvas.width,
    canvas.height
  );

  canvas.toBlob(async (blob) => {
  if (!blob) {
    webcamProcessingRef.current = false;
    return;
  }

    const formData = new FormData();

    formData.append(
      "file",
      blob,
      "webcam_frame.jpg"
    );

    formData.append(
      "model",
      selectedModel
    );

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("Webcam prediction failed");
      }

      const data = await response.json();
      webcamLastPredictionTimeRef.current = Date.now();

      setPrediction(data.prediction);
      setTopPredictions(data.top_3 || []);
      setConfidence(data.confidence);
      setInferenceTime(data.inference_time);
      
  setPredictionHistory((prev) => {
  if (
    prev.length > 0 &&
    prev[0].prediction === data.prediction
  ) {
    return prev;
  }

  return [
    {
      prediction: data.prediction,
      confidence: data.confidence,
      model: selectedModel,
      inferenceTime: data.inference_time,
      time: new Date().toLocaleTimeString(),
    },
    ...prev,
  ];
});
      


            } catch (error) {
      console.error(
        "Webcam prediction error:",
        error
      );
    } finally {
      webcamProcessingRef.current = false;
    }
  }, "image/jpeg");
};
  const startCamera = async () => {
  try {
    setError("");

    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });

    streamRef.current = stream;

    // Render the video element first
    setCameraOn(true);

  } catch (error) {
    console.error("Camera error:", error);

    setError(
      "Unable to access camera. Please allow camera permission."
    );
  }
};
const clearPredictionHistory = () => {
  predictionHistoryRef.current = [];
  setPredictionHistory([]);
};

  // -----------------------------
  // STOP CAMERA
  // -----------------------------
  const stopCamera = () => {

  if (predictionIntervalRef.current) {
    clearInterval(
      predictionIntervalRef.current
    );

    predictionIntervalRef.current = null;
  }

  if (streamRef.current) {
    streamRef.current
      .getTracks()
      .forEach((track) => track.stop());

    streamRef.current = null;
  }

  if (videoRef.current) {
    videoRef.current.srcObject = null;
  }

  setCameraOn(false);

  setPrediction(null);
  setConfidence(null);
  predictionHistoryRef.current = [];
  setPredictionHistory([]);
};

  return (
    <div className="app">

      {/* =========================
          HEADER
      ========================== */}
      <header className="header">
        <div className="header-content">
          <h1>Indian Sign Language</h1>
          <p>Deep Learning Based Recognition System</p>
        </div>
      </header>


      {/* =========================
          MAIN CONTENT
      ========================== */}
      <main className="container">

        {/* HERO SECTION */}
        <section className="hero">

          <h2>Indian Sign Language Recognition</h2>

          <p>
           Deep Learning-Based 26-Class Indian Sign Language Recognition
          </p>

        </section>


        {/* =========================
            IMAGE UPLOAD CARD
        ========================== */}
        <section className="card">

          <h2>Upload Sign Image</h2>


         {/* MODEL SELECTION */}
         <div className="model-selection">

         <h3>Select Deep Learning Model</h3>

        <div className="model-options">

        <label>
         <input
          type="radio"
          value="cnn"
          checked={selectedModel === "cnn"}
          onChange={(e) => setSelectedModel(e.target.value)}
        />

        <div className="model-info">
         <strong>CNN</strong>
         <span>Custom Convolutional Neural Network</span>
         <small>Accuracy: 99.91%</small>
        </div>
        </label>

    <label>
      <input
        type="radio"
        value="mobilenet"
        checked={selectedModel === "mobilenet"}
        onChange={(e) => setSelectedModel(e.target.value)}
      />

      <div className="model-info">
        <strong>MobileNetV2</strong>
        <span>Lightweight Deep Learning Model</span>
        <small>Accuracy: 99.98%</small>
      </div>
    </label>

    <label>
      <input
        type="radio"
        value="efficientnet"
        checked={selectedModel === "efficientnet"}
        onChange={(e) => setSelectedModel(e.target.value)}
      />

      <div className="model-info">
        <strong>EfficientNetB0</strong>
        <span>Efficient Image Classification Model</span>
        <small>Accuracy: 99.94%</small>
      </div>
    </label>
  </div>

</div>
<div className="model-comparison">

  <h3>Model Comparison</h3>

  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Accuracy</th>
        <th>Type</th>
      </tr>
    </thead>

    <tbody>
      <tr>
        <td>CNN</td>
        <td>99.91%</td>
        <td>Custom CNN</td>
      </tr>

      <tr>
        <td>MobileNetV2</td>
        <td>99.98%</td>
        <td>Lightweight</td>
      </tr>

      <tr>
        <td>EfficientNetB0</td>
        <td>99.94%</td>
        <td>Efficient CNN</td>
      </tr>
    </tbody>
  </table>

</div>

  

          {/* IMAGE UPLOAD */}
          <label className="upload-box">

            <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
            />

            <div className="upload-icon">
              📷
            </div>

            <span>
              Choose an image
            </span>

            <small>
              JPG, JPEG or PNG
            </small>

          </label>


          {/* IMAGE PREVIEW */}
          {preview && (
            <div className="preview-section">

              <h3>
                Image Preview
              </h3>

              <img
                src={preview}
                alt="Selected sign"
                className="preview-image"
              />

            </div>
          )}


          {/* PREDICT BUTTON */}
          <button
          className="predict-button"
          onClick={handlePredict}
          disabled={loading}
          >
          {loading ? "Predicting..." : "Predict Sign"}
          </button>


          {/* ERROR */}
          {error && (
            <p className="error">
              {error}
            </p>
          )}
        </section>

        


       {/* =========================
    IMAGE PREDICTION RESULT
========================= */}
{prediction && (
  <section className="result-card">

    <h2>Prediction Result</h2>

    <div className="result-content">

      <div className="prediction-box">
        <span>Predicted Sign</span>

        <strong>
          {prediction}
        </strong>
      </div>

      <div className="confidence-box">
        <span>Confidence</span>

        <strong>
          {Number(confidence).toFixed(2)}%
        </strong>

        <div className="confidence-bar">
          <div
            className="confidence-fill"
            style={{
              width: `${Math.min(Number(confidence), 100)}%`
            }}
          />
        </div>
      </div>

      <div className="model-result-info">
        <p>
          <strong>Model Used:</strong>{" "}
          {selectedModel}
        </p>
      </div>

    </div>

  </section>
)}
{/* =========================
    PREDICTION HISTORY
========================= */}
{predictionHistory.length > 0 && (
  <section className="history-card">

    <div className="history-header">
      <h2>Prediction History</h2>

      <button
        className="clear-history-button"
        onClick={clearPredictionHistory}
      >
        Clear History
      </button>
    </div>

    <div className="history-list">
      <div className="history-labels">
       <span>Sign</span>
       <span>Confidence</span>
       <span>Inference</span>
       <span>Model</span>
       <span>Time</span>
    </div>

      {predictionHistory.slice(0, 10).map((item, index) => (
        <div className="history-item" key={index}>

          <div>
           <strong>#{index + 1} {item.prediction}</strong>
          </div>

          <div>
            {Number(item.confidence).toFixed(2)}%
          </div>

          <span>
            Inference: {Number(item.inferenceTime).toFixed(2)} ms
          </span>

          <div>
            {item.model}
          </div>

          <div>
            {item.time}
          </div>

        </div>
      ))}

    </div>

  </section>
)}
{/* =========================
    PERFORMANCE METRICS
========================= */}
{predictionHistory.length > 0 && (
  <section className="metrics-section">

    <h2>Performance Metrics</h2>

    <div className="metrics-grid">

      <div className="metric-card">
       <span>Latest Inference</span>
       <strong>
       {predictionHistory.length > 0
        ? Number(predictionHistory[0].inferenceTime).toFixed(2)
        : "0.00"} ms
       </strong>
      </div>

      <div className="metric-card">
        <span>Average Confidence</span>
        <strong>
          {averageConfidence.toFixed(2)}%
        </strong>
      </div>

      <div className="metric-card">
        <span>Average Inference</span>
        <strong>
          {averageInference.toFixed(2)} ms
        </strong>
      </div>

    </div>

  </section>
)}
        {/* =========================
            LIVE CAMERA SECTION
        ========================== */}
        <section className="camera-section">

          <h2>
            Live Sign Recognition
          </h2>

          <p className="camera-description">
            Use your webcam to capture a hand sign.
          </p>


          {/* CAMERA DISPLAY */}
          <div className={`camera-status ${cameraOn ? "active" : "inactive"}`}>
  <span className="status-dot"></span>
  {cameraOn ? "Camera Active" : "Camera Off"}
</div>
<div className="camera-box">

  {cameraOn ? (
    <video
      ref={videoRef}
      autoPlay
      playsInline
      muted
      className="camera-video"
    />
  ) : (
    <div className="camera-placeholder">
      <div className="camera-icon">
        📷
      </div>

      <p>
        Camera is currently off
      </p>
    </div>
  )}

</div>
          <canvas
            ref={canvasRef}
            style={{ display: "none" }}
          />
          {cameraOn && prediction && (
  <div className="live-prediction">
    <h3>Live Prediction</h3>

    <div className="live-sign">
      {prediction}
    </div>

    <p>
      Confidence:{" "}
      <strong>
        {Number(confidence).toFixed(2)}%
      </strong>
    </p>
    <p className="session-count">
      Predictions this session:{" "}
     <strong>{predictionHistory.length}</strong>
    </p>
  </div>
)}


          {/* CAMERA BUTTONS */}
           <div className="camera-buttons">
  {!cameraOn ? (
    <button
      className="start-camera"
      onClick={startCamera}
    >
      Start Camera
    </button>
  ) : (
    <button
      className="stop-camera"
      onClick={stopCamera}
    >
      Stop Camera
    </button>
  )}
</div>

        </section>

      </main>


      {/* =========================
          FOOTER
      ========================== */}
      <footer>

        <p>
          Indian Sign Language Recognition
        </p>

        <span>
          CNN • MobileNetV2 • EfficientNetB0 • TensorFlow • FastAPI
        </span>

      </footer>

    </div>
  );
}


export default App;