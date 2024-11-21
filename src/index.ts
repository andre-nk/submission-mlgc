import express from "express";
import multer from "multer";
import * as tf from "@tensorflow/tfjs-node";
import { v4 as uuidv4 } from "uuid";
import { Firestore } from "@google-cloud/firestore";
import { Storage } from "@google-cloud/storage";

import bodyParser from "body-parser";

const app = express();
const port = process.env.PORT || 8080;

// Middleware
app.use(bodyParser.json());

// Firebase Firestore initialization
const firestore = new Firestore();
const predictionsCollection = firestore.collection("predictions");

// Google Cloud Storage initialization
const storage = new Storage();
const bucketName = process.env.BUCKET_NAME || "cancer-prediction-model";

// Multer configuration for file upload
const upload = multer({
  limits: { fileSize: 1000000 }, // 1MB limit
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith("image/")) {
      cb(null, true);
    } else {
      cb(new Error("Invalid file type. Only images are allowed."));
    }
  },
});

// Load TensorFlow.js model from Cloud Storage
async function loadModel() {
  try {
    const bucket = storage.bucket(bucketName);
    const modelFile = bucket.file("model.json");
    const [exists] = await modelFile.exists();

    if (!exists) {
      throw new Error("Model file not found in Cloud Storage");
    }

    const modelPath = `gs://${bucketName}/model.json`;
    return await tf.loadGraphModel(modelPath);
  } catch (error) {
    console.error("Error loading model:", error);
    throw error;
  }
}

let cancerModel: tf.GraphModel;

// Prediction endpoint
app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        status: "fail",
        message: "No image uploaded",
      });
    }

    // Load model if not already loaded
    if (!cancerModel) {
      cancerModel = await loadModel();
    }

    // Preprocess image
    const imageBuffer = req.file.buffer;
    const tfImage = tf.node.decodeImage(imageBuffer);
    const resizedImage = tf.image.resizeBilinear(tfImage, [224, 224]);
    const normalizedImage = resizedImage.div(255.0).expandDims(0);

    // Perform prediction
    const prediction = (await cancerModel.predict(
      normalizedImage
    )) as tf.Tensor;
    const predictionValue = (prediction.dataSync()[0] * 100).toFixed(2);

    const result = parseFloat(predictionValue) > 50 ? "Cancer" : "Non-cancer";
    const suggestion =
      result === "Cancer"
        ? "Segera periksa ke dokter!"
        : "Penyakit kanker tidak terdeteksi.";

    const responseId = uuidv4();
    const createdAt = new Date().toISOString();

    // Save prediction to Firestore
    await predictionsCollection.doc(responseId).set({
      id: responseId,
      result,
      suggestion,
      createdAt,
    });

    // Clean up tensors
    tfImage.dispose();
    resizedImage.dispose();
    normalizedImage.dispose();
    prediction.dispose();

    res.status(200).json({
      status: "success",
      message: "Model is predicted successfully",
      data: {
        id: responseId,
        result,
        suggestion,
        createdAt,
      },
    });
  } catch (error) {
    console.error("Prediction error:", error);
    res.status(400).json({
      status: "fail",
      message: "Terjadi kesalahan dalam melakukan prediksi",
    });
  }
});

// History endpoint
app.get("/predict/histories", async (req, res) => {
  try {
    const snapshot = await predictionsCollection.get();
    const histories = snapshot.docs.map((doc) => ({
      id: doc.id,
      history: doc.data(),
    }));

    res.status(200).json({
      status: "success",
      data: histories,
    });
  } catch (error) {
    console.error("Fetching histories error:", error);
    res.status(500).json({
      status: "fail",
      message: "Error fetching prediction histories",
    });
  }
});

// Global error handler
app.use(
  (
    err: Error,
    req: express.Request,
    res: express.Response,
    next: express.NextFunction
  ) => {
    if (err instanceof multer.MulterError) {
      return res.status(413).json({
        status: "fail",
        message: "Payload content length greater than maximum allowed: 1000000",
      });
    }
    next(err);
  }
);

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});

export default app;
