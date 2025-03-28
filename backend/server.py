import cv2
import numpy as np
import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from skimage.feature import CENSURE, BRIEF
from sklearn.cluster import DBSCAN
from keras.models import load_model

app = Flask(__name__)
CORS(app)  # Allow CORS for all origins (for development)

# Load Pretrained VGG16 Model
model = load_model("D:/AI/Karthik/001-Forgery-Detection-1/backend/models/sucessForgery.keras")
print("Model loaded successfully!")

# Ensure temporary directory exists for image storage
TEMP_DIR = "./tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

def keypoint_image_detection(image):
    """ Detect keypoints, cluster them, compute match ratio, and return an image with matches drawn. """
    try:
        print("Processing image...")  
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detector = CENSURE()
        detector.detect(img_gray)
        
        if len(detector.keypoints) == 0:
            print("No keypoints found.")
            return None, None  

        keypoints = [cv2.KeyPoint(float(kp[1]), float(kp[0]), 1) for kp in detector.keypoints]

        brief = BRIEF()
        brief.extract(img_gray, detector.keypoints)

        if brief.descriptors is None:
            print("No descriptors extracted.")
            return None, None  

        descriptors = np.array(brief.descriptors)

        keypoint_coords = np.array([kp.pt for kp in keypoints])

        dbscan = DBSCAN(eps=20, min_samples=5)
        clusters = dbscan.fit_predict(keypoint_coords)

        clustered_keypoints = {}
        clustered_descriptors = {}

        for idx, cluster_id in enumerate(clusters):
            if cluster_id == -1:
                continue
            if cluster_id not in clustered_keypoints:
                clustered_keypoints[cluster_id] = []
                clustered_descriptors[cluster_id] = []
            clustered_keypoints[cluster_id].append(keypoints[idx])
            clustered_descriptors[cluster_id].append(descriptors[idx])

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches_between_clusters = []

        cluster_ids = list(clustered_keypoints.keys())
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                cluster1, cluster2 = cluster_ids[i], cluster_ids[j]

                desc1 = np.array(clustered_descriptors[cluster1], dtype=np.uint8)
                desc2 = np.array(clustered_descriptors[cluster2], dtype=np.uint8)

                if len(desc1) < 2 or len(desc2) < 2:
                    continue

                matches = bf.match(desc1, desc2)
                matches = sorted(matches, key=lambda x: x.distance)
                matches_between_clusters.extend([(cluster1, cluster2, match) for match in matches])

        match_ratio = len(matches_between_clusters) / len(keypoints) if len(keypoints) > 0 else 0

        img_matches = image.copy()
        for cluster1, cluster2, match in matches_between_clusters:
            pt1 = tuple(map(int, clustered_keypoints[cluster1][match.queryIdx].pt))
            pt2 = tuple(map(int, clustered_keypoints[cluster2][match.trainIdx].pt))
            cv2.line(img_matches, pt1, pt2, (0, 255, 0), 2)

        cv2.putText(img_matches, f"Match Ratio: {match_ratio:.2f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return img_matches, match_ratio

    except Exception as e:
        print(f"Error in keypoint detection: {e}")
        return None, None


def forgery_detection(image):
    """ Detect keypoints, apply FM-RANSAC filtering, and return a processed image with a forgery mask. """
    try:
        print("Processing image for forgery detection...")  
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detector = CENSURE()
        detector.detect(img_gray)
        
        keypoints = [cv2.KeyPoint(float(kp[1]), float(kp[0]), 1) for kp in detector.keypoints]

        brief = BRIEF()
        brief.extract(img_gray, detector.keypoints)

        if brief.descriptors is None:
            return None, "No descriptors found."

        descriptors = np.array(brief.descriptors)

        keypoint_coords = np.array([kp.pt for kp in keypoints])
        dbscan = DBSCAN(eps=20, min_samples=5)
        clusters = dbscan.fit_predict(keypoint_coords)

        clustered_keypoints = {}
        clustered_descriptors = {}

        for idx, cluster_id in enumerate(clusters):
            if cluster_id == -1:
                continue
            if cluster_id not in clustered_keypoints:
                clustered_keypoints[cluster_id] = []
                clustered_descriptors[cluster_id] = []
            clustered_keypoints[cluster_id].append(keypoints[idx])
            clustered_descriptors[cluster_id].append(descriptors[idx])

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches_between_clusters = []

        cluster_ids = list(clustered_keypoints.keys())
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                cluster1, cluster2 = cluster_ids[i], cluster_ids[j]
                desc1 = np.array(clustered_descriptors[cluster1], dtype=np.uint8)
                desc2 = np.array(clustered_descriptors[cluster2], dtype=np.uint8)

                if len(desc1) < 2 or len(desc2) < 2:
                    continue

                matches = bf.match(desc1, desc2)
                matches = sorted(matches, key=lambda x: x.distance)
                matches_between_clusters.extend([(cluster1, cluster2, match) for match in matches])

        pts1, pts2 = [], []
        for cluster1, cluster2, match in matches_between_clusters:
            pts1.append(clustered_keypoints[cluster1][match.queryIdx].pt)
            pts2.append(clustered_keypoints[cluster2][match.trainIdx].pt)

        pts1 = np.float32(pts1).reshape(-1, 1, 2)
        pts2 = np.float32(pts2).reshape(-1, 1, 2)

        if len(pts1) >= 4:
            _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
        else:
            mask = None

        forgery_mask = np.zeros_like(img_gray, dtype=np.uint8)
        if mask is not None:
            for idx, (cluster1, cluster2, match) in enumerate(matches_between_clusters):
                if mask[idx]:
                    pt = tuple(map(int, clustered_keypoints[cluster1][match.queryIdx].pt))
                    cv2.circle(forgery_mask, pt, 10, 255, thickness=-1)

        overlay = cv2.addWeighted(img_gray, 0.7, forgery_mask, 0.3, 0)
        return overlay, None

    except Exception as e:
        print(f"Error in forgery detection: {e}")
        return None, str(e)

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode("utf-8")

@app.route('/keypoint', methods=['POST'])
def key_point_approach():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    processed_image, match_ratio = keypoint_image_detection(image)
    forgery_image, _ = forgery_detection(image)

    return jsonify({
        "processed_image": image_to_base64(processed_image) if processed_image is not None else None,
        "match_ratio": match_ratio,
        "forgery_image": image_to_base64(forgery_image) if forgery_image is not None else None
    })



def vgg_prediction(image):
    """ Step 6: Process the image using VGG16 model and return the prediction """
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = 1 if prediction[0][0] > 0.5 else 0

    print(f"VGG Model Prediction: {'Tampered' if predicted_class == 1 else 'Authentic'}")

    # Convert prediction to probability
    p_model = 0.9 if predicted_class == 1 else 0.1
    return p_model


@app.route('/vggmodel', methods=['POST'])
def vgg_model():
    """ Handle image upload, process it in memory, and return the result """

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Convert file to OpenCV image (without saving)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400


        # Step 6: VGG16 Model Prediction
        p_model = vgg_prediction(image)

        # key_point = "Tampered Image ❌" if match_ratio > 0.7 else "Authentic Image ✅"
        model_result  = "Tampered Image ❌" if p_model > 0.5 else "Authentic Image ✅"

        print(model_result,'model result')

        return jsonify({
            "vgg_prediction": model_result,     
            "vgg_result":p_model   
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def keypoint_detection(image):
    """ Step 1-5: Process the image using Keypoint Detection and return match ratio """

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Keypoint Detection using CenSurE
    detector = CENSURE()
    detector.detect(gray)
    keypoints = [cv2.KeyPoint(float(kp[1]), float(kp[0]), 1) for kp in detector.keypoints]

    # Step 2: Descriptor Extraction using BRIEF
    brief = BRIEF()
    brief.extract(gray, detector.keypoints)

    if brief.descriptors is not None:
        valid_keypoints = []
        valid_descriptors = []
        for i in range(len(brief.descriptors)):
            valid_keypoints.append(keypoints[i])
            valid_descriptors.append(brief.descriptors[i])
        keypoints = valid_keypoints
        descriptors = np.array(valid_descriptors)
    else:
        return 0  # No keypoints found, match ratio is 0

    # Step 3: Cluster Keypoints using DBSCAN
    keypoint_coords = np.array([kp.pt for kp in keypoints])
    dbscan = DBSCAN(eps=20, min_samples=5)
    clusters = dbscan.fit_predict(keypoint_coords)

    clustered_keypoints = {}
    clustered_descriptors = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id == -1:
            continue
        if cluster_id not in clustered_keypoints:
            clustered_keypoints[cluster_id] = []
            clustered_descriptors[cluster_id] = []
        clustered_keypoints[cluster_id].append(keypoints[idx])
        clustered_descriptors[cluster_id].append(descriptors[idx])

    # Step 4: Match Keypoints Between Different Clusters
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_between_clusters = []

    cluster_ids = list(clustered_keypoints.keys())
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            cluster1 = cluster_ids[i]
            cluster2 = cluster_ids[j]
            desc1 = np.array(clustered_descriptors[cluster1], dtype=np.uint8)
            desc2 = np.array(clustered_descriptors[cluster2], dtype=np.uint8)
            if len(desc1) < 2 or len(desc2) < 2:
                continue
            matches = bf.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)
            matches_between_clusters.extend([(cluster1, cluster2, match) for match in matches])

    print(f"Total Matches Before RANSAC: {len(matches_between_clusters)}")

    # Step 5: Use Fundamental Matrix (FM) RANSAC to Filter Matches
    pts1, pts2 = [], []
    for cluster1, cluster2, match in matches_between_clusters:
        pts1.append(clustered_keypoints[cluster1][match.queryIdx].pt)
        pts2.append(clustered_keypoints[cluster2][match.trainIdx].pt)

    pts1 = np.float32(pts1).reshape(-1, 1, 2)
    pts2 = np.float32(pts2).reshape(-1, 1, 2)

    if len(pts1) >= 4:
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
        matchesMask = mask.ravel().tolist()
        total_matches_after_ransac = sum(matchesMask)
        match_ratio = total_matches_after_ransac / len(matchesMask) if len(matchesMask) > 0 else 0
    else:
        match_ratio = 0

    print(f"Match Ratio (After RANSAC): {match_ratio:.2f}")
    return match_ratio



@app.route('/uploadfile', methods=['POST'])
def upload_data():
    """ Handle image upload, process it in memory, and return the result """

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Convert file to OpenCV image (without saving)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Step 3: Keypoint Detection Method
        match_ratio = keypoint_detection(image)

        # Step 6: VGG16 Model Prediction
        p_model = vgg_prediction(image)

        key_point = "Tampered Image ❌" if match_ratio > 0.7 else "Authentic Image ✅"
        model_result  = "Tampered Image ❌" if p_model > 0.5 else "Authentic Image ✅"

        # Step 7: Compute Fusion Result
        fusion_result = 0.1 * match_ratio + 0.9 * p_model

        # Step 8: Final Classification
        result_text = "Tampered Image ❌" if fusion_result > 0.5 else "Authentic Image ✅"
        print(result_text,'result')

        return jsonify({
            "match_ratio": match_ratio,
            "match_result": key_point,
            "vgg_prediction": p_model,
            "vgg_result": model_result,
            "fusion_result": result_text,
            "final_prediction": round(fusion_result, 2)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

