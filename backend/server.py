from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import util

app = Flask(__name__)
load_dotenv()
CORS(app, resources={r"/*": {"origins": os.getenv("FRONTEND_URL")}})
util.load_saved_artifacts()

@app.route('/classify_image', methods=['POST'])
def classify_image():
    image_data = request.json['image_data']

    response = jsonify(util.classify_image(image_data))

    return response
  
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    app.run(port=5000)