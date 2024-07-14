# FastAPI Image Processing Application

This is a FastAPI application designed to process uploaded images and classify dental diseases using machine learning models.

## Project Structure

```
my-fastapi-app/
├── main.py
└── models/
│   ├── quadenum.pt
│   └── densenet121.h5
├── requirements.txt
├── README.md
```

## Setup and Deployment

### Requirements
- Python 3.x

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Radiance-Of-The-Smile/backend.git
   cd backend
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running Locally
To run the application locally, use the following command:
```bash
uvicorn main:app --reload
```
This will start the FastAPI server locally and reload automatically on code changes.

## Endpoints

- **GET /**: Health check endpoint
  - **Response**: `"The health check is successful!"`

- **POST /preprocess**: Endpoint to preprocess uploaded images
  - **Request**: Upload an image file
  - **Response**: 
    - `processed_image`: Base64 encoded string of the processed image
    - `anomalies`: List of detected anomalies with their labels

## File Descriptions

- **main.py**: The main FastAPI application file containing the endpoints and image processing logic.
- **models/quadenum.pt**: YOLO model for object detection.
- **models/densenet121.h5**: DenseNet model for disease classification.
- **requirements.txt**: List of required Python packages.

## Usage

1. Start the server using the local running command.
2. Use an API client (like Postman or curl) to send a GET request to the root endpoint to verify the server is running.
3. Use the POST `/preprocess` endpoint to upload an image and receive the processed image and detected anomalies.

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [Pillow](https://python-pillow.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)

## Contact

For any inquiries, please contact the project maintainers.