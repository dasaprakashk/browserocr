1. Cd to current directory
2. Create python venv and activate
3. Install requirements.txt
4. run : 'python server.py'
5. Open http://localhost:{PORT} 

Docker (local)
1. Build image:
	docker build -t browser-ocr:latest .
2. Run container:
	docker run --rm -p 8082:8082 browser-ocr:latest
3. Open:
	http://localhost:8082

Public container deploy (Render/Railway/Fly/Cloud Run)
1. Push this repo to GitHub.
2. Create a new service from this repo.
3. If platform asks for start command, use:
	python server.py
4. Ensure container/service exposes env PORT (already supported in server.py).
5. Open the generated HTTPS URL.


- 'models' folder contains the model onnx files (huggingface : monkt/paddleocr-onnx)
- 'sample_images' can be used to test in the browser
-  browser inspect element, under console, we can check if inference is running on webgl or cpu