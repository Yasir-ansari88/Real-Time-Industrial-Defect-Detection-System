##  Getting Started
 
### 1. Clone / open the project
 
```powershell
cd "E:\Real-Time Industrial Defect Detection System"
```
 
### 2. Build and start all services
 
```powershell
docker compose -f docker/docker-compose.yml up --build
```
 
This will start three containers:
 
| Service      | Container Name       | Port  |
|--------------|-----------------------|-------|
| API          | docker-api-1          | 8000  |
| Prometheus   | docker-prometheus-1   | 9090  |
| Grafana      | docker-grafana-1      | 3000  |
 
### 3. Verify everything is running
 
```powershell
docker ps
```
 
All three containers should show status `Up`.
 
## Accessing the Services
 
| Service          | URL                              | Notes                          |
|-------------------|-----------------------------------|---------------------------------|
| API Docs (Swagger)| http://localhost:8000/docs        | Test the `/predict` endpoint    |
| Raw Metrics       | http://localhost:8000/metrics     | Prometheus-format metrics       |
| Prometheus UI     | http://localhost:9090             | Query engine & scrape targets   |
| Prometheus Targets| http://localhost:9090/targets     | Confirm API scrape status = UP  |
| Grafana           | http://localhost:3000/login       | Default login: `admin` / `admin`|

## Testing the API
 
1. Open http://localhost:8000/docs
2. Expand the `/predict` endpoint
3. Click **Try it out**
4. Upload a test image from the `data/` folder
5. Click **Execute** and inspect the response — bounding boxes, class names, and confidence scores will be returned

## Setting Up the Grafana Dashboard
 
1. Log in to Grafana (`admin` / `admin`)
2. Go to **Connections → Data sources → Add data source → Prometheus**
3. Set the URL to `http://prometheus:9090` (use the Docker service name, not `localhost`, since Grafana and Prometheus communicate over the internal Docker network)
4. Click **Save & Test**
5. Create a new dashboard and add panels using the following metrics:
| Panel                         | PromQL Query                                                                                          |
|--------------------------------|--------------------------------------------------------------------------------------------------------|
| Total Defect Predictions       | `defect_detection_predictions_total`                                                                   |
| Average Inference Time         | `rate(defect_detection_inference_seconds_sum[5m]) / rate(defect_detection_inference_seconds_count[5m])`|
| Prediction Rate (per second)   | `rate(defect_detection_predictions_total[5m])`                                                         |
 
## Real-Time / Load Testing
 
A helper script (`realtime_test.py`) is included to continuously send test images to the API and simulate real-time traffic:
 
```powershell
python realtime_test.py
```
 
This sends a random image from the `data/` folder to `/predict` every 2 seconds, logging the response time and detection count to the console — useful for watching the Grafana dashboard update live.

