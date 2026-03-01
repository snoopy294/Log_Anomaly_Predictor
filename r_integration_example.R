# R Integration Example
# ---------------------
# This script demonstrates how to send log data from R to the Anomaly Detection API.

# Requirements:
# install.packages("httr")
# install.packages("readr")

library(httr)
library(readr)

# Configuration
API_URL <- "http://localhost:5000/api/upload_csv"
CSV_FILE_PATH <- "data/train_data.csv" # Replace with your CSV file

# Function to upload CSV
upload_logs <- function(file_path) {
  if (!file.exists(file_path)) {
    stop("File not found: ", file_path)
  }
  
  message("Uploading ", file_path, " to ", API_URL, "...")
  
  response <- POST(
    url = API_URL,
    body = list(file = upload_file(file_path))
  )
  
  if (status_code(response) == 200) {
    message("Success!")
    print(content(response))
  } else {
    message("Failed with status: ", status_code(response))
    print(content(response))
  }
}

# Example Usage:
# 1. Create a dummy dataframe and save it
dummy_data <- data.frame(
  timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
  entity_id = c("192.168.1.10", "192.168.1.11"),
  event_type = c("SSH_LOGIN", "HTTP_REQUEST"),
  dst_id = c("10.0.0.5", "10.0.0.6"),
  bytes = c(100, 500)
)
write_csv(dummy_data, "r_test_upload.csv")

# 2. Upload the file
upload_logs("r_test_upload.csv")
