Command to build the application. PLease remeber to change the project name and application name

gcloud builds submit --tag gcr.io/medical-app-319014/med-app  --project=medical-app-319014
Command to deploy the application

gcloud run deploy --image gcr.io/medical-app-319014/med-app  --platform managed  --project=medical-app-319014> --allow-unauthenticated