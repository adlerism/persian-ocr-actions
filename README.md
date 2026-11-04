# Persian PaddleOCR — GitHub Actions test

This starter project tests a fine-tuned Persian PP-OCRv5 Mobile recognition model on a GitHub-hosted CPU runner.

## What it does

1. Starts manually from the GitHub **Actions** tab.
2. Downloads the model ZIP from a public Google Drive share link.
3. Checks and extracts the ZIP.
4. Downloads one test image from a direct public HTTP(S) image URL.
5. Runs PaddleOCR on CPU, using the standard PP-OCRv5 mobile text detector and the custom recognition model.
6. Publishes the recognized text in the workflow summary and as downloadable artifacts.

## Important model requirement

The ZIP must contain an **exported inference model**, including:

- `inference.yml`
- `inference.pdiparams`
- `inference.json` or `inference.pdmodel`

A training checkpoint such as `best_accuracy.pdparams` by itself is not enough. It must first be exported for inference.

The workflow currently assumes that the custom model is a PP-OCRv5 Mobile **recognition** model. If the ZIP contains a different directory layout or model format, adjust `locate_recognition_model()` and the PaddleOCR initialization after inspecting the ZIP file list.

## Run the test

1. Create a new GitHub repository.
2. Upload all files from this project, preserving the `.github/workflows/` folder.
3. In Google Drive, set the model ZIP sharing to **Anyone with the link — Viewer**.
4. Open **Actions**.
5. Select **Manual Persian OCR test**.
6. Click **Run workflow**.
7. Paste the Google Drive share URL and a direct public URL to a test image.
8. After the run finishes, open its summary to see the text. Download `persian-ocr-result` under Artifacts for TXT and JSON outputs.

A normal Google Drive preview URL is not necessarily a direct image URL. Use an image URL that returns the image bytes directly.

## Limitations

- GitHub-hosted runners are temporary; this is a manual test workflow, not a continuously running Telegram bot.
- The model is downloaded on every run.
- Public Google Drive links may be subject to quota or download-confirmation pages.
- Never commit Telegram bot tokens or other secrets to the repository. Telegram integration will be added after validating model inference.
