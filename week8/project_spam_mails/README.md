# Spam Email Classifier (Streamlit Version)

This repository contains a Streamlit-based spam email classification application developed as part of the AIO2025 course. The application uses a K-Nearest Neighbors (KNN) classifier with FAISS for efficient embedding-based classification, alongside a TF-IDF baseline for comparison, integrated with the Google Gmail API for real-time email processing. This README provides detailed instructions for downloading the repository, configuring the Google API, and using the application via the Streamlit web interface.

**Note**: This README is tailored for the Streamlit version (v3.0) of the application. Previous versions focused on command-line usage; ensure you download the correct version using the specified tag. The Streamlit app offers an interactive dashboard for data analysis, model evaluation, and email fetching/classification.

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Downloading the Repository](#downloading-the-repository)
- [Configuring Google API](#configuring-google-api)
- [Usage](#usage)
  - [Running the Streamlit App](#running-the-streamlit-app)
  - [App Features](#app-features)
  - [Important Notes](#important-notes)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Future Development](#future-development)
- [Recent Updates (v3.0)](#recent-updates-v30)

---

## Prerequisites

Before running the application, ensure you have the following installed:

- **Python 3.10+**
- **Required Python Packages**:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `transformers`
  - `faiss-cpu` (or `faiss-gpu` if GPU is available)
  - `google-api-python-client`
  - `google-auth-oauthlib`
  - `nltk`
  - `matplotlib`
  - `seaborn`
  - `streamlit` (for the web interface)
  - `plotly` (for interactive charts)
  - `torch` (for embedding generation)

Install dependencies using:
```bash
pip install -r requirements.txt
```
Note: A `requirements.txt` file should be created in the project directory with the listed packages. If not present, generate it based on your environment:
```bash
pip freeze > requirements.txt
```

NLTK Data: Download required NLTK data by running:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Downloading the Repository

This project is part of a larger AIO2025 repository. To avoid downloading unnecessary course content, follow these steps to download only the `project_spam_mails` directory for the Streamlit version:

### Download Manually from GitHub:

1. Visit https://github.com/sonvt8/AIO2025/tags.
2. Locate the tag `v3.0-streamlit`.
3. Click on the tag name, then click **Browse code** to view the contents.
4. Navigate to the `week8/project_spam_mails` directory.
5. Click **Download ZIP** or use the **Code** button > **Download ZIP** to download the entire tag as a ZIP file.
6. Extract the ZIP file and move the `week8/project_spam_mails` folder to your desired location.
7. Optionally, delete the remaining extracted folders to keep only `project_spam_mails`.

Example:
```bash
mv AIO2025-week8-project_spam_mails-<tag>/week8/project_spam_mails .
rm -rf AIO2025-week8-project_spam_mails-<tag>
```

**Important**: Do not download the entire AIO2025 repository unless necessary, as it contains other course materials unrelated to this project. Ensure you use the correct tag to match the Streamlit version.

---

## Configuring Google API

To enable email classification via the Gmail API, follow these steps to configure authentication:

### Enable Gmail API:
- Go to the [Google Cloud Console](https://console.cloud.google.com/).
- Create a new project or select an existing one.
- Navigate to APIs & Services > Library, search for "Gmail API," and enable it.

### Create Credentials:
- Go to APIs & Services > Credentials.
- Click **Create Credentials** > **OAuth 2.0 Client IDs**.
- Configure the consent screen if prompted.
  - Choose External user type.
  - Select Individual (not Organization) as the scope.
- Fill in the required fields and save.
- Select **Desktop app** as the application type and download the JSON file (e.g., `credentials.json`).

### Place Credentials File:
```bash
mkdir -p cache/input
mv credentials.json cache/input/
```

### Authenticate:
Run the Streamlit app (`streamlit run app.py`) for the first time. A browser window will open to authenticate and generate a `token.json` file in `cache/input/`. This step is required for Gmail integration in the "Lấy Thư" page.

> Ensure `credentials.json` is kept secure and added to `.gitignore`. Without these files, Gmail fetching will fail.

---

## Usage

The spam email classification application is now accessible via a Streamlit web interface. The app provides an interactive dashboard for exploring the dataset, evaluating models, and fetching/classifying emails from Gmail.

### Running the Streamlit App

1. Navigate to the project directory:
   ```bash
   cd project_spam_mails
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser at `http://localhost:8501` (or the provided URL).

**Note**: The app runs locally. For deployment (e.g., on Streamlit Cloud or Heroku), upload the repository and configure secrets for `credentials.json` and `token.json`.

### App Features

The Streamlit app has four main pages:

- **🏠 Tổng quan (Overview)**: Displays key metrics (total emails, spam/ham ratios) and navigation buttons to other pages.
- **📊 Phân tích Dữ liệu (Data Analysis)**: Visualizes spam/ham distribution (bar chart) and embeddings via t-SNE (scatter plot on 1,000 samples).
- **📈 Đánh giá Bộ phân loại (Model Evaluation)**: Evaluates KNN (with multiple k-values) and TF-IDF models, showing metrics, confusion matrices, and comparisons. Results are cached for efficiency.
- **✉️ Lấy Thư (Fetch Emails)**: Fetches up to 10 unread emails from Gmail, classifies them (using selected method: KNN or TF-IDF), applies labels (`Inbox_Custom` or `Spam_Custom`), marks as read, saves locally (`inbox/` or `spam/`), and displays in folders. Select an email to view its content.

Navigation: Use buttons to switch pages. The app uses a dark theme for better UX.

### Important Notes

1. **Google API Files**: Ensure `cache/input/credentials.json` and `cache/input/token.json` are present and valid. If `token.json` is missing or expired, the app will prompt re-authentication via browser. Without these, the "Lấy Thư" page will fail to fetch emails.

2. **Precompute Embeddings for Better Experience**: Before running the Streamlit app, generate embeddings cache to avoid long waits or timeouts during analysis/evaluation:
   ```bash
   python main.py
   ```
   This creates embeddings in `cache/embeddings/` based on the dataset (`dataset/2cls_spam_text_cls.csv`). If the dataset changes (e.g., after merging emails), rerun with `--regenerate`:
   ```bash
   python main.py --regenerate
   ```
   **Why?** Embedding generation can take minutes (or longer on large datasets); caching ensures smooth app performance. Without cache, the app may timeout on first load.

3. **Other Notes**:
   - **Classifier Selection**: In "Lấy Thư," choose KNN (semantic similarity, slower but accurate) or TF-IDF (term frequency, faster baseline).
   - **Local Email Storage**: Fetched emails are saved as `.txt` in `inbox/` (ham) or `spam/` (spam). Use `--merge-emails` via command line to add them to the dataset:
     ```bash
     python main.py --merge-emails --regenerate
     ```
   - **OAuth Limitations**: Authentication uses OAuth for desktop apps (local browser flow). For cloud deployment, consider switching to a service account (update `email_handler.py` accordingly).
   - **Performance**: On CPU, large datasets may slow embeddings; use GPU if available (`faiss-gpu`).
   - **Logs**: Check `logs/spam_classifier.log` for detailed info on fetching, classification, and errors.
   - **Dataset Updates**: If modifying `2cls_spam_text_cls.csv`, always regenerate embeddings to avoid inconsistencies.

---

## File Structure

```
project_spam_mails/
├── cache/
│   ├── input/           # credentials.json, token.json
│   ├── output/          # plots and evaluation results
│   ├── embeddings/      # precomputed vectors
│   └── models/          # transformers & tokenizer
├── dataset/             # 2cls_spam_text_cls.csv
├── inbox/               # local ham emails
├── spam/                # local spam emails
├── logs/                # log files
├── app.py               # Streamlit app entry point
├── config.py
├── data_loader.py
├── email_handler.py
├── embedding_generator.py
├── evaluator.py
├── knn_classifier.py
├── main.py              # Command-line entry point (for training/caching)
├── spam_classifier.py
├── tfidf_classifier.py
└── README.md
```

---

## Troubleshooting

- **Streamlit Not Starting**: Ensure `streamlit` is installed (`pip install streamlit`). Run `streamlit hello` to test.
- **Embedding Timeout/Mismatch**: Run `python main.py --regenerate` to refresh cache. Check dataset size matches embeddings.
- **Gmail API Errors**: 
  - "Invalid credentials": Verify `credentials.json`.
  - "Token expired": Delete `token.json` and rerun the app.
  - "No emails fetched": Ensure unread emails exist in Gmail.
- **PyTorch Warnings**: Ignore `torch.classes` warnings if app runs; update `torch` and `transformers` if persistent.
- **Seaborn/Matplotlib Issues**: Use `plt.style.use('default')` if styles fail; ensure libraries are up-to-date.
- **Browser Not Opening for OAuth**: Run locally (not in headless mode); check firewall/port 0.
- **App Crashes on Load**: Clear Streamlit cache (`streamlit cache clear`) or check logs for dependency conflicts.

---

## License

This project is licensed under the MIT License.

---

## Future Development

Future updates may include advanced visualizations, multi-user support, or integration with other email providers. Contributions are welcome—fork and submit PRs!