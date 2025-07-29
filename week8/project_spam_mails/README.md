
# Spam Email Classifier (Command Line Version)

This repository contains a command-line-based spam email classification application developed as part of the AIO2025 course. The application uses a K-Nearest Neighbors (KNN) classifier with FAISS for efficient embedding-based classification, integrated with the Google Gmail API for real-time email processing. This README provides detailed instructions for downloading the repository, configuring the Google API, and using the application via command-line arguments.

**Note**: This README is tailored for the command-line version of the application. A future Streamlit-based version is planned, but this document applies to the current command-line implementation. Ensure you download the correct version using the specified tag.

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Downloading the Repository](#downloading-the-repository)
- [Configuring Google API](#configuring-google-api)
- [Usage](#usage)
  - [Command Line Arguments](#command-line-arguments)
  - [Examples](#examples)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Future Development](#future-development)
- [Recent Updates (v2.0)](#recent-updates-v20)

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

Install dependencies using:
```bash
pip install -r requirements.txt
```
Note: A requirements.txt file should be created in the project directory with the listed packages. If not present, generate it based on your environment:
```bash
pip freeze > requirements.txt
```

NLTK Data: Download required NLTK data by running:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## Downloading the Repository

This project is part of a larger AIO2025 repository. To avoid downloading unnecessary course content, follow these steps to download only the project_spam_mails directory for the command-line version:

### Download Manually from GitHub:

1. Visit https://github.com/sonvt8/AIO2025/tags.
2. Locate the tag `v2.0-command-line`.
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

**Important**: Do not download the entire AIO2025 repository unless necessary, as it contains other course materials unrelated to this project. Ensure you use the correct tag to match the command-line version.

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
Run the application with the `--run-email-classifier` flag for the first time. A browser window will open to authenticate and generate a `token.json` file in `cache/input/`.

> Ensure `credentials.json` is kept secure and added to `.gitignore`.

---

## Usage

The spam email classification application is controlled via command-line arguments executed through main.py. To ensure a smooth experience, follow the steps below in the recommended order. These steps account for the application's dependency on precomputed embeddings and VectorDatabase initialization to avoid errors such as missing embeddings or dataset mismatches.

### Step-by-Step Usage Instructions

#### Initial Setup and Training (First Run)

For the first run, execute main.py without any flags to train the model and generate embeddings for the dataset (2cls_spam_text_cls.csv). This step creates the necessary embeddings cache and initializes the FAISS VectorDatabase, which are required for subsequent operations.

```bash
python main.py
```

**What happens**: The application loads the dataset, preprocesses the text, generates embeddings using the transformer model (intfloat/multilingual-e5-base), trains the KNN classifier, and tests the pipeline with sample examples. The embeddings are saved in `cache/embeddings/` for reuse.  
**Why**: This ensures the embeddings and VectorDatabase are ready before performing advanced operations like merging emails or classifying new emails via Gmail API.  
**Output**: Logs in `logs/spam_classifier.log` and console output showing training progress and sample predictions.

---

#### Evaluate Model Performance

After the initial run, you can evaluate the model's performance using different k-values for the KNN classifier. Use the `--evaluate` flag to generate performance metrics (accuracy, precision, recall, F1-score) and visualizations (line plots, confusion matrices, and label distribution).

```bash
python main.py --evaluate
```

Optional: Specify custom k-values for evaluation (e.g., 1, 3, 7).

```bash
python main.py --evaluate --k-values "1,3,7"
```

**What happens**: The application loads the precomputed embeddings, splits the dataset into train/test sets, evaluates the model for each k-value, and saves results (including error analysis) in `cache/output/error_analysis.json` and visualizations in `cache/output/evaluation_summary.png`.  
**Why**: This step helps assess the model's effectiveness without regenerating embeddings, leveraging the cache from the first run.  
**Recommendation**: Do not use `--regenerate` unless the dataset changes (e.g., after merging emails), as it wastes time regenerating embeddings unnecessarily.

**Note**: If the dataset changes (e.g., after merging emails), use `--regenerate` to update embeddings:

```bash
python main.py --evaluate --regenerate
```

---

#### Merge Emails from Local Folders

If you have new emails in the `inbox/` or `spam/` directories, merge them into the dataset (`2cls_spam_text_cls.csv`) using the `--merge-emails` flag. This step requires embeddings to be regenerated afterward to maintain consistency.

```bash
python main.py --merge-emails --regenerate
```

**What happens**: The application reads `.txt` files from `inbox/` (labeled as ham) and `spam/` (labeled as spam), removes duplicates, appends them to the dataset, and regenerates embeddings to match the updated dataset.  
**Why**: Merging emails updates the dataset, but the embeddings cache must be refreshed to avoid a mismatch error (dataset size not matching cached embeddings).  
**Recommendation**: Always use `--regenerate` with `--merge-emails` to avoid embedding mismatch errors in subsequent runs. If omitted, the application will warn via logs, but combining them saves debugging time.

> Always use `--regenerate` with `--merge-emails` to avoid errors due to dataset-embedding inconsistency. Check `logs/spam_classifier.log` for merge statistics.

---

#### Run Real-Time Email Classification with Gmail API

To classify unread emails from your Gmail account and move them to custom labels (`Inbox_Custom` or `Spam_Custom`), use the `--run-email-classifier` flag. This step requires a trained model and embeddings from the first run.

```bash
python main.py --run-email-classifier
```

**What happens**: The application authenticates with Gmail API, fetches up to 10 unread emails every 30 seconds, classifies them as spam or ham, applies the appropriate label, marks them as read, and saves them locally in `inbox/` or `spam/` as `.txt` files.  
**Why**: This enables real-time email processing, but it relies on the precomputed embeddings and trained classifier from the initial run.  
**Recommendation**: Only run after training embeddings (first step), as missing embeddings will cause errors. Do not use `--regenerate` unless you want to update the model with new data.

> Ensure `cache/input/credentials.json` is present and authentication is complete. If embeddings are missing, run `python main.py` first.

**Stop the process**: Press Ctrl+C to safely stop the program. A log will confirm safe exit.

---

#### Regenerate Embeddings (Optional)

If you modify the dataset (e.g., after merging emails or manually editing `2cls_spam_text_cls.csv`), regenerate embeddings to ensure consistency. Use the `--regenerate` flag with any command:

```bash
python main.py --regenerate
```

**What happens**: The application deletes the existing embeddings cache and generates new ones for the current dataset.  
**Why**: To synchronize embeddings with the updated dataset, avoiding mismatch errors.  
**Recommendation**: Use only when the dataset changes; avoid overuse as it can be time-consuming (depending on dataset size, it may take minutes to hours). Combine with other commands like `--evaluate` or `--merge-emails` if needed.

---

### Command Line Arguments

- `--regenerate`: Force regeneration of embeddings (default: False). Use after dataset changes.
- `--run-email-classifier`: Enable real-time email classification via Gmail API (default: False).
- `--merge-emails`: Merge emails from inbox/ and spam/ into the dataset (default: False). Use with `--regenerate`.
- `--evaluate`: Evaluate model performance with metrics and visualizations (default: False).
- `--k-values`: Specify custom k-values for evaluation (e.g., "1,3,5"). Default: [1, 3, 5].

---

### Examples

**First Run (Train and Generate Embeddings)**:
```bash
python main.py
```

**Evaluate with Custom k-Values**:
```bash
python main.py --evaluate --k-values "1,3,7"
```

**Merge Emails and Regenerate Embeddings**:
```bash
python main.py --merge-emails --regenerate
```

**Run Gmail API Classification**:
```bash
python main.py --run-email-classifier
```

---

### Notes on Execution Order

- Always start with `python main.py`: This ensures embeddings and the VectorDatabase are initialized.
- Avoid `--merge-emails` or `--run-email-classifier` as the first command.
- Check logs: Review `logs/spam_classifier.log` for execution info and dataset stats.
- Embedding mismatches: If errors arise, re-run with `--regenerate`.

---

### Checking and Managing Gmail API Integration

#### Verify Gmail Authentication

Ensure `cache/input/credentials.json` exists.

On first use with `--run-email-classifier`, a browser opens for Google authentication. If token expires, delete it and re-authenticate:

```bash
rm cache/input/token.json
python main.py --run-email-classifier
```

The app uses Gmail modify scope to label emails. Confirm access under your Google Account > Security > Third-party apps.

---

#### Check Gmail Labels

The app creates two labels: `Inbox_Custom` and `Spam_Custom`.

To view in Gmail:
- Open Gmail and check sidebar.
- If not visible: go to Settings > Labels > Show.

---

#### Inspect Classified Emails

Saved in `inbox/` or `spam/` folders as `.txt` files. Compare content with Gmail.

---

#### Manage Gmail Emails

If misclassified:
1. Move the email in Gmail UI.
2. Move `.txt` file to correct folder.
3. Re-run merge:

```bash
python main.py --merge-emails --regenerate
```

**Note**: Re-run merge only when you want to reuse correctly labeled emails to supplement the dataset and retrain the model to regenerate embeddings. Avoid unnecessary re-runs as it wastes time regenerating embeddings.

---

#### Monitor and Restart Classification

Classifier checks unread emails every 30 seconds. Press Ctrl+C to stop. Rerun to restart:

```bash
python main.py --run-email-classifier
```

---

#### Monitor Logs

Check `logs/spam_classifier.log` for:
- Processed message IDs
- Labeling results
- Errors (network, unreadable emails)

Example log:
```
2025-07-27 10:00:00,000 - root: Lưu email ID 12345 vào ./spam/email_12345.txt
```

---

#### Troubleshoot Gmail API Issues

- **Invalid credentials**: Check `credentials.json`.
- **Token expired**: Delete `token.json` and re-auth.
- **No emails processed**: Ensure emails are unread in Gmail.
- **Missing labels**: Labels will be auto-created.

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
├── config.py
├── data_loader.py
├── email_handler.py
├── embedding_generator.py
├── evaluator.py
├── knn_classifier.py
├── main.py
├── spam_classifier.py
├── tfidf_classifier.py
└── README.md
```

---

## Troubleshooting

- **Seaborn style error**: Update Matplotlib or switch to `plt.style.use('default')`.
- **CSV not found**: Ensure `dataset/2cls_spam_text_cls.csv` exists or run `--merge-emails`.
- **Auth failed**: Check credentials.json and regenerate token.
- **Embedding mismatch**: Use `--regenerate` after dataset changes.

---

## License

This project is licensed under the MIT License.

---

## Future Development

This version will evolve into a Streamlit-based interactive interface with visualization and real-time classification. Stay tuned for tags and updates on GitHub.

---

## Recent Updates (v2.0)

- **TF-IDF Integration**: Added TF-IDF classifier alongside KNN for benchmarking. Use `--evaluate` to compare performance (accuracy, precision, recall, F1) between TF-IDF (baseline) and best KNN (k selected by max F1).
- **Enhanced Visualization**: Updated evaluation plots to 5 rows:
  - Row 1: Lineplot of KNN metrics (k=1,3,5).
  - Row 2: Heatmaps of KNN confusion matrices per k.
  - Row 3: Barplot of label distribution.
  - Row 4: Grouped barplot comparing TF-IDF vs best KNN metrics.
  - Row 5: Heatmap of TF-IDF confusion matrix.
  - Improved spacing (via `height_ratios` and `hspace`) to avoid label overlap on x-axis.
- **Performance Optimization**: Ensured single training run for KNN and TF-IDF pipelines, reusing embeddings for efficiency.
- **Tag v2.0**: Released as `v2.0-command-line` to reflect these updates. Download from https://github.com/sonvt8/AIO2025/tags.
```