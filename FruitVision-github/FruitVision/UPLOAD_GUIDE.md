# 📤 GitHub Upload Guide — FruitVision

A step-by-step guide to uploading this project professionally.

---

## Prerequisites

- [ ] [Git installed](https://git-scm.com/downloads) — verify with `git --version`
- [ ] A [GitHub account](https://github.com)
- [ ] Python 3.10+ installed

---

## Step 1 — Set Up Git (first-time only)

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

---

## Step 2 — Create the Repository on GitHub

1. Go to **github.com → New repository** (the `+` button top-right)
2. Fill in:
   - **Repository name:** `FruitVision`
   - **Description:** `AI-powered fruit classification & ripeness detection using Digital Image Processing`
   - **Visibility:** Public *(so others can see your DIP project)*
   - ☐ Do **NOT** check "Add README" — you already have one
3. Click **Create repository**
4. Copy the repo URL shown (looks like `https://github.com/your-username/FruitVision.git`)

---

## Step 3 — Set Up Local Folder

Place all your files in one folder:

```
FruitVision/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## Step 4 — Initialize Git & Push

Open a terminal **inside** the `FruitVision/` folder and run these commands one by one:

```bash
# 1. Initialize git
git init

# 2. Add all files
git add .

# 3. Commit with a clear message
git commit -m "feat: initial release — FruitVision DIP system

- Streamlit 4-tab dashboard (Analysis, Visualization, Knowledge, Features)
- FastAPI /predict endpoint with 23-feature extraction pipeline
- HSV/LAB color, GLCM texture, Sobel/Canny edge, Otsu segmentation
- Supports Apple, Banana, Mango, Orange, Strawberry
- scikit-image GLCM with numpy fallback"

# 4. Rename default branch to 'main'
git branch -M main

# 5. Link to your GitHub repo (replace URL with yours)
git remote add origin https://github.com/your-username/FruitVision.git

# 6. Push!
git push -u origin main
```

GitHub will ask for your **username + password** (use a Personal Access Token as password — see below).

---

## Creating a Personal Access Token (PAT)

GitHub no longer accepts passwords for push. Use a token:

1. GitHub → **Settings** → **Developer Settings** → **Personal Access Tokens** → **Tokens (classic)**
2. Click **Generate new token (classic)**
3. Give it a name, set expiry, check **`repo`** scope
4. Copy the token — use it as your password when git asks

---

## Step 5 — Polish Your GitHub Repo Page

After pushing, go to your repo page on GitHub and:

1. **Add topics** (click the ⚙️ gear next to "About"):
   `python`, `streamlit`, `fastapi`, `opencv`, `digital-image-processing`, `fruit-detection`, `computer-vision`

2. **Add a description** in the About box:
   *AI-powered fruit classification & ripeness detection using 23 DIP features — no deep learning required.*

3. **Pin the repo** to your profile:
   Go to your profile → click **Customize profile** → pin this repo

---

## Step 6 — Future Updates

Whenever you make changes:

```bash
git add .
git commit -m "fix: improve banana classification threshold"
git push
```

---

## Useful Git Commands

| Command | What it does |
|---|---|
| `git status` | See what files changed |
| `git log --oneline` | See commit history |
| `git diff` | See exact line changes |
| `git pull` | Get latest changes from GitHub |

---

*FruitVision · COMP-342L P06 · Spring 2025 · Pak-Austria Fachhochschule*
