# Deploying SafeSOS to Hugging Face Spaces

This guide walks through getting the SafeSOS app live on a public URL,
free, in about 10 minutes.

---

## 1. Local sanity check (don't skip)

Before pushing to HF, confirm the app runs locally end-to-end.

```bash
cd ~/Projects/DL/SafeSOS

# Make sure these exist:
ls -lh models/checkpoints/mobilenet_best.pth
ls -lh models/checkpoints/mobilenet_meta.json

# Install gradio
pip install "gradio>=4.0"

# Run
python main.py
```

Open http://localhost:7860 in your browser. Allow camera access. You
should see:

- The dark themed UI with the SafeSOS logo
- A live webcam feed on the left
- Status pill, message card, and confidence bar on the right
- Browser TTS speaking the predicted message

If any of these don't work, fix locally first.

---

## 2. Create a Hugging Face Space

1. Go to https://huggingface.co/spaces and click **"Create new Space"**
2. Owner: your HF username
3. Space name: `safesos`
4. License: MIT
5. SDK: **Gradio**
6. Space hardware: **CPU basic (free)** — works fine for inference
7. Visibility: **Public**

Click **"Create Space"**.

You'll land on the empty Space page with git push instructions.

---

## 3. Configure the local repo for HF push

In your SafeSOS repo:

```bash
# Add HF as a second remote (keep GitHub as origin)
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/safesos

# Verify
git remote -v
# should show both:
#   origin -> github.com/...
#   hf     -> huggingface.co/spaces/...
```

---

## 4. Prepare files for the Space

HF Spaces requires the README to be at the repo root with their YAML
header. Your GitHub README is different (project docs), so we swap
just before pushing.

```bash
# Back up your GitHub README
cp README.md README_github.md

# Replace with the HF-flavoured README
cp spaces_README.md README.md
# Edit it: replace YOUR-USERNAME with your actual GitHub username
```

Also ensure HF Spaces can find the model checkpoint. Since you've
already added `mobilenet_best.pth` to git (it's small ~6MB), it will
push automatically.

---

## 5. Push to HF

```bash
# First push needs HF token authentication
# Get token: https://huggingface.co/settings/tokens (create one with WRITE access)

git add README.md
git commit -m "chore: add HF Spaces README"

# Push to HF (will prompt for username and token)
git push hf main
```

If you get an LFS error for the .pth file, install LFS:
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "chore: track .pth via LFS"
git push hf main
```

---

## 6. Watch the build

Go back to your Space URL on huggingface.co. You'll see:

- "Building" → installing dependencies (~2-3 min)
- "Running" → app is live

Click the "App" tab. Your SafeSOS demo is now public.

Public URL format:
```
https://huggingface.co/spaces/YOUR_HF_USERNAME/safesos
```

---

## 7. Restore your GitHub README

After HF is live:

```bash
cp README_github.md README.md
rm README_github.md
git add README.md
git commit -m "chore: restore GitHub README"
git push origin main   # only push to GitHub, not HF
```

This way GitHub has the developer-facing README and HF has the
end-user one.

---

## Troubleshooting

### Build fails with "torch download timeout"

Add a more specific torch requirement in `requirements.txt`:
```
torch==2.1.0
torchvision==0.16.0
```

### Camera permission denied in Gradio

The deployed Space is HTTPS, so the browser permission prompt should
work. If your local browser blocked it once, click the camera icon in
the URL bar to reset.

### App is slow on first prediction

That's the model loading lazily. After the first frame, inference is
under 100ms per frame on free CPU.

### Memory limit exceeded

Free tier has 16GB RAM. Our model is tiny (~6MB). If you hit this it's
probably a memory leak in the streaming loop — drop the stream rate to
0.5s in `app/interface.py`.

---

## Submission checklist

For the course rubric:

- [ ] Space is public and live
- [ ] URL works in incognito mode (no cached auth)
- [ ] Camera input works
- [ ] At least 3 different gestures recognized correctly
- [ ] Abstention triggers when you hold an ambiguous pose
- [ ] TTS plays audio (browser-dependent; some block autoplay)
- [ ] URL added to your written report
- [ ] URL added to your GitHub README
