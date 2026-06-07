---
title: SafeSOS
emoji: 🆘
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: main.py
pinned: true
license: mit
short_description: Selective static gesture recognition for silent emergency communication
---

# 🆘 SafeSOS — Silent Emergency Communication

**SafeSOS** recognises a small vocabulary of static hand gestures and
translates them into spoken emergency messages, for use in high-stakes
silent-communication scenarios such as non-verbal ICU patients,
hearing-impaired users in distress, and similar safety-critical
settings.

## How it works

Show one of these gestures to your webcam. SafeSOS uses a fine-tuned
MobileNetV3-Small with temperature-scaled selective prediction — the
model abstains when it's not confident, rather than guessing.

| Gesture | Meaning |
|---|---|
| 📞 call | "I need to make a call" |
| ✋ palm | "I need help" |
| ✋ stop | "Stop, do not approach" |
| ✊ fist | "Alert, I am in danger" |
| 👌 ok | "I am OK" |
| ✌️ peace | "Confirmed safe" |
| ☝️ one / two / three | numeric (person count, floor) |

## ⚠️ Research prototype

This is a research prototype built as a course project. It is **not** a
medical device and **must not** be relied on for actual emergencies.
The selective-prediction mechanism reduces — but does not eliminate —
the risk of misrecognition.

## Links

- 📄 [Source code on GitHub](https://github.com/Kattine/SafeSOS)
- 📝 Paper available in the GitHub repo

Built for the Computer Vision module project.
