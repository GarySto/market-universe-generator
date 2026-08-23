"""
alerts_gmail.py - Gmail alert sender (defect D3). New module.
Replaces the dead Outlook basic-auth path. Works locally and in Actions.

Env vars:
  GMAIL_USER          - the HQ Gmail address (e.g. garystow.hq@gmail.com)
  GMAIL_APP_PASSWORD  - 16-char app password (NOT the account password)
  ALERT_EMAILS        - comma-separated recipients (existing secret, unchanged)

Setup of the app password is PATCH_GUIDE.md step 2.
Manual test:  py alerts_gmail.py            (sends a test email)
"""
import os
import smtplib
import ssl
from email.mime.text import MIMEText


def send_alert(subject: str, body: str) -> bool:
    user = os.environ.get("GMAIL_USER", "")
    pw = os.environ.get("GMAIL_APP_PASSWORD", "")
    to = [a.strip() for a in os.environ.get("ALERT_EMAILS", user).split(",") if a.strip()]
    if not user or not pw:
        print("[alerts_gmail] GMAIL_USER / GMAIL_APP_PASSWORD not set - alert NOT sent:")
        print(f"  SUBJECT: {subject}")
        return False
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = ", ".join(to)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
            s.login(user, pw)
            s.sendmail(user, to, msg.as_string())
        print(f"[alerts_gmail] sent: {subject} -> {', '.join(to)}")
        return True
    except Exception as e:
        print(f"[alerts_gmail] SEND FAILED: {e}")
        return False


if __name__ == "__main__":
    ok = send_alert("HQ test alert", "If you are reading this, the alert spine is alive.")
    raise SystemExit(0 if ok else 1)
