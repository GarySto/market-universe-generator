"""
send_alert.py — runs after universe.py in the GitHub Action.
Reads output/universe.csv and sends an email if any ticker scores above 12
with a real premarket gap (gap_pct > 0).

Required GitHub Secrets:
  OUTLOOK_USER     — your Hotmail/Outlook sending address (e.g. gary@hotmail.com)
  OUTLOOK_PASSWORD — your Outlook password for that account
  ALERT_EMAILS     — comma-separated recipient list (e.g. gary@hotmail.com,gary@microsoft.com)
"""

import os
import sys
import smtplib
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

SCORE_THRESHOLD = 12
DASHBOARD_URL   = "https://market-universe-generator-7jrhjfbttwfzlappdxaaysq.streamlit.app/"


def load_universe():
    path = "output/universe.csv"
    if not os.path.exists(path):
        print("No universe.csv found — skipping alert")
        sys.exit(0)
    df = pd.read_csv(path)
    df = df[df["ticker"] != "Ticker"].drop_duplicates(subset="ticker")
    return df


def build_email(candidates):
    today = datetime.utcnow().strftime("%A %-d %B %Y")
    count = len(candidates)

    # ── Plain text version ──────────────────────────────────────────────────
    lines = [
        f"Momentum Scanner — {today}",
        f"{count} ticker{'s' if count > 1 else ''} scoring above {SCORE_THRESHOLD} with premarket gap",
        "",
    ]
    for _, row in candidates.iterrows():
        gap_str  = f"+{row['gap_pct']*100:.1f}%" if row['gap_pct'] > 0 else f"{row['gap_pct']*100:.1f}%"
        rvol_str = f"{row['rvol']:.1f}x"
        lines.append(
            f"  {row['ticker']:<6}  Score: {row['score']:.1f}  "
            f"Gap: {gap_str}  RVOL: {rvol_str}  "
            f"Breakout: {row['breakout_score']:.2f}  "
            f"Trend: {int(row['trend_5d'])}/5"
        )
    lines += [
        "",
        f"Dashboard: {DASHBOARD_URL}",
        "",
        "Entry window: 13:30-14:15 BST. Target: +10%. Hard exit: 14:45 BST.",
        "Run the pre-trade confirmation check before committing.",
        "",
        "This is an automated alert. Not financial advice.",
    ]
    plain = "\n".join(lines)

    # ── HTML version ────────────────────────────────────────────────────────
    rows_html = ""
    for _, row in candidates.iterrows():
        gap_pct  = row["gap_pct"] * 100
        gap_str  = f"+{gap_pct:.1f}%" if gap_pct > 0 else f"{gap_pct:.1f}%"
        gap_col  = "#27ae60" if gap_pct > 0 else "#e74c3c"
        score    = row["score"]
        score_col = "#27ae60" if score >= 15 else "#f39c12" if score >= 12 else "#e74c3c"

        rows_html += f"""
        <tr>
          <td style="padding:10px 12px;font-weight:600;font-size:15px;">{row['ticker']}</td>
          <td style="padding:10px 12px;color:{score_col};font-weight:700;font-size:15px;">{score:.1f}</td>
          <td style="padding:10px 12px;color:{gap_col};font-weight:600;">{gap_str}</td>
          <td style="padding:10px 12px;">{row['rvol']:.1f}x</td>
          <td style="padding:10px 12px;">{row['breakout_score']:.2f}</td>
          <td style="padding:10px 12px;">{int(row['trend_5d'])}/5</td>
        </tr>"""

    html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f5f5f5;font-family:Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f5f5f5;padding:30px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;">

        <!-- Header -->
        <tr><td style="background:#1a1a2e;padding:28px 32px;">
          <p style="margin:0;color:#c8410a;font-size:11px;letter-spacing:2px;text-transform:uppercase;font-family:monospace;">
            Momentum Scanner Alert
          </p>
          <h1 style="margin:8px 0 4px;color:#ffffff;font-size:24px;font-weight:700;">
            {count} strong candidate{'s' if count > 1 else ''} today
          </h1>
          <p style="margin:0;color:#aaaaaa;font-size:13px;">{today}</p>
        </td></tr>

        <!-- Intro -->
        <tr><td style="padding:24px 32px 0;">
          <p style="margin:0;color:#444;font-size:14px;line-height:1.6;">
            The following ticker{'s have' if count > 1 else ' has'} scored above <strong>{SCORE_THRESHOLD}</strong>
            with a real premarket gap. Open the dashboard and run the pre-trade confirmation
            check before the market opens at <strong>14:30 BST</strong>.
          </p>
        </td></tr>

        <!-- Table -->
        <tr><td style="padding:20px 32px;">
          <table width="100%" cellpadding="0" cellspacing="0"
                 style="border-collapse:collapse;border-radius:6px;overflow:hidden;border:1px solid #e0e0e0;">
            <tr style="background:#f8f8f8;">
              <th style="padding:10px 12px;text-align:left;font-size:11px;color:#888;letter-spacing:1px;text-transform:uppercase;">Ticker</th>
              <th style="padding:10px 12px;text-align:left;font-size:11px;color:#888;letter-spacing:1px;text-transform:uppercase;">Score</th>
              <th style="padding:10px 12px;text-align:left;font-size:11px;color:#888;letter-spacing:1px;text-transform:uppercase;">Gap %</th>
              <th style="padding:10px 12px;text-align:left;font-size:11px;color:#888;letter-spacing:1px;text-transform:uppercase;">RVOL</th>
              <th style="padding:10px 12px;text-align:left;font-size:11px;color:#888;letter-spacing:1px;text-transform:uppercase;">Breakout</th>
              <th style="padding:10px 12px;text-align:left;font-size:11px;color:#888;letter-spacing:1px;text-transform:uppercase;">Trend</th>
            </tr>
            {rows_html}
          </table>
        </td></tr>

        <!-- Key info -->
        <tr><td style="padding:0 32px 24px;">
          <table width="100%" cellpadding="0" cellspacing="0">
            <tr>
              <td width="33%" style="padding:12px;background:#f0f9f4;border-radius:6px;text-align:center;">
                <p style="margin:0;font-size:11px;color:#888;text-transform:uppercase;">Entry window</p>
                <p style="margin:4px 0 0;font-size:14px;font-weight:700;color:#27ae60;">13:30 – 14:15 BST</p>
              </td>
              <td width="4%"></td>
              <td width="33%" style="padding:12px;background:#fff8f0;border-radius:6px;text-align:center;">
                <p style="margin:0;font-size:11px;color:#888;text-transform:uppercase;">Target</p>
                <p style="margin:4px 0 0;font-size:14px;font-weight:700;color:#f39c12;">+10% from entry</p>
              </td>
              <td width="4%"></td>
              <td width="33%" style="padding:12px;background:#fff0f0;border-radius:6px;text-align:center;">
                <p style="margin:0;font-size:11px;color:#888;text-transform:uppercase;">Hard exit</p>
                <p style="margin:4px 0 0;font-size:14px;font-weight:700;color:#e74c3c;">14:45 BST</p>
              </td>
            </tr>
          </table>
        </td></tr>

        <!-- CTA -->
        <tr><td style="padding:0 32px 32px;text-align:center;">
          <a href="{DASHBOARD_URL}"
             style="display:inline-block;background:#c8410a;color:#ffffff;text-decoration:none;
                    padding:14px 32px;border-radius:6px;font-weight:700;font-size:14px;
                    letter-spacing:0.5px;">
            Open Dashboard
          </a>
        </td></tr>

        <!-- Footer -->
        <tr><td style="background:#f8f8f8;padding:16px 32px;border-top:1px solid #e0e0e0;">
          <p style="margin:0;font-size:11px;color:#aaa;text-align:center;">
            Automated alert from your Momentum Scanner &nbsp;|&nbsp;
            Not financial advice &nbsp;|&nbsp;
            <a href="{DASHBOARD_URL}" style="color:#aaa;">View dashboard</a>
          </p>
        </td></tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""

    return plain, html


def send_email(plain, html, candidates):
    outlook_user     = os.environ.get("OUTLOOK_USER", "")
    outlook_password = os.environ.get("OUTLOOK_PASSWORD", "")
    alert_emails     = os.environ.get("ALERT_EMAILS", "")

    if not outlook_user or not outlook_password or not alert_emails:
        print("Email secrets not configured — skipping alert")
        sys.exit(0)

    recipients = [e.strip() for e in alert_emails.split(",") if e.strip()]
    today      = datetime.utcnow().strftime("%A %-d %B %Y")
    count      = len(candidates)
    tickers    = ", ".join(candidates["ticker"].tolist())

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Momentum Scanner: {count} strong candidate{'s' if count > 1 else ''} today — {tickers}"
    msg["From"]    = f"Momentum Scanner <{outlook_user}>"
    msg["To"]      = ", ".join(recipients)

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html,  "html"))

    try:
        with smtplib.SMTP("smtp-mail.outlook.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(outlook_user, outlook_password)
            server.sendmail(outlook_user, recipients, msg.as_string())
        print(f"Alert sent to: {', '.join(recipients)}")
    except Exception as e:
        print(f"Failed to send email: {e}")
        sys.exit(1)


if __name__ == "__main__":
    df = load_universe()

    # Only alert on strong signals with a real premarket gap
    candidates = df[
        (df["score"] > SCORE_THRESHOLD) &
        (df["gap_pct"] > 0)
    ].head(5)

    if candidates.empty:
        print(f"No tickers above score {SCORE_THRESHOLD} with a real gap today — no alert sent")
        sys.exit(0)

    tickers = ", ".join(candidates["ticker"].tolist())
    print(f"Found {len(candidates)} strong candidate(s): {tickers}")

    plain, html = build_email(candidates)
    send_email(plain, html, candidates)
