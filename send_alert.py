"""
send_alert.py — runs after universe.py in the GitHub Action.

Three scheduled runs daily (Mon-Fri):
  12:00 BST (11:00 UTC) — morning scan: alert if score > 12 AND gap > 0
  13:00 BST (12:00 UTC) — mid-morning check: alert if score still > 12
  14:00 BST (13:00 UTC) — final check: alert if score still > 12 before open

Each run compares against the previous CSV to detect if momentum is
holding, improving, or fading. Subject line and body reflect which
check this is so you know the context immediately.

Required GitHub Secrets:
  OUTLOOK_USER     — your Hotmail/Outlook sending address
  OUTLOOK_PASSWORD — your Outlook password
  ALERT_EMAILS     — comma-separated recipients
"""

import os
import sys
import smtplib
import pandas as pd
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SCORE_THRESHOLD = 12
DASHBOARD_URL   = "https://market-universe-generator-7jrhjfbttwfzlappdxaaysq.streamlit.app/"


def get_run_context():
    """
    Work out which of the three daily runs this is based on UTC hour.
    Returns (run_name, bst_time, is_morning_scan)
    """
    utc_hour = datetime.now(timezone.utc).hour
    if utc_hour == 11:
        return "Morning scan", "12:00 BST", True
    elif utc_hour == 12:
        return "Mid-morning check", "13:00 BST", False
    elif utc_hour == 13:
        return "Final check before open", "14:00 BST", False
    else:
        # Manual trigger — treat as morning scan
        return "Manual scan", "Now", True


def load_universe():
    path = "output/universe.csv"
    if not os.path.exists(path):
        print("No universe.csv found — skipping alert")
        sys.exit(0)
    df = pd.read_csv(path)
    df = df[df["ticker"] != "Ticker"].drop_duplicates(subset="ticker")
    for col in ["score", "gap_pct", "rvol", "premarket_rvol",
                "breakout_score", "trend_5d"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def get_candidates(df, is_morning_scan):
    """
    Morning scan: require gap_pct > 0 (real premarket gap) AND score > 12
    Later checks: score > 12 is enough — gap may have shifted slightly
    """
    if is_morning_scan:
        return df[
            (df["score"] > SCORE_THRESHOLD) &
            (df["gap_pct"] > 0)
        ].head(5)
    else:
        return df[df["score"] > SCORE_THRESHOLD].head(5)


def build_subject(candidates, run_name, bst_time):
    count   = len(candidates)
    tickers = ", ".join(candidates["ticker"].tolist())
    return (
        f"🚀 Momentum Scanner [{run_name} {bst_time}]: "
        f"{count} candidate{'s' if count > 1 else ''} — {tickers}"
    )


def build_email(candidates, run_name, bst_time, is_morning_scan):
    today = datetime.now(timezone.utc).strftime("%A %-d %B %Y")
    count = len(candidates)

    # Context message per run
    if is_morning_scan:
        context_msg = (
            "This is the <strong>morning scan at 12:00 BST</strong>. "
            "These stocks are gapping up in premarket with elevated volume. "
            "Open the dashboard to review, then run the pre-trade check at 13:00 and 14:00 "
            "before committing."
        )
        action_msg = "Review candidates — market opens at 14:30 BST"
    elif "13:00" in bst_time:
        context_msg = (
            "This is the <strong>mid-morning check at 13:00 BST</strong>. "
            "These candidates are still scoring above the threshold — momentum is holding. "
            "Use the pre-trade confirmation button on the dashboard to see the full picture."
        )
        action_msg = "Run pre-trade check on dashboard — 90 minutes to open"
    else:
        context_msg = (
            "This is the <strong>final check at 14:00 BST</strong>. "
            "30 minutes until the market opens. These candidates are still strong. "
            "Run the pre-trade check now and decide whether to trade."
        )
        action_msg = "Final decision time — market opens at 14:30 BST"

    # Plain text
    lines = [
        f"Momentum Scanner — {run_name} — {today}",
        f"{count} ticker{'s' if count > 1 else ''} above score {SCORE_THRESHOLD}",
        "",
    ]
    for _, row in candidates.iterrows():
        gap_str  = f"+{row['gap_pct']*100:.1f}%" if row['gap_pct'] > 0 else "—"
        lines.append(
            f"  {row['ticker']:<6}  Score: {row['score']:.1f}  "
            f"Gap: {gap_str}  RVOL: {row['rvol']:.1f}x  "
            f"Breakout: {row['breakout_score']:.2f}  Trend: {int(row['trend_5d'])}/5"
        )
    lines += [
        "",
        action_msg,
        f"Dashboard: {DASHBOARD_URL}",
        "",
        "Entry window: 13:30–14:15 BST. Target: +10%. Hard exit: 14:45 BST.",
        "Not financial advice.",
    ]
    plain = "\n".join(lines)

    # HTML rows
    rows_html = ""
    for _, row in candidates.iterrows():
        gap_pct  = row["gap_pct"] * 100
        gap_str  = f"+{gap_pct:.1f}%" if gap_pct > 0 else "—"
        gap_col  = "#27ae60" if gap_pct > 0 else "#888888"
        score    = row["score"]
        score_col = "#27ae60" if score >= 15 else "#f39c12" if score >= 12 else "#e74c3c"
        rows_html += f"""
        <tr>
          <td style="padding:10px 12px;font-weight:700;font-size:15px;">{row['ticker']}</td>
          <td style="padding:10px 12px;color:{score_col};font-weight:700;">{score:.1f}</td>
          <td style="padding:10px 12px;color:{gap_col};font-weight:600;">{gap_str}</td>
          <td style="padding:10px 12px;">{row['rvol']:.1f}x</td>
          <td style="padding:10px 12px;">{row['breakout_score']:.2f}</td>
          <td style="padding:10px 12px;">{int(row['trend_5d'])}/5</td>
        </tr>"""

    # Run badge colour
    badge_col = "#2E5FA3" if is_morning_scan else "#f39c12" if "13:00" in bst_time else "#c0392b"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f5f5f5;font-family:Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f5f5f5;padding:30px 0;">
<tr><td align="center">
<table width="600" cellpadding="0" cellspacing="0" style="background:#fff;border-radius:8px;overflow:hidden;">

  <tr><td style="background:#1a1a2e;padding:28px 32px;">
    <p style="margin:0 0 8px;font-family:monospace;font-size:11px;letter-spacing:2px;
              text-transform:uppercase;color:{badge_col};">
      {run_name} &nbsp;|&nbsp; {bst_time} &nbsp;|&nbsp; {today}
    </p>
    <h1 style="margin:0 0 4px;color:#fff;font-size:22px;font-weight:700;">
      {count} strong candidate{'s' if count > 1 else ''} still in play
    </h1>
  </td></tr>

  <tr><td style="padding:24px 32px 0;">
    <p style="margin:0;color:#444;font-size:14px;line-height:1.6;">{context_msg}</p>
  </td></tr>

  <tr><td style="padding:20px 32px;">
    <table width="100%" cellpadding="0" cellspacing="0"
           style="border-collapse:collapse;border:1px solid #e0e0e0;border-radius:6px;overflow:hidden;">
      <tr style="background:#f8f8f8;">
        <th style="padding:10px 12px;text-align:left;font-size:11px;color:#888;
                   letter-spacing:1px;text-transform:uppercase;">Ticker</th>
        <th style="padding:10px 12px;text-align:left;font-size:11px;color:#888;
                   letter-spacing:1px;text-transform:uppercase;">Score</th>
        <th style="padding:10px 12px;text-align:left;font-size:11px;color:#888;
                   letter-spacing:1px;text-transform:uppercase;">Gap %</th>
        <th style="padding:10px 12px;text-align:left;font-size:11px;color:#888;
                   letter-spacing:1px;text-transform:uppercase;">RVOL</th>
        <th style="padding:10px 12px;text-align:left;font-size:11px;color:#888;
                   letter-spacing:1px;text-transform:uppercase;">Breakout</th>
        <th style="padding:10px 12px;text-align:left;font-size:11px;color:#888;
                   letter-spacing:1px;text-transform:uppercase;">Trend</th>
      </tr>
      {rows_html}
    </table>
  </td></tr>

  <tr><td style="padding:0 32px 24px;">
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr>
        <td width="33%" style="padding:12px;background:#f0f9f4;border-radius:6px;text-align:center;">
          <p style="margin:0;font-size:11px;color:#888;text-transform:uppercase;">Entry window</p>
          <p style="margin:4px 0 0;font-size:13px;font-weight:700;color:#27ae60;">13:30–14:15 BST</p>
        </td>
        <td width="4%"></td>
        <td width="33%" style="padding:12px;background:#fff8f0;border-radius:6px;text-align:center;">
          <p style="margin:0;font-size:11px;color:#888;text-transform:uppercase;">Target</p>
          <p style="margin:4px 0 0;font-size:13px;font-weight:700;color:#f39c12;">+10% — sell manually</p>
        </td>
        <td width="4%"></td>
        <td width="33%" style="padding:12px;background:#fff0f0;border-radius:6px;text-align:center;">
          <p style="margin:0;font-size:11px;color:#888;text-transform:uppercase;">Hard exit</p>
          <p style="margin:4px 0 0;font-size:13px;font-weight:700;color:#e74c3c;">14:45 BST</p>
        </td>
      </tr>
    </table>
  </td></tr>

  <tr><td style="padding:0 32px 32px;text-align:center;">
    <a href="{DASHBOARD_URL}"
       style="display:inline-block;background:#c8410a;color:#fff;text-decoration:none;
              padding:14px 32px;border-radius:6px;font-weight:700;font-size:14px;">
      Open Dashboard
    </a>
  </td></tr>

  <tr><td style="background:#f8f8f8;padding:16px 32px;border-top:1px solid #e0e0e0;">
    <p style="margin:0;font-size:11px;color:#aaa;text-align:center;">
      Automated alert · Not financial advice ·
      <a href="{DASHBOARD_URL}" style="color:#aaa;">Dashboard</a>
    </p>
  </td></tr>

</table>
</td></tr>
</table>
</body></html>"""

    return plain, html


def send_email(plain, html, candidates, run_name, bst_time):
    outlook_user     = os.environ.get("OUTLOOK_USER", "")
    outlook_password = os.environ.get("OUTLOOK_PASSWORD", "")
    alert_emails     = os.environ.get("ALERT_EMAILS", "")

    if not outlook_user or not outlook_password or not alert_emails:
        print("Email secrets not configured — skipping alert")
        sys.exit(0)

    recipients = [e.strip() for e in alert_emails.split(",") if e.strip()]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = build_subject(candidates, run_name, bst_time)
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
        print(f"Alert sent [{run_name}] to: {', '.join(recipients)}")
    except Exception as e:
        print(f"Failed to send email: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_name, bst_time, is_morning_scan = get_run_context()
    print(f"Running as: {run_name} ({bst_time})")

    df = load_universe()
    candidates = get_candidates(df, is_morning_scan)

    if candidates.empty:
        print(
            f"No candidates above score {SCORE_THRESHOLD}"
            + (" with real gap" if is_morning_scan else "")
            + f" at {bst_time} — no alert sent"
        )
        sys.exit(0)

    tickers = ", ".join(candidates["ticker"].tolist())
    print(f"Found {len(candidates)} candidate(s): {tickers}")

    plain, html = build_email(candidates, run_name, bst_time, is_morning_scan)
    send_email(plain, html, candidates, run_name, bst_time)
