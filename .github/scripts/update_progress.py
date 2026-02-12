#!/usr/bin/env python3
"""Update milestone progress in RELEASE_PROGRESS.md."""

import json
import os
import sys
from datetime import datetime
from urllib.request import Request, urlopen


def get_milestones(owner, repo, token):
    """Get open milestones from GitHub API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/milestones?state=open"
    headers = {"Authorization": f"token {token}"} if token else {}
    req = Request(url, headers=headers)
    with urlopen(req) as response:
        return json.loads(response.read())


def get_milestone_issues(owner, repo, milestone_number, token):
    """Get issues for a milestone."""
    url = f"https://api.github.com/repos/{owner}/{repo}/issues?milestone={milestone_number}&state=all&per_page=100"
    headers = {"Authorization": f"token {token}"} if token else {}
    req = Request(url, headers=headers)
    with urlopen(req) as response:
        return json.loads(response.read())


def generate_progress_bar(percentage, width=20):
    """Generate a Unicode progress bar."""
    filled = int((percentage / 100) * width)
    return "█" * filled + "░" * (width - filled)


def main():
    owner = os.getenv("GITHUB_REPOSITORY_OWNER") or sys.argv[1]
    repo = os.getenv("GITHUB_REPOSITORY", "").split("/")[-1] or sys.argv[2]
    token = os.getenv("GITHUB_TOKEN", "")

    milestones = get_milestones(owner, repo, token)

    if not milestones:
        print("No open milestones found")
        return

    report = "# Release Progress\n\n"
    report += f"_Updated: {datetime.utcnow().strftime('%Y-%m-%d')}_\n\n"

    for ms in milestones:
        total = ms["open_issues"] + ms["closed_issues"]
        completed = ms["closed_issues"]
        pct = int((completed / total) * 100) if total > 0 else 0

        report += f"## {ms['title']}\n\n"
        if ms.get("due_on"):
            due = datetime.fromisoformat(ms["due_on"].replace("Z", "+00:00"))
            report += f"**Due:** {due.strftime('%Y-%m-%d')}\n\n"

        report += f"`{generate_progress_bar(pct)}` **{pct}%** ({completed}/{total})\n\n"

        issues = get_milestone_issues(owner, repo, ms["number"], token)
        open_issues = [i for i in issues if i["state"] == "open" and not i.get("pull_request")]

        if open_issues:
            report += "**Open:**\n"
            for issue in open_issues[:5]:
                report += f"- #{issue['number']} {issue['title']}\n"
            if len(open_issues) > 5:
                report += f"- ... and {len(open_issues) - 5} more\n"
            report += "\n"

    with open("RELEASE_PROGRESS.md", "w") as f:
        f.write(report)

    print("Progress updated successfully")


if __name__ == "__main__":
    main()
