# Release Progress Tracker

_Last updated: This file is automatically updated by the Release Progress Tracker workflow_

This file tracks progress toward upcoming releases by monitoring GitHub milestones, issues, and pull requests.

## How It Works

The Release Progress Tracker workflow runs automatically when:
- Issues or PRs are opened, closed, or updated
- Milestones are created or modified
- Daily at 00:00 UTC
- Manually triggered via workflow dispatch

The workflow will populate this file with:
- Progress bars showing completion percentage for each milestone
- Lists of open and recently completed issues/PRs
- Due dates for upcoming releases

## Setting Up Milestones

To track progress for a release:

1. Create a milestone in GitHub (e.g., "v1.0.0", "v2.0.0-alpha.1")
2. Add a description and due date (optional but recommended)
3. Assign issues and PRs to the milestone
4. The workflow will automatically track progress

## Example Progress Display

When milestones are configured, they will appear here with progress bars like:

### v1.0.0

**Due:** Jan 31, 2026

**Progress:** 60% (6/10 issues completed)

`████████████░░░░░░░░` 60%

#### Open Issues (4)
- [ ] #123 Add new feature
- [ ] #124 Fix critical bug

#### Recently Completed Issues (showing 6)
- [x] #120 Update documentation
- [x] #119 Improve performance
