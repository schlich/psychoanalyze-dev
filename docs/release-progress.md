# Release Progress Tracker

The Release Progress Tracker is an automated GitHub workflow that provides visibility into development progress toward upcoming releases.

## Overview

This workflow automatically tracks:

- **Milestones**: Progress toward release milestones
- **Issues**: Open and completed issues for each milestone
- **Pull Requests**: Open and merged PRs for each milestone
- **Progress Visualization**: Visual progress bars showing completion percentage

## How It Works

The workflow is triggered by:

- Issue events (opened, closed, labeled, milestoned, etc.)
- Pull request events (opened, closed, labeled)
- Daily at 00:00 UTC (scheduled)
- Manual workflow dispatch

When triggered, the workflow:

1. Queries all open milestones in the repository
2. For each milestone, calculates completion percentage
3. Generates a visual progress bar
4. Lists open and recently completed issues/PRs
5. Updates `RELEASE_PROGRESS.md` in the repository

## Setting Up Milestones

To start tracking progress for a release:

1. **Create a milestone** in GitHub Issues
   - Go to Issues → Milestones → New milestone
   - Example: `v1.0.0`, `v2.0.0-alpha.1`

2. **Add details** (recommended)
   - Description: What's included in this release
   - Due date: Target release date

3. **Assign issues and PRs** to the milestone
   - When creating/editing issues: Set the milestone field
   - When creating/editing PRs: Set the milestone field

4. The workflow will automatically update progress

## Progress File

The tracker updates [`RELEASE_PROGRESS.md`](../RELEASE_PROGRESS.md) with:

### Example Output

```markdown
# Release Progress Tracker

_Last updated: Wed, 12 Feb 2026 02:40:00 GMT_

## Milestones

### v1.0.0

**Due:** Jan 31, 2026

**Progress:** 60% (6/10 issues completed)

`████████████░░░░░░░░` 60%

#### Open Issues (4)
- [ ] #123 Add new feature `enhancement`
- [ ] #124 Fix critical bug `bug` `priority:high`
- [ ] #125 Update documentation `docs`
- [ ] #126 Add tests `test`

#### Recently Completed Issues (showing 6 of 6)
- [x] #120 Update documentation `docs`
- [x] #119 Improve performance `enhancement`
- [x] #118 Fix minor bug `bug`
```

## Viewing Progress

You can view the progress in multiple ways:

1. **In the repository**: Navigate to [`RELEASE_PROGRESS.md`](../RELEASE_PROGRESS.md)
2. **From README**: Click the "Release Progress Tracker" link in the main README
3. **GitHub Milestones page**: Go to Issues → Milestones for a quick overview

## Workflow Details

### File Location

`.github/workflows/release-progress.yml`

### Permissions Required

- `contents: write` - To commit updates to RELEASE_PROGRESS.md
- `issues: write` - To read issue and milestone data
- `pull-requests: read` - To read PR data

### Customization

You can customize the workflow by editing `.github/workflows/release-progress.yml`:

- **Progress bar length**: Change `const blocks = 20;` to adjust bar width
- **Number of recent items**: Change `.slice(0, 10)` to show more/fewer completed items
- **Schedule**: Modify the `cron` expression to change update frequency
- **Triggers**: Add or remove event types in the `on:` section

## Troubleshooting

### Progress not updating

1. Check that issues/PRs are assigned to a milestone
2. Verify the workflow has the required permissions
3. Check workflow runs in the Actions tab
4. Manually trigger the workflow via workflow_dispatch

### No milestones shown

The workflow only tracks **open** milestones. If no milestones exist or all are closed, the file will show setup instructions.

### Merge conflicts

If manual edits are made to `RELEASE_PROGRESS.md`, the workflow may encounter merge conflicts. To resolve:

1. Don't manually edit `RELEASE_PROGRESS.md` (it's auto-generated)
2. If conflicts occur, delete the file and let the workflow regenerate it

## Best Practices

1. **Create milestones early** - Set up milestones when planning releases
2. **Set due dates** - Helps track progress over time
3. **Use labels** - Labels make it easier to categorize and filter issues
4. **Close issues promptly** - Keeps progress accurate and up-to-date
5. **Review regularly** - Check progress during standups or sprint reviews

## Integration with CI/CD

This tracker complements your release process:

1. Track development progress via milestones
2. When a milestone is 100% complete, create a release
3. Use [semantic versioning](https://semver.org/) for version numbers
4. The tracker helps communicate progress to stakeholders
