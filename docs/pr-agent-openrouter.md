# PR-Agent With OpenRouter

This repository is configured to run [PR-Agent](https://github.com/The-PR-Agent/pr-agent) on GitHub pull requests through OpenRouter.

## GitHub Setup

Add this repository secret:

```text
OPENROUTER_API_KEY=<your OpenRouter API key>
```

The built-in `GITHUB_TOKEN` is provided automatically by GitHub Actions.

Optional repository variables:

```text
PR_AGENT_MODEL=openrouter/deepseek/deepseek-v4-pro
PR_AGENT_FALLBACK_MODELS=["openrouter/deepseek/deepseek-chat-v3.1"]
```

If those variables are not set, the workflow uses the same defaults.

## Automatic Behavior

On pull requests opened, reopened, marked ready for review, or review-requested, PR-Agent will automatically:

- run `/describe`
- run `/review`

It will not run `/improve` automatically to avoid noisy suggestions and unnecessary OpenRouter spend. You can still trigger it manually by commenting:

```text
/improve
```

Useful manual commands:

```text
/review
/describe
/ask What are the riskiest runtime changes in this PR?
/improve
```

## Local / Codex Usage

Codex can use the same repo configuration when you ask it to inspect PR-Agent output or trigger a manual PR-Agent command on a GitHub PR.

For local CLI experimentation, install PR-Agent separately and provide secrets as environment variables:

```powershell
$env:OPENROUTER__KEY="..."
$env:GITHUB_TOKEN="..."
python -m pr_agent.cli --pr_url=https://github.com/<owner>/<repo>/pull/<number> review
```

Do not commit local secret files such as `.secrets.toml` or `.env`.
