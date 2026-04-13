ORCHESTRATOR_SYSTEM = """\
You are a senior data quality analyst AI orchestrating a multi-agent data quality control pipeline.

Your responsibilities:
1. Analyze QC findings from the QC Runner agent.
2. Decide the support level (L1 = auto-fix, L2 = human review) for borderline issues.
3. Summarize findings in plain English for the data team.
4. Escalate critical or unusual patterns.

Rules:
- Confidence >= 0.85 → L1 (auto-correct)
- 0.50 <= Confidence < 0.85 → L2 (flag for human)
- Confidence < 0.50 → skip (no action)
- Business-critical columns (customer_id, primary keys) → always L2 regardless of confidence
- Duplicate issues → always L2 (safety, never auto-delete)

Always respond with valid JSON matching the requested schema.
"""


def build_triage_prompt(issues: list[dict], borderline_only: bool = True) -> str:
    issues_json = "\n".join(str(i) for i in issues)
    return f"""\
The QC Runner has reported the following {"borderline (0.75-0.85 confidence) " if borderline_only else ""}issues:

{issues_json}

For each issue, decide:
1. support_level: "L1" or "L2"
2. reasoning: one sentence explaining your decision

Respond as JSON array:
[
  {{"issue_id": "...", "support_level": "L1"|"L2", "reasoning": "..."}}
]
"""


def build_summary_prompt(stats: dict) -> str:
    return f"""\
Summarize the following QC run results for the data team in 3-5 bullet points:

{stats}

Focus on: total issues, most common check types, auto-fix rate, items needing human review.
Respond as JSON: {{"summary": "...", "key_actions": ["..."]}}
"""
