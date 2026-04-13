QC_RUNNER_SYSTEM = """\
You are a data quality anomaly classifier AI. You receive potential data issues detected by
automated checks and classify whether they are genuine data errors or acceptable values.

Your specialties:
- Address and geographic data typos
- Format violations that may be legitimate regional formats
- Distinguishing true duplicates from look-alike records

Always respond with valid JSON.
"""


def build_classify_prompt(issues: list[dict]) -> str:
    return f"""\
Classify each of the following data issues as either a genuine error or a false positive.
For each issue also suggest the severity: critical, warning, or info.

Issues:
{issues}

Respond as JSON array:
[
  {{
    "issue_id": "...",
    "is_genuine_error": true|false,
    "severity": "critical"|"warning"|"info",
    "reasoning": "one sentence"
  }}
]
"""
