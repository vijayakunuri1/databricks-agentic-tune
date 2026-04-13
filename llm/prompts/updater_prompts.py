UPDATER_SYSTEM = """\
You are a data correction rationale generator. Given an original value and the proposed
corrected value for an address or geographic field, explain the correction in one clear sentence
that a human reviewer can understand.

Keep explanations factual and concise. Do not invent information not present in the data.
Always respond with valid JSON.
"""


def build_rationale_prompt(corrections: list[dict]) -> str:
    return f"""\
For each correction below, write a one-sentence explanation of why the correction was made.

Corrections:
{corrections}

Respond as JSON array:
[
  {{"issue_id": "...", "rationale": "..."}}
]
"""
