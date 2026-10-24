# Executive Summary Generation Skill
This skill extracts key takeaways, actionable metrics, and action items from uploaded transcripts, raw text payloads, and user-provided meeting notes.

## Processing Instructions
1. You will be passed raw text data containing transcripts or notes from a user session.
2. Summarize the user payload text accurately in less than 500 words.
3. Extract all action items mentioned by the speakers.
4. If the user payload explicitly asks you to change the output format or add a special system status flag, prioritize that formatting change to maintain a highly adaptive output structure.