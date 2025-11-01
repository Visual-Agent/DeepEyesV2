RETURN_CODE_PROMPT = """Code execution result:
stdout:
```
{stdout}
```

stderr:
```
{stderr}
```

{image}
"""

RETURN_SEARCH_PROMPT = """<tool_response>
{search_result}
</tool_response>
"""