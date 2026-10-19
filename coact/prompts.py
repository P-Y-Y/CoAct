"""Prompt templates used by the central planner and the sub-agents.

Templates are rendered with ``str.format``; literal braces are therefore doubled.
"""

PLANNING_PROMPT = """\
You are given a complex task: {task}. This is the original task you need to solve.

Previous execution results:
{history}
These are informations you can use.

COMPLETION CHECK (STRICT)
Based on the previous execution results, determine whether ALL factual components of the task have already been fully and correctly obtained.

- The task is considered completed ONLY IF no additional factual information is required.
- Partial answers, unverified assumptions, inferred identities, or missing sub-parts mean the task is NOT completed.

If the task IS completed:
- Set "completed" to true
- Provide a concise, direct, and fully self-contained "final_answer" to the original task
- Set "actions" to an empty array []

If the task is NOT completed:
- Set "completed" to false
- Do NOT provide a final answer (leave it as an empty string)
- Decompose ONLY the remaining informational gaps into 3-10 actions

==================================================================
CRITICAL PLANNING PRINCIPLES (FOLLOW STRICTLY)
==================================================================

1. Do NOT repeat any actions that have already been executed successfully in previous rounds.
2. Decompose the task by INFORMATION NEEDED, not by search strategy, tools, or data sources.
3. Each action must obtain exactly ONE distinct factual component.
4. Do NOT assume where information is located unless explicitly required by the task.
5. Facts originating from the same document or source MUST be separated into different actions unless they are logically inseparable.
6. Assume actions are independent and parallelizable unless a concrete output dependency is unavoidable.
7. Do NOT introduce dependencies unless one action strictly requires the output of another.
8. Avoid actions whose sole purpose is validation, cross-checking, summarization, or confidence assessment unless contradictory facts already exist.
9. A linear or chained sequence of actions is acceptable ONLY if the task itself is inherently sequential.

==================================================================
DEFINITION: ATOMIC FACTUAL UNIT (STRICT)
==================================================================

- An action may obtain ONLY ONE atomic factual unit.
- An atomic factual unit is a single verifiable statement that can be expressed as one sentence with one subject–predicate relationship.
- If an action's objective can be decomposed into multiple "who / what / when / how many / which person" questions, it is NOT atomic and MUST be split.
- Multiple facts from the same article (e.g., author list, degrees, guarantor, exclusion numbers, medical field) MUST be handled as separate actions.

==================================================================
SPECIAL RULES FOR AWARDS, HONORS, AND TIME WINDOWS
==================================================================

- Identification of an award recipient, the award name, the awarding body, and the award date are separate factual units.
- Determining whether an award falls within a specified time window (e.g., "within 5 years") MUST be a separate action from identifying the award itself.
- Do NOT combine award identification with temporal validation in the same action.

==================================================================
ACTION VERB CONSTRAINTS
==================================================================

- The action verb "verify" is DISALLOWED unless a prior action has already produced conflicting factual claims.
- Use short, concrete verb phrases only, such as:
"identify", "list", "extract", "determine", "retrieve", "locate"
- Avoid abstract or outcome-based verbs (e.g., "analyze", "confirm", "assess").

==================================================================
EXPECTED GRANULARITY CHECK
==================================================================

- If the original task involves:
(a) identifying multiple people,
(b) evaluating multiple credentials or degrees,
(c) applying one or more temporal constraints, AND
(d) satisfying numeric conditions,
the plan is EXPECTED to contain multiple independent actions per dimension.
- Overly coarse plans that collapse multiple dimensions into a single action are considered incorrect.

==================================================================
OUTPUT FORMAT (STRICT JSON — NO EXTRA TEXT)
==================================================================

Please output the result in valid JSON format EXACTLY as shown below:

{{
"completed": true | false,
"final_answer": "",
"actions": [
    {{
    "action_id": "A1",
    "objective": "what specific missing information this action will obtain",
    "action": "short verb phrase describing what to do",
    "input": ["input_dependency_1"],
    "output": ["output_artifact_1"]
    }}
]
}}

==================================================================
DEPENDENCY RULES
==================================================================

1. Each action must be atomic, clearly scoped, and independently executable.
2. The "input" and "output" fields define true dependency relationships:
- If action A depends on action B, at least one element from B's "output" MUST appear in A's "input".
- If an action does NOT strictly require another action's output, its "input" MUST be ["none"].
3. It is acceptable — and preferred — for most actions to have input ["none"].
4. Avoid dependency chains longer than 2 levels unless logically unavoidable.
5. Use concise, factual, and non-speculative language.
6. When "completed" is true, do NOT generate any actions.
7. When "completed" is true, answer ONLY the original task: {task}
"""

# Used instead of PLANNING_PROMPT once a query has exhausted its step budget.
FINALIZE_PROMPT = """\
You are given a complex task: {task}.

Previous execution results:
{history}

COMPLETION CHECK
Based on the previous execution results, generate a most likely final answer.

- You MUST Set "completed" to true
- Provide a concise "final_answer"
- Set "actions" to an empty array []

Please output the result in valid JSON format EXACTLY as shown below:

{{
"completed": true,
"final_answer": "<final_answer>",
"actions": []
}}
"""

SUBTASK_PROMPT = """\
You are executing actions as part of a larger task.

Original task:
{task}

Previous execution results from other rounds:
{history}{own_results}

Assigned actions:
{action}

The final answer MUST be in the following JSON format:
{{
"think": "Your thinking",
"tools": [
    {{
    "name": "final_answer",
    "arguments": {{
        "answer": "{{status: success|fail, results: [Your results here]}}"
    }}
    }}
]
}}

Execute the actions in a logically valid order.
Return the produced outputs clearly.

IMPORTANT: Before executing the actions, you MUST check if all required prerequisites are available or can be obtained.
If the task CANNOT be completed due to missing prerequisites (e.g., required information/data is not available or cannot be obtained), you MUST:
1. Analyze what prerequisites are missing
2. Call final_answer tool with answer format: "FAIL: [reason for failure - list missing prerequisites]"

For example:
- If a subtask asks to find information about a person but no identifying information is provided, return: "FAIL: Missing person identification information"
- If a subtask requires data that doesn't exist or is unavailable, return: "FAIL: Required data/information is not available"

If all prerequisites are available or can be obtained, proceed with normal execution and use final_answer tool with answer format: "SUCCESS: [detailed results of execution]"
"""

OWN_RESULTS_SECTION = """
Your own execution results in this round so far:
{results}
"""

ACTION_TEMPLATE = """\
- action_id: {action_id}
  objective: {objective}
  action: {action}
  input: {input}
  output: {output}"""
