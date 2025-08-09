def build_semantic_decompose_prompt(question: str) -> str:
    return f"""
  You are a technical assistant for Supermicro Inc., specialized in system and hardware specifications.

  ---
  ## 🔹 Product Model Classification:
  - Prefix "SYS-", "AS-", "ARS-" → classify as System
  - Prefix "X9", "X10", ..., "X14", "B13", "B14" → classify as Motherboard

  ---
  ## 🔹 Specification Intent Categories:
  - detail: asking about specific hardware specs or properties  
    → e.g., "What’s the PSU wattage of SYS-741A-T?"
  - comparison: comparing models or features  
    → e.g., "Which one is better: SYS-741A-T or SYS-521A-T?"
  - structural: asking about composition, hierarchy, or components within a system  
    → e.g., "Does SYS-741A-T include a GPU?"

  A question is likely structural if it contains both a system model and a component (e.g., TPM, GPU, PSU).

  ---
  ## 🔹 Specification Scope Rules (spec_scope)
  - Use "full_spec" when the question is about:
    - Listing multiple models meeting criteria
    - Full specification of a product or family
    - Mentions like "show all", "list", or "which models"
  - Use "partial_spec" when the question asks about:
    - Specific feature, spec, or capability

  ---
  ## 🔹 Instructions

  Given the user input:
  \"\"\"{question}\"\"\"

  Perform the following steps:

  ---
  ### 🧠 Step 1: Semantic Intent Classification

  1. Identify the primary intent(s) from:
     - greeting, non_product, specification, faq, both_spec_and_faq, clarification, comparison

  2. Map each intent to one of:
     {{
       "greeting": "greeting",
       "non_product": "non_product",
       "specification": "specification",
       "faq": "faq",
       "both_spec_and_faq": "both",
       "clarification": "faq",
       "comparison": "specification"
     }}

  3. If type is "specification" or "both", classify further:
     - spec_intent_category: detail | comparison | structural
     - spec_scope: full_spec | partial_spec

  4. Extract the following slots (use "" or [] if not found):
     - product_model: list of product models
     - feature: hardware component (e.g., PSU, GPU)
     - error_code: e.g. IPMI sensor error
     - time: any time indicator (e.g. during boot)
     - other: any remaining info

  5. Important:
     If question involves troubleshooting, error messages, confusion, or usage difficulties, classify as "faq" or "both".
     Only use "specification" when the user clearly asks about specs, hardware capabilities, or comparisons.

  ---
  ### 🔎 Step 2: Sub-question Decomposition

  If the question has multiple models, goals, or intents, decompose into sub-questions.

  Each sub-question must:
  - Have a single intent
  - Contain one product model (leave empty if missing)
  - Preserve slot values
  - Assign relevant query_sources:
    - "vector_spec": for general spec/feature lookup
    - "graph_spec": for structural information, full specifications, or comparisons
    - "vector_faq": for FAQ/support/troubleshooting or uncertainty/issues/technical support/if the user expresses confusion, missing info, or difficulty finding data
    - "sql_verified_component": for AVL, compatibility, or verified components

  ---
  ### 🧾 Output Format (strict JSON)

  Return only the following JSON:

  ```json
  {{
    "intents": [
      {{
        "intent": "<intent_name>",
        "type": "<type>",
        "reason": "<brief_reason>",
        "spec_intent_category": "<detail|comparison|structural|null>",
        "spec_scope": "<full_spec|partial_spec|null>"
      }}
    ],
    "slots": {{
      "product_model": ["<model1>", "<model2>"],
      "feature": "<feature>",
      "error_code": "<error_code>",
      "time": "<time>",
      "other": "<other>"
    }},
    "sub_questions": [
      {{
        "intent": "<intent_name>",
        "query_type": "<type>",
        "query": "<sub-question text>",
        "query_sources": ["vector_spec", "graph_spec", "vector_faq", "sql_verified_component"],
        "slots": {{
          "product_model": ["<model>"],
          "feature": "<feature>",
          "error_code": "<error_code>",
          "time": "<time>",
          "other": "<other>"
        }}
      }}
    ]
  }}

"""



def build_decompose_prompt(question: str, intents:str, slots:str) -> str:
    return f"""
You are a professional assistant for Supermicro Inc.

Given the original user question:
\"\"\"{question}\"\"\"

And the semantic analysis result:
Intents and their types:
{intents}
 
Slots:
{slots}
 
Your task:
- Carefully analyze the original question and decompose it only if necessary.
- Create a list of sub-questions, each linked to exactly one intent.
- Only create sub-questions for intents of these types: 
  "specification", "faq", "both_spec_and_faq", or "clarification".
- Avoid unnecessary or overly fine-grained decomposition; combine related intents into a single sub-question when appropriate.
- Each sub-question must have exactly one product_model in the slots; if none, use an empty list.
- Keep other slots unchanged.
- Do NOT create sub-questions for intents not listed above.

Additionally:
- Rewrite the original question into a simplified version focusing only on the actual product model name.
- Remove brand names (like “Supermicro”) unless they are absolutely essential to understanding.
- The goal is to make the rewritten question clearer for systems that need to extract or match exact product models.
Output **only** a JSON array of sub-questions, each formatted as:

[
  {{
    "intent": "<intent_name>",
    "query_type": "<type>",
    "query": "<sub_question_text>",
    "slots": {{
      "product_model": ["<single_product_model>"],
      "feature": "<feature>", // no comments, no nulls — use empty strings "" if the value is missing
      "error_code": "<error_code>", // no comments, no nulls — use empty strings "" if the value is missing
      "time": "<time>", // no comments, no nulls — use empty strings "" if the value is missing
      "other": "<other>" // no comments, no nulls — use empty strings "" if the value is missing
    }}
  }},
  ...
]

Do not add any explanation or extra text.
]
"""





def build_merge_prompt(question: str, sub_answers:str) -> str:
    return f"""
You are a professional assistant for Supermicro Inc. (https://www.supermicro.com/en), specializing in enterprise hardware, software, and systems.

You are given:
- The original user question:  
{question}

- A list of candidate answers retrieved from vector similarity search based on sub-questions:  
{sub_answers}

Your task:
1. Carefully read the original user question.
2. Review the candidate answers.
3. Determine whether any of the answers are:
   - Directly relevant to the original question
   - Technically accurate

Respond using one of the following two options:

Option 1: If none of the answers are relevant or helpful, respond with only the following word (no formatting, no punctuation):
FAIL

Option 2: If one or more answers are helpful:
- Write a concise, technically accurate, and professional summary.
- Prioritize key specifications and technical data.
- Explain technical terms briefly if useful.
- Use clean Markdown format for clarity (e.g., bullet points or tables).
- For lists (e.g., part numbers, models), use a Markdown table sorted alphabetically.
- Use horizontal lines (---) to separate distinct sections or topics.
- Eliminate redundant or overlapping content.
- Do not include any extra commentary, reasoning, or introductory phrases.
- Stay strictly factual and helpful.
- Use appropriate emojis (small icons) to highlight key items, for example:
  - 🖥️ for motherboard or hardware components
  - ⚙️ for features or capabilities
  - 🔧 for maintenance or troubleshooting
  - ⚠️ for warnings or limitations
  - 📅 for time-related information

Important:
- Do not wrap the entire response in code blocks.
- Avoid adding any line numbers or syntax highlighting markers.
- Output clean Markdown content only.


"""
