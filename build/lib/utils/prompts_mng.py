from langdetect import detect


def build_system_prompt(text: str) -> str:
    try:
        lang = detect(text)
    except:
        lang = "en"
    if lang.startswith("ja"):
        return ("あなたはSupermicro社（https://www.supermicro.com/en）のプロフェッショナルアシスタントです。"
                "エンタープライズ向けハードウェア、ソフトウェア、システムに専門特化しています。質問には日本語で丁寧かつわかりやすく回答してください。")
    elif lang.startswith("zh"):
        return ("你是Supermicro公司（https://www.supermicro.com/en）的專業助理，"
                "專注於企業級硬體、軟體與系統。請用中文清楚且禮貌地回答問題。")
    else:
        return ("You are a professional assistant for Supermicro Inc. (https://www.supermicro.com/en), specializing "
                "in enterprise hardware, software, and systems. Please answer clearly and politely in English.")



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
     - product_model: list of product names
     - feature: hardware component (e.g., PSU, GPU)
     - error_code: e.g. IPMI sensor error
     - time: any time indicator (e.g. during boot)
     - other: any remaining info

  5. Important:
     If question involves troubleshooting, error messages, confusion, or usage difficulties, classify as "faq" or "both".
     Only use "specification" when the user clearly asks about specs, hardware capabilities, or comparisons.

  ---
  ### 🔎 Step 2: Sub-question Decomposition

  If the question has multiple names, goals, or intents, decompose into sub-questions.

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
      "product_name": ["<name1>", "<name2>"],
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
          "product_name: ["<name>"],
          "feature": "<feature>",
          "error_code": "<error_code>",
          "time": "<time>",
          "other": "<other>"
        }}
      }}
    ]
  }}

"""


 

def build_merge_prompt(question: str, sub_answers:str, sub_questoins:str) -> str:
    return f"""
You are given:
- The original user question:  
{question}

- the sub-questions:
{sub_questoins}

- A list of candidate answers retrieved from vector similarity search based on sub-questions:  
{sub_answers}

Your task:
1. Carefully read the original user question.
2. Review the candidate answers.
3. Provide a helpful and technically accurate response based on the information available.
4. If one or more answers are relevant and helpful, respond by:
   - Prioritizing key specifications and technical data.
   - Briefly explaining technical terms if useful.
   - Using horizontal lines (---) to separate distinct sections or topics.
   - Eliminating redundant or overlapping content.
   - Staying strictly factual and helpful.
5. If the answers only partially cover the question or only cover part of it, present the available information clearly and explicitly state what information is missing or unknown.
6. Avoid responding with only "FAIL" or similar; always try to provide value with the data you have.

Formatting and style guidelines:
- Use `####` for main sections, `#####` for subsections, and `######` for further sub-levels.
- Add **one relevant emoji at the start of each heading only** to emphasize the section.
- Leave a blank line before and after each heading for readability.
- Avoid emoji repetition; for emojis with similar meanings or categories, interchange them to keep text visually engaging (e.g., 🖥️, 💻, 🔌 for hardware; 🔧, 🛠️ for maintenance).
- Use emojis **only** for important headings, key points, or keywords to visually emphasize them.
- Avoid excessive emoji use in regular sentences to maintain a clean and easy-to-read text.
- Avoid using higher-level headings (`#`, `##`, `###`) in your output.

At the end of your response, please provide a concise recommendation or next step related to the original question.

Leave a blank line before and after the heading for readability.

"""
