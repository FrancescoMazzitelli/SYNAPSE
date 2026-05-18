from preprocess import *
from langroid.language_models import LLMMessage
from synthesize import SurveyAgent
from typing import Tuple, Union, Dict, List, Set, Any
from dataclasses import dataclass
import json
import re
import random

GENERIC_PHRASES = ["other", "something else", "please describe", "other (specify)"]


def _is_generic_answer(answer: str) -> bool:
    """Check if an answer string is a generic/catch-all option."""
    lowered = answer.lower().strip()
    for phrase in GENERIC_PHRASES:
        if phrase in lowered:
            return True
    return False


def _has_specific_options(possible_answers: List[Any]) -> bool:
    """Check if there is at least one non-generic option available."""
    for a in possible_answers:
        if isinstance(a, str) and not _is_generic_answer(a):
            return True
        if not isinstance(a, str):
            return True
    return False


def _response_from_tool_message(response_doc):
    """
    Parse the LLM response to extract tool content and scrap.
    The response should be JSON with 'request' and 'value' fields.
    """
    if response_doc is None:
        return None, ""
    
    content = response_doc if isinstance(response_doc, str) else str(response_doc)
    
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "request" in parsed:
            return parsed, ""
    except (json.JSONDecodeError, TypeError):
        pass
    
    match = re.search(r'\{[^}]*"request"[^}]*\}', content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return parsed, content.replace(match.group(), "").strip()
        except json.JSONDecodeError:
            pass
    
    return None, content


@dataclass
class AgenticResponsePackage:
    """Result container for the Agentic Pipeline."""
    agent_id: str
    agent_bio: str
    serial_number: str
    logic_flow: List[str]
    parsed_responses: List[str | int | List[int]]
    responses_scraps: List[str]
    encoded_responses: List[int | str]
    tool_dtypes: List[str]
    dtype_matches: List[bool]
    n_questions: int
    bad_iteration: bool
    tool_choices: List[str]


class AgenticSurveyPipeline:
    """
    Enhanced Survey Pipeline where agents intelligently select tools
    based on the answer sets provided.
    """
    def __init__(self, questions: Dict, agents: List[SurveyAgent], mapping_dict: Dict,
                 ladarag_engine=None):
        self.questions = questions
        self.agents = agents
        self.mapping_dict = mapping_dict
        self.respondent_summaries = []
        self.ladarag = ladarag_engine

    def _generate_memory_context(self, history: List[Tuple[str, str]]) -> str:
        if not history:
            return ""
        
        recent = history[-5:]
        context_block = "\n\n### YOUR PREVIOUS RESPONSES (for consistency):\n"
        for q, a in recent:
            context_block += f"Question: {q}\nAnswer: {a}\n"
        return context_block

    def _create_agent_prompt(self, question_text: str, question_info: Dict, memory_prompt: str,
                              od_context: str = "") -> str:
        possible_answers = question_info.get("possible_answers", [])
        answer_count = question_info.get("answer_count", 0)
        dtype = question_info.get("dtype", "TEXT")

        if possible_answers:
            if len(possible_answers) > 60:
                sampled_answers = random.sample(possible_answers, 60)
            else:
                sampled_answers = possible_answers
        else:
            sampled_answers = []
        
        from preprocess import format_answer_set_for_prompt
        formatted_answers = format_answer_set_for_prompt(sampled_answers) if possible_answers else "Open-ended response (no predefined answers)"
        
        od_section = od_context if od_context else ""

        if dtype == "SINGLE":
            format_instruction = '{"request": "single", "value": "<one option from the list>"}'
            extra_rules = "- Select exactly ONE option from the list."
        elif dtype == "MULTIPLE":
            format_instruction = '{"request": "multiple", "value": ["<option1>", "<option2>", ...]}'
            extra_rules = "- Select ALL options that apply. Return them as a JSON array."
        elif dtype == "NUMERIC":
            format_instruction = '{"request": "numeric", "value": <number>}'
            extra_rules = "- Provide a numeric value (integer or decimal) that fits the persona."
        else:
            format_instruction = '{"request": "text", "value": "<your answer>"}'
            extra_rules = "- Provide a free-text answer."

        prompt = f"""
### CURRENT QUESTION:
{question_text}

### POSSIBLE ANSWERS:
{formatted_answers}

### RECENT ANSWERS (memory for consistency):
{memory_prompt or 'None'}
{od_section}
### RESPONSE INSTRUCTIONS (STRICT)
You must respond with ONLY valid JSON.
Do NOT include explanations.
Do NOT include markdown.
Do NOT include any text before or after the JSON.
Do NOT describe the tool.
Do NOT repeat the question.

Expected response format:
{format_instruction}

### STRICT RULES:
- You are roleplaying as the person described in the PERSONA section. Every answer must reflect that person's sociodemographic background, lifestyle, and likely behaviors.
- Your answer MUST be consistent with all previous answers in this survey.
- If a possible answer would contradict previous responses, abstain by returning "N/A".
- {extra_rules}
- Always return ONLY one JSON object in the expected format above.
- NEVER invent new answer options or modify the provided possible answers.
- Keep your answers consistent across all questions in this survey, not just the current topic.
- CRITICAL: You MUST choose a substantive answer based on your persona. Do NOT select non-answers like "I prefer not to answer", "Appropriate skip", "Don't know", "Not ascertained", or similar refusal options. Always pick an actual response that reflects your assigned profile.
"""
        return prompt

    def run(self):
        for agent in self.agents:
            history = []
            logic_flow = []
            parsed_responses = []
            scraps = []
            tool_dtypes = []
            encoded_responses = []
            tool_choices = []
            
            for q_key in self.questions.keys():
                question_info = self.questions[q_key]
                q_text = question_info.get("text", q_key)
                
                q_desc = self.mapping_dict.get(q_key, q_key)
                if isinstance(q_desc, tuple):
                    q_desc = q_desc[0]
                
                od_context = ""
                if self.ladarag and self.ladarag.is_enabled():
                    try:
                        is_od = self.ladarag.classify_question(
                            question_text=q_text,
                            question_desc=q_desc,
                            agent_bio=agent.bio
                        )
                        if is_od:
                            ladarag_result = self.ladarag.query(
                                question_text=q_text,
                                question_desc=q_desc,
                                agent_bio=agent.bio
                            )
                            od_context = self.ladarag.format_od_context(ladarag_result)
                    except Exception as e:
                        print(f"[LADARAG] Error processing question '{q_key}' for agent {agent.agent_id}: {e}")
                
                memory_prompt = self._generate_memory_context(history)
                expected_dtype = question_info.get("dtype", "TEXT")
                possible_answers = question_info.get("possible_answers", [])
                retry_feedback_given = False

                for attempt in range(2):
                    prompt = self._create_agent_prompt(q_text, question_info, memory_prompt, od_context)
                    if retry_feedback_given:
                        prompt += "\n\n### FEEDBACK\nYour previous response selected a generic or non-specific option, or was not a valid answer. Please carefully review ALL available options and select the most appropriate specific option that accurately reflects your profile. Do not default to catch-all options.\n"

                    response_doc = agent.llm_response(prompt)
                    
                    tool_content, scrap = _response_from_tool_message(response_doc)
                    agent.clear_history()
                    
                    val = "N/A"
                    t_type = expected_dtype
                    if tool_content:
                        if isinstance(tool_content, dict):
                            t_type = tool_content.get("request", expected_dtype)
                            val = next((v for k, v in tool_content.items() if k != "request"), "N/A")
                        elif isinstance(tool_content, str):
                            t_type = expected_dtype
                            val = tool_content
                    
                    if possible_answers and val != "N/A":
                        if expected_dtype == "MULTIPLE":
                            if not isinstance(val, list):
                                val = [val]
                            validated_val = [v for v in val if v in possible_answers]
                            if not validated_val:
                                val = [possible_answers[0]] if possible_answers else ["N/A"]
                            else:
                                val = validated_val
                        else:
                            if isinstance(val, list):
                                val = val[0] if val else "N/A"
                            if val not in possible_answers:
                                try:
                                    if isinstance(val, str) and any(isinstance(a, (int, float)) for a in possible_answers):
                                        val = int(val) if val.isdigit() else float(val)
                                    elif isinstance(val, (int, float)) and any(isinstance(a, str) for a in possible_answers):
                                        val = str(val)
                                except:
                                    pass
                                
                                if val not in possible_answers:
                                    val = possible_answers[0] if possible_answers else "N/A"

                    # Decide whether to retry (only on first attempt)
                    if attempt == 0:
                        should_retry = False
                        if val == "N/A" and _has_specific_options(possible_answers):
                            should_retry = True
                        elif isinstance(val, str) and _is_generic_answer(val) and _has_specific_options(possible_answers):
                            should_retry = True
                        elif isinstance(val, list):
                            if any(isinstance(v, str) and _is_generic_answer(v) for v in val) and _has_specific_options(possible_answers):
                                should_retry = True
                        
                        if should_retry:
                            retry_feedback_given = True
                            continue
                    
                    break
                
                history.append((q_desc, str(val)))
                logic_flow.append(q_key)
                parsed_responses.append(val)
                scraps.append(scrap)
                tool_dtypes.append(t_type)
                encoded_responses.append(val)
                tool_choices.append(t_type)

            self.respondent_summaries.append(AgenticResponsePackage(
                agent_id=agent.config.name,
                agent_bio=agent.bio,
                serial_number=agent.serial_number,
                logic_flow=logic_flow,
                parsed_responses=parsed_responses,
                responses_scraps=scraps,
                encoded_responses=encoded_responses,
                tool_dtypes=tool_dtypes,
                dtype_matches=[True] * len(logic_flow),
                n_questions=len(logic_flow),
                bad_iteration=False,
                tool_choices=tool_choices
            ))

    def results(self):
        return self.respondent_summaries
    
    def get_tool_choice_statistics(self) -> Dict[str, Any]:
        stats = {
            "total_questions": 0,
            "tool_usage": {},
            "by_question": {}
        }
        
        for summary in self.respondent_summaries:
            for q_key, tool_choice in zip(summary.logic_flow, summary.tool_choices):
                stats["total_questions"] += 1
                
                if tool_choice not in stats["tool_usage"]:
                    stats["tool_usage"][tool_choice] = 0
                stats["tool_usage"][tool_choice] += 1
                
                if q_key not in stats["by_question"]:
                    stats["by_question"][q_key] = {}
                if tool_choice not in stats["by_question"][q_key]:
                    stats["by_question"][q_key][tool_choice] = 0
                stats["by_question"][q_key][tool_choice] += 1
        
        return stats
