import json
import re
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "MBZUAI/LaMini-Flan-T5-783M"

explain_tokenizer = None
explain_model = None
app = FastAPI()


def _load_local_model():
    global explain_tokenizer, explain_model

    import torch

    if explain_tokenizer is None or explain_model is None:
        explain_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        explain_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    return torch


def _generate_local_text(prompt: str, max_new_tokens: int = 256) -> str:
    torch = _load_local_model()
    inputs = explain_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.inference_mode():
        outputs = explain_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=False,
        )

    return explain_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def explain_topic(topic: str) -> str:
    torch = _load_local_model()
    input_text = (
        f"Explain the concept of '{topic}' in a simple and clear way "
        "for a school student."
    )
    inputs = explain_tokenizer(input_text, return_tensors="pt")

    with torch.inference_mode():
        outputs = explain_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )

    return explain_tokenizer.decode(outputs[0], skip_special_tokens=True)


def answer_question_with_gemini(question: str) -> str:
    try:
        prompt = (
            "Answer the student's question clearly and simply. "
            "If useful, explain the key idea step by step.\n\n"
            f"Question: {question}"
        )
        return _generate_local_text(prompt, max_new_tokens=256)
    except Exception as error:
        return f"An error in QnA: {error}"


def summarize_text(text: str) -> str:
    try:
        prompt = f"Summarize the following text in simple language:\n\n{text}"
        return _generate_local_text(prompt, max_new_tokens=200)
    except Exception as error:
        return f"An error in Summary: {error}"


def get_learning_recommendations(topic: str) -> str:
    prompt = f"""
You are an AI tutor. The student wants to learn about: {topic}.
Suggest a structured, adaptive learning path. Include key topics in learning
order and recommended resources. Organize it into beginner, intermediate, and
advanced levels when appropriate, and briefly explain how to adapt the path
based on the student's progress.
"""

    try:
        return _generate_local_text(prompt, max_new_tokens=400)
    except Exception as error:
        return f"Error occurred: {error}"


def clean_json_block(text: str) -> str:
    cleaned_text = text.strip()
    cleaned_text = re.sub(r"\A```(?:json)?\s*", "", cleaned_text, flags=re.IGNORECASE)
    return re.sub(r"\s*```\Z", "", cleaned_text).strip()


def generate_quiz(text: str) -> list:
    try:
        prompt = f"""
You are a quiz generator. Create exactly 3 multiple-choice questions from the passage.
Each question must have a "question" string, exactly 4 "options" strings, and an
"answer" string that exactly matches one of those options.
Return only a valid JSON array, with no Markdown fences or extra text, in this format:
[
  {{"question": "What is ...?", "options": ["A", "B", "C", "D"], "answer": "A"}}
]

Passage:
{text}
"""
        response_text = _generate_local_text(prompt, max_new_tokens=600)
        quiz = json.loads(clean_json_block(response_text))

        if not isinstance(quiz, list) or len(quiz) != 3:
            raise ValueError("The response must contain exactly 3 questions.")

        for question in quiz:
            if not isinstance(question, dict):
                raise ValueError("Each question must be a JSON object.")
            options = question.get("options")
            if (
                not isinstance(question.get("question"), str)
                or not isinstance(options, list)
                or len(options) != 4
                or not all(isinstance(option, str) for option in options)
                or question.get("answer") not in options
            ):
                raise ValueError("Each question must have 4 options and a matching answer.")

        return quiz
    except Exception as error:
        raise RuntimeError(f"Quiz generation failed: {error}") from error


@app.get("/qa")
async def answer_question(question: str = Query(...)):
    answer = await run_in_threadpool(answer_question_with_gemini, question)
    return {"answer": answer}


@app.post("/explain/")
async def explain_api(request: Request):
    data = await request.json()
    topic = data.get("topic") if isinstance(data, dict) else None
    if not isinstance(topic, str) or not topic.strip():
        return JSONResponse(
            content={"error": "Please provide a topic."},
            status_code=400,
        )

    topic = topic.strip()
    explanation = await run_in_threadpool(explain_topic, topic)
    return {"topic": topic, "explanation": explanation}


@app.post("/summarize/")
async def summarize_api(request: Request):
    data = await request.json()
    text = data.get("text") if isinstance(data, dict) else None
    if not isinstance(text, str) or not text.strip():
        return JSONResponse(
            content={"error": "Please provide text to summarize."},
            status_code=400,
        )

    summary = await run_in_threadpool(summarize_text, text.strip())
    return {"summary": summary}


@app.post("/quiz/")
async def quiz_api(request: Request):
    data = await request.json()
    text = data.get("text") if isinstance(data, dict) else None
    if not isinstance(text, str) or not text.strip():
        return JSONResponse(
            content={"error": "Please provide text for the quiz."},
            status_code=400,
        )

    try:
        quiz = await run_in_threadpool(generate_quiz, text.strip())
    except RuntimeError as error:
        return JSONResponse(content={"error": str(error)}, status_code=502)

    return {"quiz": quiz}


app.mount(
    "/",
    StaticFiles(directory=Path(__file__).resolve().parent / "static", html=True),
    name="frontend",
)
