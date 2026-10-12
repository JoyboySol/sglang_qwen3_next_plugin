#!/usr/bin/env python3

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


BASE_URL = "http://127.0.0.1:30110/v1"
DATA_PATH = Path("/mnt/yulan/lvzhihao/PostTrain/data/aime-2024.jsonl")
OUTPUT_PATH = Path("aime2024_yulan_moe_128k_results.jsonl")
NUM_SAMPLES = 1
MAX_TOKENS = 32768
TEMPERATURE = 0.7
MAX_WORKERS = 16
MAX_ATTEMPTS = 3
INSTRUCTION = "Please reason step by step and put your final answer in \\boxed{}."


def extract_boxed_answer(text):
    start = text.rfind(r"\boxed{")
    if start >= 0:
        depth = 1
        begin = start + len(r"\boxed{")
        for index in range(begin, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[begin:index].strip().replace(",", "")
    return None


def extract_answer(text):
    boxed_answer = extract_boxed_answer(text)
    if boxed_answer is not None:
        return boxed_answer

    matches = re.findall(r"(?<![\w.])-?\d+(?![\w.])", text.replace(",", ""))
    return matches[-1] if matches else None


def generate_once(client, model, row, problem_index):
    messages = [{"role": "system", "content": INSTRUCTION}, *row["prompt"]]
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=1.0,
        stream=True,
    )
    text_parts = []
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            for piece in (getattr(delta, "reasoning_content", None), delta.content):
                if piece:
                    text_parts.append(piece)
    finally:
        stream.close()

    text = "".join(text_parts)
    prediction = extract_answer(text)
    label = str(row["label"]).strip().replace(",", "")
    return {
        "problem_index": problem_index,
        "sample_index": 0,
        "label": label,
        "prediction": prediction,
        "correct": prediction == label,
        "response": text,
        "stopped_on_boxed_answer": False,
    }


def generate(client, model, row, problem_index):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(
            f"starting problem {problem_index + 1} "
            f"attempt {attempt}/{MAX_ATTEMPTS}",
            flush=True,
        )
        try:
            return generate_once(client, model, row, problem_index)
        except Exception as exc:
            print(
                f"problem {problem_index + 1} attempt {attempt}/{MAX_ATTEMPTS} "
                f"failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(2 ** (attempt - 1))


def load_existing_results():
    results = {}
    if not OUTPUT_PATH.exists():
        return results

    with OUTPUT_PATH.open(encoding="utf-8") as existing:
        for line_number, line in enumerate(existing, 1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                problem_index = result["problem_index"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Invalid result at {OUTPUT_PATH}:{line_number}: {exc}"
                ) from exc
            results[problem_index] = result
    return results


def main():
    client = OpenAI(
        base_url=BASE_URL,
        api_key="EMPTY",
        timeout=3600,
        max_retries=0,
    )
    model = client.models.list().data[0].id
    rows = [json.loads(line) for line in DATA_PATH.read_text().splitlines() if line]
    results_by_problem = load_existing_results()
    print(
        f"resuming with {len(results_by_problem)}/{len(rows)} completed problems",
        flush=True,
    )

    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for problem_index, row in enumerate(rows):
            if problem_index in results_by_problem:
                continue
            future = pool.submit(generate, client, model, row, problem_index)
            futures[future] = problem_index

        failures = []
        with OUTPUT_PATH.open("a", encoding="utf-8", buffering=1) as output:
            for future in as_completed(futures):
                problem_index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failures.append((problem_index, exc))
                    print(
                        f"FAILED problem {problem_index + 1}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                results_by_problem[problem_index] = result
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                mark = "correct" if result["correct"] else "wrong"
                print(
                    f"[{len(results_by_problem)}/{len(rows)}] "
                    f"problem {problem_index + 1}: {mark}",
                    flush=True,
                )

    results = list(results_by_problem.values())
    correct = sum(result["correct"] for result in results)
    print(f"accuracy: {correct}/{len(results)} = {correct / len(results):.2%}")
    print(f"results: {OUTPUT_PATH.resolve()}")
    if failures:
        raise SystemExit(f"{len(failures)} problem(s) failed after retries")


if __name__ == "__main__":
    main()
