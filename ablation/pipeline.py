"""Survey execution for the ablation study (new methods, no core edits).

The repository's ``main.run_gp_survey`` hard-codes ``AgenticSurveyPipeline``.
Instead of changing that method, this module keeps a small new copy of the same
batched execution loop that accepts a ``pipeline_factory``. Likewise
``postprocess_response`` is copied here so the driver never needs to import
``main``.
"""

from __future__ import annotations

import threading
from queue import Empty, Queue
from typing import Callable, Dict, List

from tqdm import tqdm

from gp_survey_engine import AgenticSurveyPipeline
from postprocess import ProcessSurveyResponse
from synthesize import SurveyAgent


def postprocess_response(
    result_queue: Queue,
    stop_event: threading.Event,
    postprocessor: ProcessSurveyResponse,
    date_str: str,
) -> None:
    while not stop_event.is_set() or not result_queue.empty():
        try:
            result = result_queue.get(timeout=1.0)
            postprocessor.serialize_response(result)
        except Empty:
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"Error at postprocess thread: {exc}")
            print(type(exc))


def run_gp_survey(
    result_queue: Queue,
    stop_event: threading.Event,
    key_mapping: Dict,
    questions: Dict,
    agents: List[SurveyAgent],
    batch_size: int,
    timeout_per_batch: float = 600000,
    pipeline_factory: Callable = AgenticSurveyPipeline,
) -> None:
    try:
        for i in tqdm(range(0, len(agents), batch_size), desc="running batches"):
            batch = agents[i : i + batch_size]

            def batch_runner():
                nonlocal batch_results
                engine = pipeline_factory(questions, batch, key_mapping)
                engine.run()
                batch_results = engine.results()

            batch_results = []
            thread = threading.Thread(target=batch_runner)
            thread.start()
            thread.join(timeout=timeout_per_batch)

            if thread.is_alive():
                print(f"Batch {i // batch_size} timed out. Skipping.")
                continue

            for result in batch_results:
                result_queue.put(result)

    except Exception as exc:  # noqa: BLE001
        print(f"Exception in run_survey: {exc}")
    finally:
        stop_event.set()
