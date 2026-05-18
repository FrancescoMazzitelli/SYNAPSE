import sys
import os
from pathlib import Path
import threading
import time
from queue import Queue, Empty
from tqdm import tqdm
from typing import Dict, List
from types import SimpleNamespace

from preprocess import map_generic_data, get_questions_generic
from synthesize import load_config, SurveyAgent, build_generic_agents, infer_dtypes_with_llm
from gp_survey_engine import AgenticSurveyPipeline
from postprocess import ProcessSurveyResponse
from ladarag import LADARAG, LADARAGConfig
import json as json_module

from langroid.utils.configuration import settings

settings.quiet = True

# Static config folder
config_folder = "configs/Generic"

# Will be assigned later inside main_gp()
RUN_FOLDER = None

def postprocess_response(
    result_queue: Queue,
    stop_event: threading.Event,
    postprocessor: ProcessSurveyResponse,
    date_str: str):
    while not stop_event.is_set() or not result_queue.empty():
        try:
            result = result_queue.get(timeout=1.0)
            postprocessor.serialize_response(result)
        except Empty:
            continue
        except Exception as e:
            print(f"Error at postprocess thread: {e}")
            print(type(e))

def run_gp_survey(
    result_queue: Queue,
    stop_event: threading.Event,
    key_mapping: Dict,
    questions: Dict,
    agents: List[SurveyAgent],
    batch_size: int,
    timeout_per_batch: float = 600000,
    ladarag_engine=None
):
    try:
        for i in tqdm(range(0, len(agents), batch_size), desc="running batches"):
            batch = agents[i: i+batch_size]

            def batch_runner():
                nonlocal batch_results
                SE = AgenticSurveyPipeline(questions, batch, key_mapping, ladarag_engine=ladarag_engine)
                SE.run()
                batch_results = SE.results()

            batch_results = []
            thread = threading.Thread(target=batch_runner)
            thread.start()
            thread.join(timeout=timeout_per_batch)

            if thread.is_alive():
                print(f"Batch {i // batch_size} timed out. Skipping.")
                continue

            for r in batch_results:
                result_queue.put(r)

    except Exception as e:
        print(f"Exception in run_survey: {e}")
    finally:
        stop_event.set()
        
def main_gp(survey_path, dict_path):
    global RUN_FOLDER

    start_time = time.time()

    date_str = time.strftime("%Y%m%d_%H%M")

    dataset_name = os.path.basename(
        os.path.dirname(survey_path)
    )

    RUN_FOLDER = os.path.join(
        "run",
        f"{dataset_name}_{date_str}"
    )

    os.makedirs(RUN_FOLDER, exist_ok=True)

    model_conf, synth_conf, survey_conf, analysis = load_config(config_folder)

    synth = SimpleNamespace(**synth_conf)
    batch_size = synth.batch_size

    data_file = survey_path
    desc_file = dict_path
    
    print("Step 1: Mapping data headers to questions...")
    key_mapping = map_generic_data(data_file, desc_file)

    print("Step 1.5: Inferring response types with LLM...")
    key_mapping = infer_dtypes_with_llm(key_mapping, ollama_model=model_conf.get("chat_model", "phi4-reasoning:14b"))

    print("Step 2: Building agents with demographic profiles...")
    agents = build_generic_agents(
        config_folder=config_folder, 
        map=key_mapping, 
        data_path=data_file,
        desc_path=desc_file
    )
    
    print("Step 3: Preparing questions...")
    questions = get_questions_generic(key_desc_quest_mapping=key_mapping)

    print("Step 3.5: Initializing LADARAG engine (OD-trip retrieval)...")
    with open(os.path.join(config_folder, "config.json")) as f:
        full_config = json_module.load(f)
    ladarag_engine = LADARAG(LADARAGConfig(full_config.get("ladarag", {})))
    if ladarag_engine.is_enabled():
        print(f"  LADARAG enabled with {len(ladarag_engine.catalog.get_all())} service(s)")
    else:
        print("  LADARAG disabled (set 'ladarag.enabled: true' in config.json to enable)")

    print("Step 4: Initializing postprocessor...")
    postprocessor = ProcessSurveyResponse(
        config_folder,
        batch_size=synth_conf.get("batch_size", 10),
        RUN_FOLDER=RUN_FOLDER,
        source="GENERIC",
        date_str=date_str
    )

    print("Step 5: Running survey pipeline...")
    stop_event = threading.Event()
    result_queue = Queue()
    
    survey_thread = threading.Thread(
        target=run_gp_survey,
        args=(
            result_queue,
            stop_event,
            key_mapping,
            questions,
            agents,
            batch_size,
            600000,
            ladarag_engine))
    
    print("Step 6: Processing and saving results...")

    postprocessing_thread = threading.Thread(
        target=postprocess_response,
        args=(result_queue, stop_event, postprocessor, date_str)
    )

    survey_thread.start()
    postprocessing_thread.start()
    survey_thread.join()
    postprocessing_thread.join()
    
    end_time = time.time()
    duration_hour = (end_time - start_time) / 3600.00

    write_success = postprocessor.write_results(RUN_FOLDER, date_str)

    log_path = os.path.join(RUN_FOLDER, "log.txt")
    with open(log_path, "w") as f:
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        f.write(f"Duration (hours): {duration_hour:.2f}\n")
        f.write(f"Successful write to disk: {write_success}\n")
        f.write(f"Data file: {data_file}\n")
        f.write(f"Description file: {desc_file}\n")

        f.write("\nModel Configuration:\n")
        for key, value in model_conf.items():
            f.write(f"{key}: {value}\n")

        f.write("\nSynthesis Configuration:\n")
        for key, value in synth_conf.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n{'='*60}")
    print(f"Generic experiment completed successfully!")
    print(f"Results saved to: {RUN_FOLDER}")
    print(f"Duration: {duration_hour:.2f} hours")
    print(f"{'='*60}\n")
    
    return RUN_FOLDER

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python main.py <survey.csv> <dictionary.csv>")
        sys.exit(1)

    survey_path = sys.argv[1]
    dict_path = sys.argv[2]

    print(f"Survey file: {survey_path}")
    print(f"Dictionary file: {dict_path}")

    main_gp(survey_path, dict_path)
