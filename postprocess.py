from synthesize import load_config
from preprocess import generate_questions
from pathlib import Path
from dataclasses import asdict
import copy
import pandas as pd
import json
import re


class ProcessSurveyResponse:
    def __init__(self, config_folder: str, batch_size: int, RUN_FOLDER: str, source: str, date_str: str):
        self.batch_size = batch_size
        self.n_batches = 0
        self.batches_written = 1
        self.RUN_FOLDER = RUN_FOLDER
        self.source = source
        self.date_str = date_str
        self._prepare_dataset()

    def _prepare_dataset(self):
        self.multiple_choice_cols = []
        self.ground_truth_df = pd.DataFrame()
        ground_truth_cols = []
            
        self.synthetic_columns = ["agent_id", "serial_number", "agent_bio"]
        self.synthetic_columns.extend(ground_truth_cols)
        self.synthetic_dataset = pd.DataFrame(columns=self.synthetic_columns)
        self.batch_dataset = copy.deepcopy(self.synthetic_dataset)
        self.synthetic_asdict = []
        self.batch_asdict = []

    def serialize_response(self, agent_response):
        response_dict = asdict(agent_response)
        response_cols = [col.lower() for col in response_dict["logic_flow"]]
        new_row = {}

        new_row["agent_id"]       = agent_response.agent_id
        new_row["serial_number"]  = agent_response.serial_number
        new_row["agent_bio"]      = agent_response.agent_bio

        for col, val in zip(response_cols, response_dict["encoded_responses"]):
            if col in self.multiple_choice_cols:
                if isinstance(val, list):
                    new_row[col] = [self._coerce_to_int(x) for x in val]
                else:
                    new_row[col] = [self._coerce_to_int(val)]
            else:
                if isinstance(val, str):
                    val = val.replace('\n', '').replace('\r', '')
                    new_row[col] = val
                else:
                    new_row[col] = self._coerce_to_int(val)

        self.synthetic_dataset = pd.concat([self.synthetic_dataset, pd.DataFrame([new_row])], ignore_index=True)
        self.synthetic_asdict.append(response_dict)

        self.batch_dataset = pd.concat([self.batch_dataset, pd.DataFrame([new_row])], ignore_index=True)
        self.batch_asdict.append(response_dict)

        self._batch_write_results()

    def _coerce_to_int(self, value):
        try:
            if isinstance(value, int):
                return value
            return int(value)
        except (ValueError, TypeError):
            return value

    def write_results(self, RUN_FOLDER, date_str) -> bool | str:
        write_success = True
        try:
            self.synthetic_dataset.to_csv(Path(RUN_FOLDER) / "_".join((date_str, "results.csv")), index=False)
        except Exception as e:
            write_success = e

        try:
            with open(Path(RUN_FOLDER) / "_".join((date_str, "results.json")), "w") as f:
                json.dump(self.synthetic_asdict, f, indent=4)
        except Exception as e:
            write_success = e

        return write_success

    def _batch_write_results(self) -> None:
        if self.n_batches % self.batch_size == 0:
            try:
                self.batch_dataset.to_csv(Path(self.RUN_FOLDER) / f"batch_{self.batches_written}_{self.date_str}_results.csv", index=False)
            except Exception as e:
                print(e)

            try:
                with open(Path(self.RUN_FOLDER) / f"batch_{self.batches_written}_{self.date_str}_results.json", "w") as f:
                    json.dump(self.batch_asdict, f, indent=4)
            except Exception as e:
                print(e)

            self.batches_written += 1
            self.batch_dataset = self.batch_dataset[0:0]
            self.batch_asdict = []

        self.n_batches += 1


def _extract_first_int(x):
    if isinstance(x, str):
        m = re.search(r"\d+", x)
        if m:
            return int(m.group())
        else:
            return None
    return x
