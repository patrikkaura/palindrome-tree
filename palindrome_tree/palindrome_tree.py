import pkg_resources
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict

import requests
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from palindrome_tree.models import PalindromesApiResponse, PalindromeTreeResult


class PalindromeTree:
    """
    Palindrome tree predicts locations through gradient boosted
    decision tree for further analysis via palindromes.ibp.cz
    """

    _BATCH_LIMIT: int = 1_000_000
    _FIXED_WINDOW_SIZE: int = 30
    _VALIDATION_BATCH_SIZE: int = 100
    _ENCODING: Dict[str, float] = {'A': 0.25, 'C': 0.5, 'G': 0.75, 'T': 1}
    _API_ENDPOINT: str = "http://palindromes.ibp.cz/rest/analyze/palindrome"

    def __init__(self):
        self._predicted_intervals: List[Tuple[int, int]] = []
        self._sequence: Optional[str] = None
        self._validation_collector: List['PalindromesApiResponse'] = []

    @property
    def results(self) -> Optional[pd.DataFrame]:
        """
        Return pandas dataframe with results if predicted intervals
        and sequence is set
        :return: pd.DataFrame with predicted intervals
        """
        if self._predicted_intervals and self._sequence:
            return self._process_results()
        return

    @property
    def validated_results(self) -> Optional[pd.DataFrame]:
        """
        Return pandas dataframe with validated palindromes if sequence and
        validation collector is set
        :return: pd.DataFrame with validated palindromes
        """
        if self._validation_collector and self._sequence:
            return pd.DataFrame(
                data=self._validation_collector,
                columns=asdict(self._validation_collector[0]).keys()
            )
        return

    def _sequence_convertor(self) -> np.array:
        """
        Convert sequences with class defined _ENCODING
        NOTE: don't change cause tree is trained to use exactly these parameters
        :return: numpy array with converted windows
        """
        converted_sequences = []

        for i in range(0, len(self._sequence) - self._FIXED_WINDOW_SIZE):
            converted = []
            for base in self._sequence[i:i + self._FIXED_WINDOW_SIZE]:
                converted.append(
                    self._ENCODING.get(base.upper(), 0)
                )
            converted_sequences.append(converted)

            if (i + 1) % self._BATCH_LIMIT == 0:
                yield np.array(converted_sequences)
                converted_sequences = []
        yield np.array(converted_sequences)

    @staticmethod
    def _init_tree() -> XGBClassifier:
        """
        Create model instance and load parameters from json model file
        :return: instance of gradient boosted tree
        """
        xgb = XGBClassifier()
        xgb.load_model(
            pkg_resources.resource_filename(
                __name__,
                '/model/palindrome-xgboost-tree.json'
            )
        )
        return xgb

    @staticmethod
    def _predict(*, model: XGBClassifier, converted_sequences: np.array) -> List[int]:
        """
        Return indexes with positive predictions
        :param model:
        :param converted_sequences:
        """
        results: List[int] = []
        predictions = model.predict(converted_sequences)
        predictions = list(predictions)

        for index, prediction in enumerate(predictions):
            if bool(prediction):
                results.append(index)
        return results

    def _create_intervals(self, predictions: List[int], batch_index: int) -> List[Tuple[int, int]]:
        """
        Create intervals used for merging
        :param predictions: predicted positions
        :param batch_index: index of currently processed batch
        :return: intervals with fixed window size
        """
        offset = batch_index * self._BATCH_LIMIT
        return [(i + offset, i + offset + self._FIXED_WINDOW_SIZE) for i in predictions]

    @staticmethod
    def _merge_results(*, results: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Return merged adjacent results from predict method
        :param results: predicted intervals
        :return: merged adjacent intervals
        """
        results = sorted(results, key=lambda x: x[0])
        i = 0
        for result in results:
            if result[0] > results[i][1]:
                i += 1
                results[i] = result
            else:
                results[i] = (results[i][0], result[1])
        return results[:i + 1]

    def _process_results(self) -> pd.DataFrame:
        """
        Process results and convert them into pandas dataframe
        :return: results in pandas dataframe table
        """
        data = []

        for position in self._predicted_intervals:
            start, end = position

            data.append(
                PalindromeTreeResult(
                    start=start,
                    end=end,
                    length=end - start,
                    sequence=self._sequence[start:end],
                )
            )
        return pd.DataFrame(
            data=data,
            columns=asdict(data[0]).keys()
        )

    def sequence_generator(self) -> str:
        """
        Return batch of sequences for API validation
        :return: batch of sequences
        """
        batch = ""

        for i, position in enumerate(self._predicted_intervals):
            batch += self._sequence[position[0]:position[1]].upper()
            batch += "\n"

            if (i + 1) % self._VALIDATION_BATCH_SIZE == 0:
                yield batch
                batch = ""
        yield batch

    def _validate_with_api(self) -> None:
        """
        Validate found regions for palindrome existence
        :return: results from palindrome api in dataframe
        """
        print("STARTING API VALIDATION PROCESS")

        for index, sequence_for_api in enumerate(self.sequence_generator()):
            print(f"VALIDATING BATCH NUMBER {index}")

            response = requests.post(
                url=self._API_ENDPOINT,
                json={
                    "cycle": True,
                    "dinucleotide": True,
                    "mismatches": "0,1",
                    "sequence": sequence_for_api,
                    "size": "6-30",
                    "spacer": "0-10",
                }
            )
            if response.ok:
                data = response.json()
                for palindrome in data['palindromes']:
                    self._validation_collector.append(
                        PalindromesApiResponse(**palindrome)
                    )
            else:
                print(f"VALIDATION OF BATCH NUMBER {index} FAILED")
        print("VALIDATION PROCESS FINISHED!")

    def analyse(self, sequence: str, validate_with_api: bool = False) -> None:
        """
        Analyse sequence for possible palindromes
        :param sequence: input sequence in FASTA or txt file
        :param validate_with_api: boolean flag tells if app should validate with API
        :return:
        """
        model = self._init_tree()

        self._predicted_intervals = []
        self._validation_collector = []
        self._sequence = sequence

        for batch_index, converted_sequences in enumerate(self._sequence_convertor()):
            predictions = self._predict(
                model=model, converted_sequences=converted_sequences
            )
            result_intervals = self._create_intervals(
                predictions=predictions, batch_index=batch_index
            )
            merged_intervals = self._merge_results(
                results=result_intervals
            )
            self._predicted_intervals.extend(
                merged_intervals
            )

        self._predicted_intervals = self._merge_results(results=self._predicted_intervals)

        print("DECISION TREE ANALYSIS COMPLETED")
        print(f"FOUND {len(self._predicted_intervals)} POSSIBLE PALINDROME REGIONS")

        if validate_with_api:
            self._validate_with_api()
