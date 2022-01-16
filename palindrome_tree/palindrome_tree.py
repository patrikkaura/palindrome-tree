import pkg_resources
from typing import List, Dict
from dataclasses import asdict

import requests
import pandas as pd
import numpy as np

from time import sleep
from xgboost import XGBClassifier

from palindrome_tree.models import PalindromesApiResponse, PalindromeTreeResult

class PalindromeTree():
    """
    Palindrome tree predicts locations thru gradient boosted
    decision tree for further analysis via palindromes.ibp.cz    
    """

    _FIXED_WINDOW_SIZE: int = 30
    _ENCODING: Dict[str, float] = {'A': 0.25, 'C': 0.5, 'G': 0.75, 'T': 1}
    _API_ENDPOINT: str = "http://palindromes.ibp.cz/rest/analyze/palindrome"

    def _sequence_convertor(self, *, sequence: str) -> np.array:
        """
        Convert sequences with class defined _ENCODING 
        NOTE: don't change cause tree is trained to use exactly this parameters
        :param sequence: input sequence for conversion
        :return: numpy array with converted windows 
        """
        converted_sequences = []

        for i in range(0, len(sequence) - self._FIXED_WINDOW_SIZE):    
            converted = []
            for base in sequence[i:i+self._FIXED_WINDOW_SIZE]:
                converted.append(
                    self._ENCODING.get(base, 0)
                )
            converted_sequences.append(converted)
        return np.array(converted_sequences)

    def _init_tree(self) -> XGBClassifier:
        """
        Create model instance and load parameters from json model file
        :param model_path: path to file with model params in json
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

    def _predict(self, *, model: XGBClassifier, converted_sequences: np.array) -> List[int]:
        """
        Return indexes with positive predictions
        :param model:
        :param converted_sequences:
        :return: 
        """
        results: List[int] = []
        predictions = model.predict(converted_sequences)
        predictions = list(predictions)

        for index, prediction in enumerate(predictions):
            if bool(prediction):   
                results.append(index)
        return results

    def _process_results(self, *, sequence: str, predicted_position: List[int]) -> pd.DataFrame:
        """
        Process results and convert them into pandas dataframe
        :param sequence: original sequence
        :param predicted_position: predicted position with possible palindromes
        :return: results in pandas dataframe table
        """
        data = []

        for position in predicted_position: 
            data.append(
                PalindromeTreeResult(
                    position=position,
                    sequence=sequence[position:position+self._FIXED_WINDOW_SIZE],
                )
            )
        return pd.DataFrame(
                data=data,
                columns=asdict(data[0]).keys()
        )

    def _validate_with_api(self, predicted_position: List[int], sequence: str) -> pd.DataFrame:
        """
        Validate found regions for palindrome existance
        :param predicted_position: predicted position with possible palindromes
        :param sequence: original sequence
        :return: results from palindrome api in dataframe
        """
        validation_collector: List['PalindromesApiResponse'] = []

        print("STARTING API VALIDATION PROCESS")

        for index, position in enumerate(predicted_position):
            print(f"VALIDATING {index} / {len(predicted_position)}")

            posible_sequence: str = sequence[position:position+self._FIXED_WINDOW_SIZE]

            response = requests.post(
                url=self._API_ENDPOINT,
                json={
                    "cycle": False,
                    "dinucleotide": False,
                    "mismatches": "0,1,2",
                    "sequence": posible_sequence,
                    "size": "6-30",
                    "spacer": "0-10",
                }
            )
            if response.ok:
                data = response.json()
                for palindrome in data['palindromes']:
                    validation_collector.append(
                        PalindromesApiResponse(**palindrome, original_index=index)
                    )
            else:
                print(f"VALIDATION FAILED :( PLEASE TRY MANUALLY {posible_sequence}")

        print("VALIDATION PROCESS FINISHED!")

        return pd.DataFrame(
                data=validation_collector,
                columns=asdict(validation_collector[0]).keys()
            )

    def analyse(self, sequence: str, check_with_api: bool = False) -> pd.DataFrame:
        """
        Analyse sequence for possible palindromes
        :param sequence:
        :param check_with_api:
        :return:
        """
        model = self._init_tree()
        converted_sequences = self._sequence_convertor(sequence=sequence)
        predicted_position = self._predict(model=model, converted_sequences=converted_sequences)

        print("DECISION TREE ANALYSIS COMPLETED")
        print(f"FOUND {len(predicted_position)} POSSIBLE PALINDROME REGIONS")

        if check_with_api:
            return self._validate_with_api(predicted_position=predicted_position, sequence=sequence)

        return self._process_results(
                    sequence=sequence,
                    predicted_position=predicted_position
                )
