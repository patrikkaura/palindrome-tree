from dataclasses import dataclass, asdict
from typing import Dict

@dataclass
class PalindromesApiResponse:
    """
    Palindrome api response used for serializing 
    into dataframe format
    """
    original_index: int
    after: str
    before: str
    mismatches: int
    opposite: str
    position: int
    sequence: str
    signature: str
    spacer: str
    sequence: int
    stability_NNModel: Dict[str, int]

@dataclass
class PalindromeTreeResult:
    """
    Palindrome tree results used for serializing
    into dataframe format
    """
    position: int
    sequence: str