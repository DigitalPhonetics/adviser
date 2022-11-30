from typing import Dict, Any

class JSONSerializable:
    def to_json(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_json(cls, json: Dict[str, Any]):
        raise NotImplementedError