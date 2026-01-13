from typing import Any, Dict, List, Union

# A common way to represent a generic JSON object
JSONValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JSONObject = Dict[str, JSONValue] | List[Dict[str, JSONValue]]
