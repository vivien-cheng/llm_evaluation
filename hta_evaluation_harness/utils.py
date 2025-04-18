import json
import os
from typing import List, Dict, Any, Union

def load_json(file_path: str, load_by_line: bool = False) -> Union[Dict, List[Dict[str, Any]], None]:
    """
    Loads JSON data from a file.

    Args:
        file_path: Path to the JSON or JSONL file.
        load_by_line: If True, assumes JSONL format (one JSON object per line).
                      If False, assumes a single JSON object or list in the file.

    Returns:
        The loaded JSON data (dictionary or list of dictionaries), or None if file not found.
    Raises:
        json.JSONDecodeError: If the file content is invalid JSON/JSONL.
        IOError: If there's an issue reading the file.
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if load_by_line:
                data = [json.loads(line.strip()) for line in f if line.strip()]
            else:
                data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        raise
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        raise

def save_json(data: Union[Dict, List], file_path: str, save_by_line: bool = False, indent: int = 2):
    """
    Saves data to a JSON or JSONL file.

    Args:
        data: The data to save (dictionary or list).
        file_path: Path to the output file.
        save_by_line: If True, saves as JSONL format (one JSON object per line).
                      If False, saves as a single JSON object/list with indentation.
        indent: Indentation level for standard JSON output.
    Raises:
        IOError: If there's an issue writing to the file.
        TypeError: If the data is not JSON serializable.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True) # Ensure directory exists
        with open(file_path, 'w', encoding='utf-8') as f:
            if save_by_line:
                if not isinstance(data, list):
                    raise TypeError("Data must be a list to save in JSONL format.")
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                json.dump(data, f, ensure_ascii=False, indent=indent)
    except IOError as e:
        print(f"Error writing file {file_path}: {e}")
        raise
    except TypeError as e:
        print(f"Error serializing data to JSON: {e}")
        raise

# Example Usage (can be commented out or placed in a separate test/example script)
# if __name__ == "__main__":
#     # Create dummy data and file for testing
#     dummy_data_single = {"name": "test", "value": 1}
#     dummy_data_list = [{"id": 1, "text": "line 1"}, {"id": 2, "text": "line 2"}]
#     single_json_path = "temp_single.json"
#     jsonl_path = "temp_list.jsonl"

#     try:
#         print(f"Saving single JSON to {single_json_path}")
#         save_json(dummy_data_single, single_json_path)
#         loaded_single = load_json(single_json_path)
#         print(f"Loaded single JSON: {loaded_single}")
#         assert loaded_single == dummy_data_single

#         print(f"Saving JSONL to {jsonl_path}")
#         save_json(dummy_data_list, jsonl_path, save_by_line=True)
#         loaded_list = load_json(jsonl_path, load_by_line=True)
#         print(f"Loaded JSONL: {loaded_list}")
#         assert loaded_list == dummy_data_list

#     except Exception as e:
#         print(f"An error occurred during example usage: {e}")
#     finally:
#         # Clean up dummy files
#         if os.path.exists(single_json_path):
#             os.remove(single_json_path)
#         if os.path.exists(jsonl_path):
#             os.remove(jsonl_path)
#         print("Cleanup complete.")
