import json

def validate_json(file_path):
    with open(file_path, 'r') as file:
        data = ""
        chunk_size = 1024 * 1024  # 1 MB chunks
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            data += chunk
            try:
                json.loads(data)
                data = ""
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}")
                error_position = e.pos
                file.seek(error_position)
                context = file.read(100)
                print(f"Context around error: {context}")
                break

# Replace 'path/to/your/file.json' with your actual file path
validate_json('gat/data/neo4j_graph.json')
