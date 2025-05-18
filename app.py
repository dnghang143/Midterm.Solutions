import json
import os
from typing import Any, Dict, List, Tuple, Optional, Callable
import heapq
import hashlib
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, request, render_template, url_for, send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# TransformationLibraryManager class
class TransformationLibraryManager:
    def __init__(self):
        self.operators: Dict[str, Dict[str, Any]] = {}

    def TLMinsert(self, operator_spec: str) -> bool:
        try:
            op_data = json.loads(operator_spec)
            required_keys = {"name", "params", "description"}
            if not all(key in op_data for key in required_keys):
                return False
            op_data["function"] = self._get_function(op_data["name"])
            self.operators[op_data["name"]] = op_data
            return True
        except (json.JSONDecodeError, KeyError):
            return False

    def TLMsearch(self, operator_name: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        operator = self.operators.get(operator_name)
        if not operator:
            return None
        instantiated = operator.copy()
        if params:
            instantiated["params"].update(params)
        return instantiated

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self.operators.values())

    def load_from_file(self, filename: str) -> bool:
        try:
            with open(filename, "r") as f:
                operators = json.load(f)
            for op in operators:
                self.TLMinsert(json.dumps(op))
            return True
        except Exception:
            return False

    def _get_function(self, name: str) -> Callable:
        image_ops = {
            "nonuniform_scale_x2": lambda obj, params: scale_x(obj),
            "paint": lambda obj, params: paint(obj, params.get("color", (255, 0, 255))),
        }
        string_ops = {
            "replace": lambda obj, params: obj.replace(params.get("from_char", ""), params.get("to_char", ""), 1) if isinstance(obj, str) else obj,
            "insert": lambda obj, params: obj[:params.get("position", 0)] + params.get("char", "") + obj[params.get("position", 0):] if isinstance(obj, str) else obj,
            "delete": lambda obj, params: obj[:params.get("position", 0)] + obj[params.get("position", 0) + 1:] if isinstance(obj, str) and len(obj) > params.get("position", 0) else obj,
        }
        number_ops = {
            "increment": lambda obj, params: obj + params.get("value", 1) if isinstance(obj, (int, float)) else obj,
            "decrement": lambda obj, params: obj - params.get("value", 1) if isinstance(obj, (int, float)) else obj,
        }
        list_ops = {
            "append": lambda obj, params: obj + [params.get("value", 0)] if isinstance(obj, list) else obj,
            "pop": lambda obj, params: obj[:-1] if isinstance(obj, list) and len(obj) > 0 else obj,
        }
        return image_ops.get(name, string_ops.get(name, number_ops.get(name, list_ops.get(name, lambda obj, params: obj))))

# CostFunctionServer class
class CostFunctionServer:
    def __init__(self):
        self.cost_functions: Dict[str, Dict[str, Any]] = {}

    def Costinsert(self, cost_spec: str) -> bool:
        try:
            cost_data = json.loads(cost_spec)
            required_keys = {"name", "expression", "description"}
            if not all(key in cost_data for key in required_keys):
                return False
            self.cost_functions[cost_data["name"]] = cost_data
            return True
        except (json.JSONDecodeError, KeyError):
            return False

    def EvaluateCall(self, before: Any, after: Any, operator: Dict[str, Any]) -> float:
        try:
            cost_func = self.cost_functions.get(operator["name"])
            if not cost_func:
                return float("inf")
            if cost_func["name"] == "image_diff" and isinstance(before, Image.Image):
                return image_cost(before, after) / 1000
            if isinstance(before, str) and isinstance(after, str):
                return len(before) != len(after) or sum(1 for c1, c2 in zip(before, after) if c1 != c2)
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                return abs(before - after)
            if isinstance(before, list) and isinstance(after, list):
                return abs(len(before) - len(after)) + sum(1 for i, (x, y) in enumerate(zip(before, after)) if x != y)
            return float(cost_func["expression"])
        except Exception:
            return float("inf")

    def load_from_file(self, filename: str) -> bool:
        try:
            with open(filename, "r") as f:
                cost_funcs = json.load(f)
            for cf in cost_funcs:
                self.Costinsert(json.dumps(cf))
            return True
        except Exception:
            return False

# ObjectConvertor class
class ObjectConvertor:
    def __init__(self, tlm: TransformationLibraryManager, cfs: CostFunctionServer, max_steps: int = 20):
        self.tlm = tlm
        self.cfs = cfs
        self.max_steps = max_steps
        self.counter = 0
        self.intermediate_states = []

    def convert(self, o1: Any, o2: Any) -> Tuple[List[Dict[str, Any]], float, Any, List[Tuple[str, Any]]]:
        self.intermediate_states = []
        def heuristic(current: Any, goal: Any) -> float:
            if isinstance(current, str) and isinstance(goal, str):
                diff_len = abs(len(goal) - len(current))
                diff_chars = sum(1 for c1, c2 in zip(current, goal) if c1 != c2)
                return diff_len + diff_chars
            elif isinstance(current, Image.Image) and isinstance(goal, Image.Image):
                try:
                    current_resized = current.resize(goal.size)
                    return image_cost(current_resized, goal) / 1000
                except Exception:
                    return float("inf")
            elif isinstance(current, (int, float)) and isinstance(goal, (int, float)):
                return abs(current - goal)
            elif isinstance(current, list) and isinstance(goal, list):
                return abs(len(current) - len(goal)) + sum(1 for x, y in zip(current, goal) if x != y)
            return float("inf")

        queue = [(0 + heuristic(o1, o2), 0, self.counter, o1, [], o1)]
        self.counter += 1
        visited = set()

        def obj_hash(obj: Any) -> str:
            if isinstance(obj, Image.Image):
                return hashlib.md5(np.array(obj).tobytes()).hexdigest()
            return hashlib.md5(str(obj).encode()).hexdigest()

        goal_hash = obj_hash(o2)

        while queue:
            f_score, cost, _, current, path, current_obj = heapq.heappop(queue)
            h = obj_hash(current)
            if h in visited:
                continue
            visited.add(h)

            if h == goal_hash:
                return path, cost, current_obj, self.intermediate_states

            if len(path) >= self.max_steps:
                continue

            for op in self.tlm.get_all():
                try:
                    next_obj = op["function"](current, op["params"])
                    if isinstance(next_obj, Image.Image):
                        intermediate_path = os.path.join(app.config['UPLOAD_FOLDER'], f"intermediate_{self.counter}.png")
                        next_obj.save(intermediate_path)
                        self.intermediate_states.append((f"intermediate_{self.counter}.png", None))
                    else:
                        self.intermediate_states.append((None, str(next_obj)))
                    new_cost = cost + self.cfs.EvaluateCall(current, next_obj, op)
                    f_new = new_cost + heuristic(next_obj, o2)
                    heapq.heappush(queue, (f_new, new_cost, self.counter, next_obj, path + [op], next_obj))
                    self.counter += 1
                except Exception:
                    continue

        return [], float("inf"), None, self.intermediate_states

# Utility functions
def open_image(path):
    return Image.open(path).convert("RGB")

def scale_x(img):
    arr = np.array(img)
    new_arr = np.repeat(arr, 2, axis=1)
    return Image.fromarray(new_arr.astype(np.uint8))

def paint(img, color=(255, 0, 255)):
    arr = np.array(img)
    green_mask = (arr[:, :, 1] > arr[:, :, 0]) & (arr[:, :, 1] > arr[:, :, 2]) & (arr[:, :, 1] > 100)
    arr[green_mask] = color
    return Image.fromarray(arr.astype(np.uint8))

def image_cost(before: Image.Image, after: Image.Image) -> float:
    a1 = np.array(before).astype(np.int32)
    a2 = np.array(after).astype(np.int32)
    return float(np.sum(np.abs(a1 - a2)))

# Khởi tạo thư viện
tlm = TransformationLibraryManager()
cfs = CostFunctionServer()

# Thêm toán tử mặc định
operators = [
    # Hình ảnh
    {
        "name": "nonuniform_scale_x2",
        "params": {},
        "description": "Phóng to trục x gấp đôi"
    },
    {
        "name": "paint",
        "params": {"color": [255, 0, 255]},
        "description": "Tô màu magenta cho vùng xanh lá cây"
    },
    # Chuỗi
    {
        "name": "replace",
        "params": {"from_char": "a", "to_char": "b"},
        "description": "Thay thế ký tự trong chuỗi"
    },
    {
        "name": "insert",
        "params": {"char": "x", "position": 0},
        "description": "Chèn ký tự vào chuỗi"
    },
    {
        "name": "delete",
        "params": {"position": 0},
        "description": "Xóa ký tự trong chuỗi"
    },
    # Số
    {
        "name": "increment",
        "params": {"value": 1},
        "description": "Tăng giá trị số"
    },
    {
        "name": "decrement",
        "params": {"value": 1},
        "description": "Giảm giá trị số"
    },
    # Danh sách
    {
        "name": "append",
        "params": {"value": 0},
        "description": "Thêm phần tử vào danh sách"
    },
    {
        "name": "pop",
        "params": {},
        "description": "Xóa phần tử cuối cùng của danh sách"
    }
]

for op in operators:
    tlm.TLMinsert(json.dumps(op))

# Thêm hàm chi phí mặc định
cost_functions = [
    # Hình ảnh
    {
        "name": "nonuniform_scale_x2",
        "expression": "1.0",
        "description": "Chi phí cố định cho phóng to"
    },
    {
        "name": "paint",
        "expression": "0.5",
        "description": "Chi phí cố định cho tô màu"
    },
    {
        "name": "image_diff",
        "expression": "0.0",
        "description": "Chi phí dựa trên chênh lệch pixel"
    },
    # Chuỗi
    {
        "name": "replace",
        "expression": "1.0",
        "description": "Chi phí cố định cho thay thế"
    },
    {
        "name": "insert",
        "expression": "0.5",
        "description": "Chi phí cố định cho chèn"
    },
    {
        "name": "delete",
        "expression": "0.5",
        "description": "Chi phí cố định cho xóa"
    },
    # Số
    {
        "name": "increment",
        "expression": "0.5",
        "description": "Chi phí cố định cho tăng"
    },
    {
        "name": "decrement",
        "expression": "0.5",
        "description": "Chi phí cố định cho giảm"
    },
    # Danh sách
    {
        "name": "append",
        "expression": "0.5",
        "description": "Chi phí cố định cho thêm"
    },
    {
        "name": "pop",
        "expression": "0.5",
        "description": "Chi phí cố định cho xóa"
    }
]
for cf in cost_functions:
    cfs.Costinsert(json.dumps(cf))

@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    if request.method == "POST":
        mode = request.form.get("mode")
        if not mode:
            result["error"] = "Vui lòng chọn chế độ."
        elif mode == "string":
            start = request.form.get("start_str", "").strip()
            goal = request.form.get("goal_str", "").strip()
            if not start or not goal:
                result["error"] = "Vui lòng cung cấp cả chuỗi đầu vào và mục tiêu."
            else:
                converter = ObjectConvertor(tlm, cfs, max_steps=20)
                path, cost, final_obj, intermediates = converter.convert(start, goal)
                if not path:
                    result["error"] = f"Không tìm thấy đường dẫn biến đổi trong {converter.max_steps} bước."
                else:
                    result = {
                        "path": [f"{step['name']} (params: {step['params']})" for step in path],
                        "cost": cost,
                        "type": "string",
                        "start": start,
                        "goal": goal,
                        "final": str(final_obj),
                        "intermediates": intermediates
                    }
        elif mode == "image":
            start_file = request.files.get("start_img")
            goal_file = request.files.get("goal_img")
            if not start_file or not goal_file:
                result["error"] = "Vui lòng tải lên cả hai hình ảnh."
            else:
                start_path = os.path.join(app.config['UPLOAD_FOLDER'], start_file.filename)
                goal_path = os.path.join(app.config['UPLOAD_FOLDER'], goal_file.filename)
                start_file.save(start_path)
                goal_file.save(goal_path)
                try:
                    start_img = open_image(start_path)
                    goal_img = open_image(goal_path)
                    expected_width = start_img.width * 2
                    if goal_img.width != expected_width:
                        result["error"] = f"Kích thước hình ảnh mục tiêu không khớp. Chiều rộng mục tiêu phải là {expected_width}px."
                    else:
                        converter = ObjectConvertor(tlm, cfs, max_steps=20)
                        path, cost, final_obj, intermediates = converter.convert(start_img, goal_img)
                        if not path:
                            result["error"] = f"Không tìm thấy đường dẫn biến đổi trong {converter.max_steps} bước."
                        else:
                            final_path = os.path.join(app.config['UPLOAD_FOLDER'], f"final_{start_file.filename}")
                            final_obj.save(final_path)
                            result = {
                                "path": [f"{step['name']} (params: {step['params']})" for step in path],
                                "cost": cost,
                                "type": "image",
                                "start_img": start_file.filename,
                                "goal_img": goal_file.filename,
                                "final_img": f"final_{start_file.filename}",
                                "intermediates": intermediates
                            }
                except Exception as e:
                    result["error"] = f"Lỗi xử lý hình ảnh: {str(e)}"

    return render_template("index.html", result=result)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)