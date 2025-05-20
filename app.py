import json
import os
from typing import Any, Dict, List, Tuple, Optional, Callable
import heapq
import hashlib
import uuid
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# TransformationLibraryManager class
class TransformationLibraryManager:
    def __init__(self):
        self.operators: Dict[str, Dict[str, Any]] = {}

    def TLMinsert(self, operator_spec: str) -> bool:
        """Thêm toán tử vào thư viện từ cú pháp JSON."""
        try:
            op_data = json.loads(operator_spec)
            required_keys = {"name", "params", "description"}
            if not all(key in op_data for key in required_keys):
                return False
            op_data["function"] = self._get_function(op_data["name"])
            self.operators[op_data["name"]] = op_data
            return True
        except json.JSONDecodeError:
            return False

    def TLMsearch(self, operator_name: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Tìm và trả về toán tử đã khởi tạo."""
        operator = self.operators.get(operator_name)
        if not operator:
            return None
        instantiated = operator.copy()
        if params:
            instantiated["params"].update(params)
        return instantiated

    def get_all(self) -> List[Dict[str, Any]]:
        """Trả về tất cả toán tử."""
        return list(self.operators.values())

    def _get_function(self, name: str) -> Callable:
        """Liên kết tên toán tử với hàm thực thi."""
        image_ops = {
            "nonuniform_scaling": lambda obj, params: scale_image(obj, params.get("scale_x", 2), params.get("scale_y", 1)) if isinstance(obj, Image.Image) else obj,
            "paint": lambda obj, params: paint(
                obj,
                input_color=tuple(params.get("input_color", [0, 0, 0])),
                output_color=tuple(params.get("output_color", [0, 0, 0])),
                threshold=params.get("threshold", 100)
            ) if isinstance(obj, Image.Image) else obj,
        }
        string_ops = {
            "replace": lambda obj, params: obj.replace(params.get("from_char", ""), params.get("to_char", ""), 1) if isinstance(obj, str) and params.get("from_char") in obj else obj,
            "insert": lambda obj, params: obj[:params.get("position", 0)] + params.get("char", "") + obj[params.get("position", 0):] if isinstance(obj, str) else obj,
            "delete": lambda obj, params: obj[:params.get("position", 0)] + obj[params.get("position", 0) + 1:] if isinstance(obj, str) and len(obj) > params.get("position", 0) else obj,
        }
        return image_ops.get(name, string_ops.get(name, lambda obj, params: obj))

# CostFunctionServer class
class CostFunctionServer:
    def __init__(self):
        self.cost_functions: Dict[str, Dict[str, Any]] = {}

    def Costinsert(self, cost_spec: str) -> bool:
        """Thêm hàm chi phí vào thư viện từ cú pháp JSON."""
        try:
            cost_data = json.loads(cost_spec)
            required_keys = {"name", "expression", "description"}
            if not all(key in cost_data for key in required_keys):
                return False
            self.cost_functions[cost_data["name"]] = cost_data
            return True
        except json.JSONDecodeError:
            return False

    def EvaluateCall(self, before: Any, after: Any, operator: Dict[str, Any]) -> float:
        """Tính chi phí của toán tử."""
        try:
            cost_func = self.cost_functions.get(operator["name"])
            if not cost_func:
                return float("inf")
            if isinstance(before, Image.Image) and isinstance(after, Image.Image):
                if cost_func["name"] == "image_diff":
                    return image_cost(before, after) / 1000
                return float(cost_func["expression"])
            if isinstance(before, str) and isinstance(after, str):
                return float(cost_func["expression"]) if before != after else 0.0
            return float(cost_func["expression"])
        except (KeyError, ValueError):
            return float("inf")

# ObjectConvertor class
class ObjectConvertor:
    def __init__(self, tlm: TransformationLibraryManager, cfs: CostFunctionServer, max_steps: int = 200):
        self.tlm = tlm
        self.cfs = cfs
        self.max_steps = max_steps
        self.counter = 0
        self.intermediate_states = []

    def heuristic(self, current: Any, goal: Any) -> float:
        """Tính giá trị heuristic để hướng dẫn thuật toán A*."""
        if isinstance(current, str) and isinstance(goal, str):
            diff_len = abs(len(goal) - len(current))
            diff_chars = sum(1 for c1, c2 in zip(current.ljust(len(goal), ' '), goal.ljust(len(current), ' ')) if c1 != c2)
            return diff_len * 0.5 + diff_chars * 1.0
        elif isinstance(current, Image.Image) and isinstance(goal, Image.Image):
            try:
                current_hist = np.array(current.histogram())
                goal_hist = np.array(goal.histogram())
                return float(np.sum(np.abs(current_hist - goal_hist))) / 1000
            except Exception:
                return float("inf")
        return float("inf")

    def convert(self, o1: Any, o2: Any) -> Tuple[List[Dict[str, Any]], float, Any, List[Tuple[str, Any]]]:
        """Tìm chuỗi biến đổi tối ưu từ o1 đến o2."""
        self.intermediate_states = []
        queue = [(0 + self.heuristic(o1, o2), 0, self.counter, o1, [], o1)]
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
                # Thử tất cả các tham số có thể cho toán tử
                if isinstance(current, str) and isinstance(o2, str):
                    if op["name"] == "replace":
                        # Thử thay đổi từng ký tự trong chuỗi hiện tại
                        for i, (c1, c2) in enumerate(zip(current.ljust(len(o2), ' '), o2.ljust(len(current), ' '))):
                            if c1 != c2 and c1 != ' ':
                                op_params = op["params"].copy()
                                op_params.update({"from_char": c1, "to_char": c2})
                                next_obj = op["function"](current, op_params)
                                self._process_next_state(queue, current, next_obj, op, op_params, path, cost, o2)
                    elif op["name"] == "insert" and len(current) < len(o2):
                        # Thử chèn ký tự từ chuỗi mục tiêu
                        for pos in range(len(current) + 1):
                            op_params = op["params"].copy()
                            op_params.update({"char": o2[min(pos, len(o2)-1)], "position": pos})
                            next_obj = op["function"](current, op_params)
                            self._process_next_state(queue, current, next_obj, op, op_params, path, cost, o2)
                    elif op["name"] == "delete" and len(current) > len(o2):
                        # Thử xóa từng ký tự
                        for pos in range(len(current)):
                            op_params = op["params"].copy()
                            op_params.update({"position": pos})
                            next_obj = op["function"](current, op_params)
                            self._process_next_state(queue, current, next_obj, op, op_params, path, cost, o2)
                elif isinstance(current, Image.Image) and isinstance(o2, Image.Image):
                    op_params = op["params"].copy()
                    if op["name"] == "nonuniform_scaling":
                        # Điều chỉnh tỷ lệ dựa trên kích thước mục tiêu
                        scale_x = o2.size[0] / current.size[0] if current.size[0] != 0 else 1
                        scale_y = o2.size[1] / current.size[1] if current.size[1] != 0 else 1
                        op_params.update({"scale_x": scale_x, "scale_y": scale_y})
                        next_obj = op["function"](current, op_params)
                        self._process_next_state(queue, current, next_obj, op, op_params, path, cost, o2)
                    elif op["name"] == "paint":
                        # Thử phát hiện màu thay đổi
                        input_color, output_color = detect_dominant_color_change(current, o2)
                        op_params.update({"input_color": input_color, "output_color": output_color, "threshold": 100})
                        next_obj = op["function"](current, op_params)
                        self._process_next_state(queue, current, next_obj, op, op_params, path, cost, o2)

        return [], float("inf"), None, self.intermediate_states

    def _process_next_state(self, queue: List, current: Any, next_obj: Any, op: Dict, op_params: Dict, path: List, cost: float, o2: Any) -> None:
        """Xử lý trạng thái tiếp theo và thêm vào hàng đợi."""
        if isinstance(next_obj, Image.Image):
            intermediate_filename = f"intermediate_{uuid.uuid4()}.png"
            intermediate_path = os.path.join(app.config['UPLOAD_FOLDER'], intermediate_filename)
            next_obj.save(intermediate_path)
            self.intermediate_states.append((intermediate_filename, None))
        else:
            self.intermediate_states.append((None, str(next_obj)))
        new_cost = cost + self.cfs.EvaluateCall(current, next_obj, op)
        f_new = new_cost + self.heuristic(next_obj, o2)
        heapq.heappush(queue, (f_new, new_cost, self.counter, next_obj, path + [dict(op, params=op_params)], next_obj))
        self.counter += 1

# Utility functions
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_image(file, upload_folder: str) -> str:
    if not allowed_file(file.filename):
        raise ValueError("Chỉ hỗ trợ định dạng PNG và JPEG.")
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(upload_folder, unique_filename)
    file.save(file_path)
    return unique_filename

def open_image(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Không thể mở hình ảnh: {str(e)}")

def scale_image(img: Image.Image, scale_x: float, scale_y: float) -> Image.Image:
    arr = np.array(img)
    new_width = int(arr.shape[1] * scale_x)
    new_height = int(arr.shape[0] * scale_y)
    new_arr = np.zeros((new_height, new_width, arr.shape[2]), dtype=np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            orig_i = int(i / scale_y) if scale_y != 0 else i
            orig_j = int(j / scale_x) if scale_x != 0 else j
            orig_i = min(orig_i, arr.shape[0] - 1)
            orig_j = min(orig_j, arr.shape[1] - 1)
            new_arr[i, j] = arr[orig_i, orig_j]
    return Image.fromarray(new_arr.astype(np.uint8))

def paint(img: Image.Image, input_color: Tuple[int, int, int], output_color: Tuple[int, int, int], threshold: int = 100) -> Image.Image:
    arr = np.array(img)
    color_diff = np.abs(arr - np.array(input_color))
    mask = np.sum(color_diff, axis=2) < threshold
    arr[mask] = output_color
    return Image.fromarray(arr.astype(np.uint8))

def detect_dominant_color_change(start_img: Image.Image, goal_img: Image.Image) -> Tuple[List[int], List[int]]:
    try:
        start_hist = np.array(start_img.histogram())
        goal_hist = np.array(goal_img.histogram())
        diff_hist = np.abs(start_hist - goal_hist)
        max_diff_idx = np.argmax(diff_hist)
        channel = max_diff_idx // 256
        value_start = max_diff_idx % 256
        goal_channel_hist = goal_hist[channel * 256:(channel + 1) * 256]
        value_goal = np.argmax(goal_channel_hist)
        input_color = [0, 0, 0]
        output_color = [0, 0, 0]
        input_color[channel] = value_start
        output_color[channel] = value_goal
        return input_color, output_color
    except Exception:
        return [0, 255, 0], [255, 0, 255]  # Mặc định: xanh lá → hồng

def image_cost(before: Image.Image, after: Image.Image) -> float:
    try:
        a1 = np.array(before).astype(np.int32)
        a2 = np.array(after).astype(np.int32)
        return float(np.sum(np.abs(a1 - a2)))
    except Exception:
        return float("inf")

# Khởi tạo thư viện
tlm = TransformationLibraryManager()
cfs = CostFunctionServer()

# Định nghĩa toán tử
operators = [
    # Chuỗi
    {"name": "replace", "params": {"from_char": "", "to_char": ""}, "description": "Thay thế ký tự trong chuỗi"},
    {"name": "insert", "params": {"char": "", "position": 0}, "description": "Chèn ký tự vào chuỗi"},
    {"name": "delete", "params": {"position": 0}, "description": "Xóa ký tự trong chuỗi"},
    # Hình ảnh
    {"name": "nonuniform_scaling", "params": {"scale_x": 2.0, "scale_y": 1.0}, "description": "Phóng to/thu nhỏ theo tỷ lệ"},
    {"name": "paint", "params": {"input_color": [0, 0, 0], "output_color": [0, 0, 0], "threshold": 100}, "description": "Tô màu dựa trên màu đầu vào"},
]

for op in operators:
    tlm.TLMinsert(json.dumps(op))

# Định nghĩa hàm chi phí
cost_functions = [
    # Chuỗi
    {"name": "replace", "expression": "1.0", "description": "Chi phí cố định cho thay thế"},
    {"name": "insert", "expression": "0.5", "description": "Chi phí cố định cho chèn"},
    {"name": "delete", "expression": "0.5", "description": "Chi phí cố định cho xóa"},
    # Hình ảnh
    {"name": "nonuniform_scaling", "expression": "1.0", "description": "Chi phí cố định cho phóng to/thu nhỏ"},
    {"name": "paint", "expression": "0.5", "description": "Chi phí cố định cho tô màu"},
    {"name": "image_diff", "expression": "0.0", "description": "Chi phí dựa trên chênh lệch pixel"},
]

for cf in cost_functions:
    cfs.Costinsert(json.dumps(cf))

@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    if request.method == "POST":
        mode = request.form.get("mode")
        if not mode:
            result["error"] = "Vui lòng chọn chế độ (chuỗi hoặc hình ảnh)."
        elif mode == "string":
            start = request.form.get("start_str", "").strip()
            goal = request.form.get("goal_str", "").strip()
            if not start or not goal:
                result["error"] = "Vui lòng nhập cả chuỗi đầu vào và chuỗi mục tiêu."
            else:
                try:
                    converter = ObjectConvertor(tlm, cfs, max_steps=200)
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
                except Exception as e:
                    result["error"] = f"Lỗi xử lý chuỗi: {str(e)}"
        elif mode == "image":
            start_file = request.files.get("start_img")
            goal_file = request.files.get("goal_img")
            if not start_file or not goal_file:
                result["error"] = "Vui lòng tải lên cả hai hình ảnh."
            else:
                try:
                    start_filename = save_image(start_file, app.config['UPLOAD_FOLDER'])
                    goal_filename = save_image(goal_file, app.config['UPLOAD_FOLDER'])
                    start_path = os.path.join(app.config['UPLOAD_FOLDER'], start_filename)
                    goal_path = os.path.join(app.config['UPLOAD_FOLDER'], goal_filename)
                    start_img = open_image(start_path)
                    goal_img = open_image(goal_path)
                    converter = ObjectConvertor(tlm, cfs, max_steps=200)
                    path, cost, final_obj, intermediates = converter.convert(start_img, goal_img)
                    if not path:
                        result["error"] = f"Không tìm thấy đường dẫn biến đổi trong {converter.max_steps} bước."
                    else:
                        final_filename = f"final_{uuid.uuid4()}.png"
                        final_path = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
                        final_obj.save(final_path)
                        result = {
                            "path": [f"{step['name']} (params: {step['params']})" for step in path],
                            "cost": cost,
                            "type": "image",
                            "start_img": start_filename,
                            "goal_img": goal_filename,
                            "final_img": final_filename,
                            "intermediates": intermediates
                        }
                except ValueError as e:
                    result["error"] = str(e)
                except Exception as e:
                    result["error"] = f"Lỗi xử lý hình ảnh: {str(e)}"
    return render_template("index.html", result=result)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)