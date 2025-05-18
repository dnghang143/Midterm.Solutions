from typing import Any, Callable, Dict, List, Tuple
import hashlib
import heapq
from PIL import Image, ImageOps
import numpy as np

class TransformationOperator:
    def __init__(self, name: str, function: Callable[[Any], Any]):
        self.name = name
        self.function = function

    def apply(self, obj: Any) -> Any:
        return self.function(obj)

class CostFunction:
    def __init__(self, name: str, function: Callable[[Any, Any], float]):
        self.name = name
        self.function = function

    def evaluate(self, before: Any, after: Any) -> float:
        return self.function(before, after)

class TransformationLibraryManager:
    def __init__(self):
        self.operators: Dict[str, TransformationOperator] = {}

    def insert(self, name: str, func: Callable[[Any], Any]):
        self.operators[name] = TransformationOperator(name, func)

    def search(self, name: str) -> TransformationOperator:
        return self.operators.get(name)

    def get_all(self) -> List[TransformationOperator]:
        return list(self.operators.values())

class CostFunctionServer:
    def __init__(self):
        self.cost_functions: Dict[str, CostFunction] = {}

    def insert(self, name: str, func: Callable[[Any, Any], float]):
        self.cost_functions[name] = CostFunction(name, func)

    def evaluate(self, name: str, before: Any, after: Any) -> float:
        return self.cost_functions[name].evaluate(before, after)

    def cost_insert(self, name: str, func: Callable[[Any, Any], float]):
        self.insert(name, func)

    def evaluate_call(self, name: str, before: Any, after: Any) -> float:
        return self.evaluate(name, before, after)

class ObjectConverter:
    def __init__(self, tlm: TransformationLibraryManager, cfs: CostFunctionServer, cost_function_name: str, max_steps=50, heuristic=None):
        self.tlm = tlm
        self.cfs = cfs
        self.cost_function_name = cost_function_name
        self.max_steps = max_steps
        self.heuristic = heuristic

    def convert(self, start: Any, goal: Any) -> Tuple[List[str], float, Any]:
        queue = [(0, 0, start, [], start)]
        visited = set()

        def obj_hash(obj: Any) -> str:
            if isinstance(obj, Image.Image):
                return hashlib.md5(np.array(obj).tobytes()).hexdigest()
            return hashlib.md5(str(obj).encode()).hexdigest()

        goal_hash = obj_hash(goal)

        while queue:
            f_score, cost, current, path, current_obj = heapq.heappop(queue)
            h = obj_hash(current)
            if h in visited:
                continue
            visited.add(h)

            if h == goal_hash:
                return path, cost, current_obj

            if len(path) >= self.max_steps:
                continue

            for op in self.tlm.get_all():
                try:
                    next_obj = op.apply(current)
                    g_new = cost + self.cfs.evaluate(self.cost_function_name, current, next_obj)
                    h_new = self.heuristic(next_obj, goal) if self.heuristic else 0
                    f_new = g_new + h_new
                    heapq.heappush(queue, (f_new, g_new, next_obj, path + [op.name], next_obj))
                except Exception:
                    continue

        return [], float('inf'), None

# === Image utility ===
def open_image(path):
    return Image.open(path).convert("RGB")

def resize_image(img, size=(100, 100)):
    return img.resize(size)

def rotate_image(img):
    return img.rotate(90)

def invert_image(img):
    return ImageOps.invert(img)

def scale_x(img):
    arr = np.array(img)
    new_arr = np.repeat(arr, 2, axis=1)
    return Image.fromarray(new_arr.astype(np.uint8))

def scale_y(img):
    arr = np.array(img)
    new_arr = np.repeat(arr, 2, axis=0)
    return Image.fromarray(new_arr.astype(np.uint8))

def paint(img, color=(255, 0, 255)):
    arr = np.array(img)
    arr[arr[:, :, 0] > 100] = color
    return Image.fromarray(arr.astype(np.uint8))

def image_cost(before: Image.Image, after: Image.Image) -> float:
    a1 = np.array(before).astype(np.int32)
    a2 = np.array(after).astype(np.int32)
    return float(np.sum(np.abs(a1 - a2)))