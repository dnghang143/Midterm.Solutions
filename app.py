from flask import Flask, request, render_template, redirect, url_for
import os
from transformations import *

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

tlm = TransformationLibraryManager()
cfs = CostFunctionServer()

# Chuỗi
tlm.insert("add_a", lambda s: s + "a" if isinstance(s, str) else s)
tlm.insert("replace_a_with_b", lambda s: s.replace("a", "b", 1) if isinstance(s, str) else s)

# Ảnh
tlm.insert("rotate_90", lambda img: rotate_image(img) if isinstance(img, Image.Image) else img)
tlm.insert("invert", lambda img: invert_image(img) if isinstance(img, Image.Image) else img)

# Chi phí
cfs.insert("constant", lambda a, b: 1)
cfs.insert("image_diff", lambda a, b: image_cost(a, b))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    if request.method == "POST":
        mode = request.form.get('mode')
        if not mode:
            return render_template("index.html", result={"error": "Please select a mode."})

        if mode == 'string':
            start = request.form.get('start_str', '').strip()
            goal = request.form.get('goal_str', '').strip()
            if not start or not goal:
                return render_template("index.html", result={"error": "Please provide both start and goal strings."})
            converter = ObjectConverter(tlm, cfs, "constant", max_steps=50, heuristic=heuristic_string)
            path, cost, _ = converter.convert(start, goal)
            if not path:
                result = {"error": f"Không tìm thấy đường biến đổi trong giới hạn 50 bước."}
            else:
                result = {"path": path, "cost": cost, "type": "string"}

        elif mode == 'image':
            file1 = request.files.get('img1')
            file2 = request.files.get('img2')
            if not file1 or not allowed_file(file1.filename) or not file2 or not allowed_file(file2.filename):
                return render_template("index.html", result={"error": "Vui lòng chọn đúng file ảnh (png, jpg, jpeg)."})
            path1 = os.path.join(app.config['UPLOAD_FOLDER'], 'start.jpg')
            path2 = os.path.join(app.config['UPLOAD_FOLDER'], 'goal.jpg')
            file1.save(path1)
            file2.save(path2)

            img1 = resize_image(open_image(path1))
            img2 = resize_image(open_image(path2))

            converter = ObjectConverter(tlm, cfs, "image_diff", max_steps=50, heuristic=heuristic_image)
            path, cost, final_img = converter.convert(img1, img2)

            if not path:
                result = {"error": f"Không tìm thấy đường biến đổi trong giới hạn 50 bước."}
            else:
                # Lưu ảnh cuối (nếu muốn hiển thị)
                final_path = os.path.join(app.config['UPLOAD_FOLDER'], 'final.jpg')
                if final_img:
                    final_img.save(final_path)
                result = {
                    "path": path,
                    "cost": cost,
                    "type": "image",
                    "start_img": 'uploads/start.jpg',
                    "goal_img": 'uploads/goal.jpg',
                    "final_img": 'uploads/final.jpg' if final_img else None
                }

        else:
            result = {"error": "Chế độ không hợp lệ."}

    return render_template("index.html", result=result)

if __name__ == "__main__":
    print("Server is starting...")
    app.run(debug=True)
