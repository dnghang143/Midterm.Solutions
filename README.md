# Midterm.Solutions
Bài làm giữa kỳ môn hệ thống cơ sở dữ liệu đa phương tiện
# Thông tin Sinh viên thực hiện 
Trà Nhật Đông - N21DCCN111
Đặng Thị Thúy Hằng-N21DCCN118
# Đề bài: Final project in ex5
5. As a project, develop a software package that implements the transform-ation-based approach to retrieval by similarity. In particular, your package must contain the following capabilities that can be encoded as functions:

(a) Develop a syntax in which transformation operators can be represented. Then develop a program, called TransformationLibraryManager, that takes as input, perhaps through a user interface or from a file, a transformation operator specified in your syntax, and appends it to the library through a TLMinsert routine. Similarly, write a TLMsearch routine that, given the name of an instantiated operator, will return an appropriately instantiated version of the operator.

(b) Develop a syntax in which cost functions can be represented. Then write a program, called CostFunctionServer, that has a Costinsert routine that takes as input, perhaps through a user interface or from a file, a cost function specified in your syntax, and appends it to a library of cost functions. CostFunctionServer must also have a function, called EvaluateCall, that takes an instantiated transformation operator as input and returns the cost of this operator as output, using the cost functions represented using your syntax.

(c) Develop a program, called ObjectConvertor, that takes two objects o1 and o2 as input and that uses TransformationLibraryManager and CostFunctionServer to construct a least-cost transformation sequence between o1 and o2.

(d) Demonstrate your system's operation using the simple example of transformation sequences in Figure below. In particular, specify all the operations for this example in your syntax, as well as all the cost functions.
# Mô tả chi tiết bài làm:
# Hướng dẫn cài đặt:
Yêu cầu môi trường:
Python >= 3.10
pip (Python package manager)
Cài đặt các thư viện cần thiết: gồm các thư viện trong file requirements.txt
pip install -r requirements.txt
# Hướng dẫn chạy chương trình
Sau khi cài đặt xong
#1.chạy ứng dụng Flask
python app.py
#2. Sau đó mở trình duyệt và truy cập địa chỉ:
http://127.0.0.1:5000/
Bạn sẽ thấy giao diện đơn giản cho phép:

Nhập object_1 và object_2

Chọn kiểu dữ liệu (chuỗi / danh sách / ảnh)

Bấm “Run Transformation” để thực thi

Kết quả trả về chuỗi biến đổi + trạng thái trung gian
![image](https://github.com/user-attachments/assets/a5c151aa-1782-4dc5-99b2-73432451fed8)


# Input chạy demno
# Output khi hạy demo chương trình

