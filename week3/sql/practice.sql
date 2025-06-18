-- Tạo bảng office
CREATE TABLE office (
    office_id INT PRIMARY KEY,
    office_location VARCHAR(100)
);

-- Tạo bảng employee
CREATE TABLE employee (
    employee_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    job_title VARCHAR(100),
    salary DECIMAL(10, 2),
    reports_to INT,
    office_id INT,
    FOREIGN KEY (reports_to) REFERENCES employee(employee_id),
    FOREIGN KEY (office_id) REFERENCES office(office_id)
);

-- Chèn 10 bản ghi vào bảng office
INSERT INTO office (office_id, office_location) VALUES
(1, 'Hanoi'),
(2, 'Ho Chi Minh City'),
(3, 'Da Nang'),
(4, 'Can Tho'),
(5, 'Hai Phong'),
(6, 'Nha Trang'),
(7, 'Dalat'),
(8, 'Vung Tau'),
(9, 'Hue'),
(10, 'Quy Nhon');

-- Chèn 10 bản ghi vào bảng employee
INSERT INTO employee (employee_id, first_name, last_name, job_title, salary, reports_to, office_id) VALUES
(1, 'Nguyen', 'Van A', 'Director', 80000.00, NULL, 1), -- Manager cấp cao nhất, không báo cáo ai
(2, 'Tran', 'Thi B', 'Manager', 75000.00, 1, 2),
(3, 'Le', 'Van C', 'Developer', 50000.00, 2, 3),
(4, 'Pham', 'Thi D', 'Analyst', 45000.00, 2, 4),
(5, 'Hoang', 'Van E', 'Designer', 48000.00, 2, 5),
(6, 'Do', 'Thi F', 'Engineer', 52000.00, 1, 6),
(7, 'Vu', 'Van G', 'HR Specialist', 40000.00, 1, 7),
(8, 'Dang', 'Thi H', 'Marketing', 47000.00, 2, 8),
(9, 'Bui', 'Van I', 'Support', 39000.00, 1, 9),
(10, 'Ngo', 'Thi J', 'Sales', 43000.00, 2, 10);

-- Nâng lương cho một số nhân viên cao hơn manager
UPDATE employee e1
SET salary = (
    SELECT e2.salary * 1.2 -- lấy lương của manager tăng thêm 20%
    FROM employee e2
    WHERE e2.employee_id = e1.reports_to
) + 5000 -- lấy khoản này gán cho danh sách nhân viên bên dưới
WHERE e1.reports_to IS NOT NULL
AND e1.employee_id IN (3, 5, 7);

-- Practice #1: Lấy ra danh sách các nhân viên có lương cao hơn manager của họ
SELECT * 
FROM employee e1
WHERE salary > (
  SELECT salary
  from employee e2
  WHERE e2.employee_id =  e1.reports_to
);

-- Practice #2: Lấy ra danh sách họ tên nhân viên và địa chỉ office của họ
drop TABLE if EXISTS temp_location;
CREATE TEMPORARY TABLE temp_location AS (
  SELECT first_name, last_name, o.office_location
  FROM employee e
  JOIN office o on e.office_id = o.office_id
);
SELECT * FROM temp_location;