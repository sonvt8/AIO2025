=======
/* Khoi tao bang du lieu */
DROP TABLE IF EXISTS client;
DROP TABLE IF EXISTS invoice;

CREATE TABLE client (
    client_id INT PRIMARY KEY,
    name VARCHAR(50),
    address VARCHAR(100),
    city VARCHAR(50),
    state VARCHAR(50),
    phone VARCHAR(15)
);

CREATE TABLE invoice (
    invoice_id INT PRIMARY KEY,
    number VARCHAR(15),
    client_id INT,
    invoice_total DECIMAL(10, 2),
    payment_total DECIMAL(10, 2),
    invoice_date DATE,
    due_date DATE,
    payment_date DATE,
    FOREIGN KEY (client_id) REFERENCES client(client_id)
);

INSERT INTO client (client_id, name, address, city, state, phone) VALUES
(1, 'VinTe', '123 Nevada Parkway', 'Syracuse', 'NY', '315-252-7305'),
(2, 'Myworks', '3457 Glendale Parkway', 'Huntington', 'WV', '304-659-1170'),
(3, 'Yadel', '896 Pawling Parkway', 'San Francisco', 'CA', '415-744-6037'),
(4, 'Topiclounge', '083 Farmroad', 'Portland', 'OR', '971-888-9129'),
(5, 'Zencorp', '789 Oak Street', 'Seattle', 'WA', '206-555-0198'),
(6, 'Futurize', '456 Pine Road', 'Denver', 'CO', '303-444-7283');

INSERT INTO invoice (invoice_id, number, client_id, invoice_total, payment_total, invoice_date, due_date, payment_date) VALUES
(1, '91-953-3396', 1, 101.79, 0.00, '2019-03-09', '2019-03-29', NULL),
(2, '03-898-6735', 2, 175.32, 8.18, '2019-06-11', '2019-07-01', '2019-02-12'),
(3, '20-228-0335', 3, 147.99, 0.00, '2019-07-31', '2019-08-20', NULL),
(4, '56-934-0748', 3, 152.21, 0.00, '2019-03-08', '2019-03-28', NULL),
(5, '87-052-3121', 5, 169.36, 0.00, '2019-07-18', '2019-08-07', NULL),
(6, '75-587-6626', 6, 157.78, 74.55, '2019-01-29', '2019-02-18', '2019-01-03'),
(7, '68-093-9863', 1, 138.87, 0.00, '2019-04-04', '2019-04-24', NULL),
(8, '78-145-1093', 2, 189.12, 0.00, '2019-05-20', '2019-06-09', NULL),
(9, '48-266-1571', 3, 159.50, 0.00, '2019-06-30', '2019-07-20', NULL),
(10, '33-615-4694', 2, 126.38, 68.10, '2019-03-30', '2019-04-19', '2019-01-15'),
(11, '45-789-1234', 5, 145.67, 0.00, '2019-08-01', '2019-08-21', NULL),
(12, '22-345-6789', 6, 130.45, 50.20, '2019-02-15', '2019-03-07', '2019-02-28');

SELECT * from client;
SELECT * from invoice;

/* Truy van du lieu */
SELECT client_id, SUM(invoice_total)
FROM invoice
WHERE client.client_id = client_id;

/* Truy van khach hang chi tieu nhieu nhat */
SELECT c.client_id, c.name,
	   (SELECT SUM(invoice_total)
        FROM invoice
        WHERE c.client_id = client_id) as sum_ammount
FROM client c
ORDER by sum_ammount DESC
LIMIT 1;

/* Truy van khach hang chi tieu nhieu nhat - cach 2*/
SELECT c.client_id, c.name, SUM(invoice_total)
FROM client c JOIN invoice i on c.client_id = i.client_id
GROUP BY client_id
ORDER by invoice_total DESC
LIMIT 1;

/* Procedure */
CREATE PROCEDURE create_temp_invoice()
BEGIN
    DROP TABLE IF EXISTS temp_invoice;
    CREATE TEMPORARY TABLE temp_invoice (
        client_id INT,
        invoice_sum DECIMAL(10, 2),
        invoice_avg DECIMAL(10, 2)
    );
    
    INSERT INTO temp_invoice
    SELECT client_id, SUM(invoice_total) AS invoice_sum, AVG(invoice_total) AS invoice_avg
    FROM invoice
    GROUP BY client_id;
END;

CALL create_temp_invoice();
SELECT * FROM temp_invoice;

=======
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