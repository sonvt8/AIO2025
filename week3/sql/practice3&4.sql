DROP DATABASE IF EXISTS `invoicing`;
CREATE DATABASE `invoicing`; 
USE `invoicing`;

SET NAMES utf8 ;
SET character_set_client = utf8mb4 ;

CREATE TABLE `payment_methods` (
  `payment_method_id` tinyint(4) NOT NULL AUTO_INCREMENT,
  `name` varchar(50) NOT NULL,
  PRIMARY KEY (`payment_method_id`)
);
INSERT INTO payment_methods (payment_method_id, name) VALUES
(1,'Credit Card'),
(2,'Cash'),
(3,'PayPal'),
(4,'Wire Transfer')

CREATE TABLE `clients` (
  `client_id` int(11) NOT NULL,
  `name` varchar(50) NOT NULL,
  `address` varchar(50) NOT NULL,
  `city` varchar(50) NOT NULL,
  `state` char(2) NOT NULL,
  `phone` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`client_id`)
);
INSERT INTO clients (client_id, name, address, city, state, phone) VALUES
(1,'Vinte','3 Nevada Parkway','Syracuse','NY','315-252-7305'),
(2,'Myworks','34267 Glendale Parkway','Huntington','WV','304-659-1170'),
(3,'Yadel','096 Pawling Parkway','San Francisco','CA','415-144-6037'),
(4,'Kwideo','81674 Westerfield Circle','Waco','TX','254-750-0784'),
(5,'Topiclounge','0863 Farmco Road','Portland','OR','971-888-9129')

CREATE TABLE `invoices` (
  `invoice_id` int(11) NOT NULL,
  `number` varchar(50) NOT NULL,
  `client_id` int(11) NOT NULL,
  `invoice_total` decimal(9,2) NOT NULL,
  `payment_total` decimal(9,2) NOT NULL DEFAULT '0.00',
  `invoice_date` date NOT NULL,
  `due_date` date NOT NULL,
  `payment_date` date DEFAULT NULL,
  PRIMARY KEY (`invoice_id`),
  KEY `FK_client_id` (`client_id`),
  CONSTRAINT `FK_client_id` FOREIGN KEY (`client_id`) REFERENCES `clients` (`client_id`) ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO invoices (invoice_id, number, client_id, invoice_total, payment_total, invoice_date, due_date, payment_date) VALUES
(1,'91-953-3396',2,101.79,0.00,'2019-03-09','2019-03-29',NULL),
(2,'03-898-6735',5,175.32,8.18,'2019-06-11','2019-07-01','2019-02-12'),
(3,'20-228-0335',5,147.99,0.00,'2019-07-31','2019-08-20',NULL),
(4,'56-934-0748',3,152.21,0.00,'2019-03-08','2019-03-28',NULL),
(5,'87-052-3121',5,169.36,0.00,'2019-07-18','2019-08-07',NULL),
(6,'75-587-6626',1,157.78,74.55,'2019-01-29','2019-02-18','2019-01-03'),
(7,'68-093-9863',3,133.87,0.00,'2019-09-04','2019-09-24',NULL),
(8,'78-145-1093',1,189.12,0.00,'2019-05-20','2019-06-09',NULL),
(9,'77-593-0081',5,172.17,0.00,'2019-07-09','2019-07-29',NULL),
(10,'48-266-1517',1,159.50,0.00,'2019-06-30','2019-07-20',NULL),
(11,'20-848-0181',3,126.15,0.03,'2019-01-07','2019-01-27','2019-01-11'),
(13,'41-666-1035',5,135.01,87.44,'2019-06-25','2019-07-15','2019-01-26'),
(15,'55-105-9605',3,167.29,80.31,'2019-11-25','2019-12-15','2019-01-15'),
(16,'10-451-8824',1,162.02,0.00,'2019-03-30','2019-04-19',NULL),
(17,'33-615-4694',3,126.38,68.10,'2019-07-30','2019-08-19','2019-01-15'),
(18,'52-269-9803',5,180.17,42.77,'2019-05-23','2019-06-12','2019-01-08'),
(19,'83-559-4105',1,134.47,0.00,'2019-11-23','2019-12-13',NULL)


CREATE TABLE `payments` (
  `payment_id` int(11) NOT NULL AUTO_INCREMENT,
  `client_id` int(11) NOT NULL,
  `invoice_id` int(11) NOT NULL,
  `date` date NOT NULL,
  `amount` decimal(9,2) NOT NULL,
  `payment_method` tinyint(4) NOT NULL,
  PRIMARY KEY (`payment_id`),
  KEY `fk_client_id_idx` (`client_id`),
  KEY `fk_invoice_id_idx` (`invoice_id`),
  KEY `fk_payment_payment_method_idx` (`payment_method`),
  CONSTRAINT `fk_payment_client` FOREIGN KEY (`client_id`) REFERENCES `clients` (`client_id`) ON UPDATE CASCADE,
  CONSTRAINT `fk_payment_invoice` FOREIGN KEY (`invoice_id`) REFERENCES `invoices` (`invoice_id`) ON UPDATE CASCADE,
  CONSTRAINT `fk_payment_payment_method` FOREIGN KEY (`payment_method`) REFERENCES `payment_methods` (`payment_method_id`)
);
INSERT INTO payments (client_id, invoice_id, date, amount, payment_method) VALUES
(5,2,'2019-02-12',8.18,1),
(1,6,'2019-01-03',74.55,1),
(3,11,'2019-01-11',0.03,1),
(5,13,'2019-01-26',87.44,1),
(3,15,'2019-01-15',80.31,1),
(3,17,'2019-01-15',68.10,1),
(5,18,'2019-01-08',32.77,1),
(5,18,'2019-01-08',10.00,2)

SELECT * from payments;
SELECT * from invoices;
SELECT * from clients;
SELECT * from payment_methods;


/* Procedure practice 3: write a stored procedure
-- that inserts a new payment record, then updates the corresponding invoice with the payment total?
*/
DROP PROCEDURE if EXISTS update_payment_invoice;
CREATE PROCEDURE update_payment_invoice(
  IN `p_client_id` INT,
  IN `p_invoice_id` INT,
  IN `p_new_date` DATE,
  IN `p_new_amount` DECIMAL,
  IN `p_payment_method_id` INT
)
BEGIN
    INSERT INTO payments (client_id, invoice_id, date, amount, payment_method) VALUES
    (p_client_id, p_invoice_id, p_new_date, p_new_amount, p_payment_method_id);
    
    UPDATE invoices i
    SET i.payment_total = i.payment_total + p_new_amount
    WHERE i.invoice_id = p_invoice_id;
END

CALL update_payment_invoice(1,6,CURDATE(),2.35,2);

/* Procedure practice 4: create a stored procedure that finds the client who has
made the most payments in the last year
*/
DROP PROCEDURE if EXISTS find_most_payment_client;
CREATE PROCEDURE find_most_payment_client(
    IN p_date_comparision DATE
)
BEGIN
    SELECT c.name, COUNT(p.client_id) AS transaction_count
    FROM payments p
    JOIN clients c ON p.client_id = c.client_id
    WHERE p.date >= DATE_SUB(p_date_comparision, INTERVAL 1 MONTH)
    GROUP BY p.client_id, c.name
    ORDER BY transaction_count DESC
    LIMIT 1;
END