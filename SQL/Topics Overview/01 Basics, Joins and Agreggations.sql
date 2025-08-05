"""
BASICS

    SELECT, WHERE; ORDER BY, GROUP BY, ETC
"""
-- @block [SELECT + WHERE] Products with items fewer than 20
"""
    Prompt:
    You manage a warehouse inventory system. The table Products contains product_id, product_name, category, and stock_quantity.
    
    Task:
    Show the name and quantity of all products that have fewer than 20 units in stock.

"""

    -- SELECT/WHERE Solution
    SELECT DISTINCT p.product_id, p.stock_quantity
    FROM Products p
    WHERE p.stock_quantity < 20 --';'
;

-- @block [WHERE + WERE / DATE] Return users signed before Jan 1, 2023
"""
    Prompt:
    You manage an app with a table Users(user_id, signup_date).
    
    Task:
    Return all users who signed up before January 1, 2023.

"""

    -- SELECT + WHERE / DATE Solution
    SELECT user_id
    FROM Users
    WHERE signup_date < DATE '2023-01-01' --';'
;




"""
JOINS

    INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL OUTER JOIN
    Join logic based on user_id, product_id, date
"""
-- @block [LEFT JOIN] Users that never have placed an order

"""
    Prompt:
    You have a table users with all registered users and a table orders with customer orders. You need to find all users who have never placed an order.
    
    Tables:
        ○ users(id, name)
        ○ orders(id, user_id, order_date)
    
    Expected Result:
    List of users who do not appear in the orders table.

"""

    -- Join Solution
    SELECT u.id, u.name
    FROM users u LEFT JOIN orders o
        ON u.id = o.user_id
    WHERE o.id IS NULL -- ';'


    -- CTE Solution
    WITH users_with_orders AS (
        SELECT DISTINCT user_id
        FROM orders
    )

    SELECT u.* 
    FROM users u
    WHERE u.id NOT IN (SELECT * FROM users_with_orders) -- ';'

;

-- @block [JOIN (INNER JOIN)] Return orders placed with existing user
"""
    Prompt:
    Return all orders placed, including the user's name for each. Only include orders that are tied to an existing user.

    Tables:
        ○ users(id, name)
        ○ orders(id, user_id, order_date)

    Expected Result:
    Each order, along with the user's name.

"""

    -- Solution
    SELECT o.id AS order_id, o.user_id, u.name
    FROM orders o INNER JOIN user u
        ON o.user_id = u.id
;

-- @block [LEFT JOIN] Return all product with sells even if never sold
"""
    Prompt:
	You have a products table and a sales table. You want to see every product, including those that have never been sold.
	
	Tables:
		○ products(id, name)
		○ sales(id, product_id, quantity)
	
	Expected Result:
	All products listed, with sale quantities if any, and NULL for products with no sales.


"""

    -- Solution
    SELECT p.id AS product_id, p.name AS product_name, s.quantity AS quantity
    FROM products p LEFT JOIN sales s
        ON p.id = s.product_id
;

-- @block [LEFT JOIN] Sales Records of Deleted Products
"""
    Prompt:
	You want to find sales records that reference a product ID that no longer exists in the products table. These could be orphaned rows due to a deletion.
	
	Tables:
		- products(id, name)
        - sales(id, product_id, quantity)
	
	Expected Result:
    Only sales where the product_id does not match any existing product.
"""

    -- Solution
    SELECT s.product_id
    FROM sales s LEFT JOIN products p
        ON s.product_id = p.id
    WHERE p.id IS NULL
;

-- @block [INNER JOIN + LEFT JOIN] Return Orders Paid but not Shipped
"""
    Prompt:
	You have the following tables:
		• orders(id, customer_id, created_at)
		• payments(order_id, amount, paid_at)
		• shipments(order_id, shipped_at)
		
    Return all orders that have been paid but not yet shipped.
"""

    -- Solution
    SELECT o.*, p.paid_at
    FROM orders o
        INNER JOIN payments p ON o.id = p.order_id
        LEFT JOIN shipments s ON o.id = s.order_id
    WHERE s.order_id IS NULL
;

-- @block [DISTINCT] Select all categories present in the table

"""
    Prompt:
    A company tracks its customer support tickets in a table called SupportTickets. 
    Each row includes the ticket ID, the customer ID, the ticket status ("open", "closed", "pending"), 
    and the category ("billing", "technical", "general").

    Task:
    List all distinct ticket categories currently in use in the system.

"""

    -- SELECT / DISTINCT approach

    SELECT DISTINCT st.category
    FROM SupportTickets st --';'
;

-- @block [LEFT JOIN + COALESCE ] Employees with total compensation
"""
    Prompt:
    You have two tables: Employees(employee_id, name) and Salaries(employee_id, base_salary, bonus).
    
    Task:
    Produce a list of all employees along with their total compensation (base + bonus). Include employees even if they don't have salary data recorded yet.

"""

    -- LEFT JOIN / COALESCE Solution
    SELECT 
        e.name AS employee_name,
        COALESCE(s.base_salary, 0) + COALESCE(s.bonus, 0) AS total_compensation
    FROM Employees e LEFT JOIN Salaries s 
        ON e.employee_id = s.employee_id --';'
;

-- @block [LEFT JOIN + COALESCE] Show all customers and their most recent purchase with NULL handling
"""
    Prompt:
    Your company uses two tables: customers and purchases.
    Not every customer has made a purchase yet. Management wants a list of all customers and their most recent purchase date — or a note saying "No purchases yet" for those who haven't bought anything.
    
    Approach:
    1. SELECT the customer id, and an agreggation with MAX() on the date of the purchases.
        - The date of the customers will be NULL in those not having purchases yet, so a COALESCE() will handle that.
    2. LEFT JOIN customers and purchases.
    3. GROUP BY customers

"""

    -- LEFT JOIN + COALESCE Solution
    SELECT
        c.customer_id,
        COALESCE(
            TO_CHAR(MAX(p.purchase_date), 'YYYY-MM-DD'),
            'No purchases yet') AS last_purchase
    FROM customers c 
    LEFT JOIN purchases p 
        ON c.customer_id = p.customer_id
    GROUP BY c.customer_id--';'


    -- WINDOW FUNC Alternative
    SELECT customer_id,
       COALESCE(TO_CHAR(purchase_date, 'YYYY-MM-DD'), 'No purchases yet') AS last_purchase
    FROM (
        SELECT c.customer_id,
            p.purchase_date,
            ROW_NUMBER() OVER (PARTITION BY c.customer_id ORDER BY p.purchase_date DESC) AS rn
        FROM customers c
        LEFT JOIN purchases p ON c.customer_id = p.customer_id
    ) ranked
    WHERE rn = 1 OR rn IS NULL --';'
;






"""
AGGREGATIONS

    COUNT(), SUM(), AVG(), MAX(), MIN()
    Grouped aggreagatuins and conditional counts (COUNT(CASE WHEN))

"""
-- @block [SUM()/GROUP BY/ORDER BY] Top three selling products | ALT: SUBQUERY + WINDOW FUNC

"""
    Prompt:
    The table Sales has columns: order_id, product_id, and quantity.

    Task:
    Find the top 3 products (by product_id) that have been sold in the highest total quantity.

"""

    -- ORDER BY/LIMIT approach

    SELECT s.product_id, SUM(quantity) AS total_quantity
    FROM Sales s
    GROUP BY s.product_id
    ORDER BY total_quantity DESC
    LIMIT 3 --';'


    -- SUBQUERY/WINDOW FUNC approach

    SELECT s.product_id, total_quantity
    FROM (
        SELECT product_id, SUM(quantity) AS total_quantity,
            RANK() OVER (ORDER BY SUM(quantity) DESC) as RANK
        FROM Sales
        GROUP BY product_id
    ) ranked
    WHERE rnk <= 3 --';'

;

-- @block [MAX()] Last user visit

"""
    Prompt:
    You have a table WebTraffic(session_id, user_id, visit_time) representing web sessions.

    Task:
    For each user, return their most recent visit time.

"""

    -- COUNT/DATE_TRUNC approach

    SELECT wt.user_id, MAX(wt.visit_time) AS last_visit
    FROM WebTraffic wt
    GROUP BY wt.user_id --";"

;

-- @block [AVG() + COUNT()] Average prices per product category
"""
    Prompt:
	You have a products(id, name, category, price) table.
		
	Return the average price per category, but only for categories where more than 10 products exist.
    Sort the results with the most expensive categories first.
"""

    -- Solution
    SELECT p.category, AVG(p.price) AS avg_price
    GROUP BY p.category
    HAVING COUNT(p.id) > 10
    ORDER BY  avg_price DESC
;

-- @block [CASE] Product name, price, and price category
"""
    Prompt:
	You have a table products(id, name, price).
	
	Write a query that returns each product's name, price, and a column called price_category with the following logic:
		• If price is less than 20 → 'Low'
		• If price is between 20 and 100 → 'Medium'
    If price is over 100 → 'High'
"""

    -- Solution
    SELECT p.name, p.price, 
        CASE
            WHEN p.price < 20 THEN 'Low'
            WHEN p.price BETWEEN 20 and 100 THEN 'Medium'
            ELSE 'High'
        END AS price_category --';'
;

-- @block [COUNT(CASE)] High and Low Value Orders
"""
    Prompt:
	You have a table orders(id, customer_id, total, created_at).

	Write a query that returns two columns:
		• high_value_orders: count of orders with total > 100
		• low_value_orders: count of orders with total ≤ 100
		
	Each row in the result should summarize the total count for each category across the whole table (i.e., one row total).

"""

    -- CASE Solution
    SELECT 
        COUNT(CASE WHEN o.total > 100 THEN 1 ELSE NULL END) AS high_value_orders
        COUNT(CASE WHEN o.total <= 100 THEN 1 ELSE NULL END) AS low_value_orders
    FROM orders o --';'


    -- SUBQUERY Solution
    SELECT category, COUNT(*) AS category_count
        FROM(
            SELECT *,
                CASE
                    WHEN o.total > 100 THEN 'high_value_orders'
                    ELSE 'low_value_orders'
                END AS category
            FROM orders o
        ) AS categorized
       GROUP BY category --';'

       """
       The thing with the SUBQUERY Approach, more than being a bit let readable
       is that the solution will be given in rows instead of columns as the prompt requested:

       Category             category_count
       high_value_orders    45
       low_value_orders     20

       """
;

-- @block [SUM(CASE) | ATL: CTE + LEFT JOIN | OPT ALT: CTE + FULL OUTER JOIN + COALESCE] High and Low Value Orders
"""
    	Prompt:
        You have a table orders(id, customer_id, total, status).
        
        Each order can have one of several statuses: 'completed', 'pending', or 'cancelled'.
       
        Write a query that shows, for each customer, the total dollar value of:
            • Completed orders → column: total_completed
            • Cancelled orders → column: total_cancelled
            
        One row per customer.

"""

    -- CASE Solution
    SELECT o.id, o.customer_id,
        SUM(CASE(WHEN o.status = 'completed' THEN o.total ELSE 0 END)) AS total_completed
        SUM(CASE(WHEN o.status = 'cancelled' THEN o.total ELSE 0 END)) AS total_cancelled
    FROM orders o
    GROUP BY o.customer_id --';'


    -- CTE Solution
    WITH total_completed_table AS (
        SELECT o.customer_id, SUM(o.total) AS sum_total_completed
        FROM Orders o
        WHERE o.status = 'completed'
        GROUP BY o.customer_id
    ),

    WITH total_cancelled_table AS (
        SELECT o.customer_id, SUM(o.total) AS sum_total_cancelled
        FROM Orders o
        WHERE o.status = 'cancelled'
        GROUP BY o.customer_id
    ),

    SELECT 
        o.customer_id, 
        SUM(co.sum_total_completed) AS total_completed,
        SUM(ca.sum_total_cancelled) AS total_cancelled
    FROM
        orders o 
        LEFT JOIN total_completed_table co ON o.customer_id = co.customer_id
        LEFT JOIN total_cancelled_table ca ON o.customer_id = ca.customer_id
    GROUP BY o.customer_id


    -- Optimized CTE Solution
    	WITH completed_orders AS (
	    SELECT customer_id, SUM(total) AS total_completed
	    FROM orders
	    WHERE status = 'completed'
	    GROUP BY customer_id
	),
	cancelled_orders AS (
	    SELECT customer_id, SUM(total) AS total_cancelled
	    FROM orders
	    WHERE status = 'cancelled'
	    GROUP BY customer_id
	)
	
	SELECT
	    COALESCE(co.customer_id, ca.customer_id) AS customer_id,
	    COALESCE(co.total_completed, 0) AS total_completed,
	    COALESCE(ca.total_cancelled, 0) AS total_cancelled
	FROM completed_orders co
	FULL OUTER JOIN cancelled_orders ca ON co.customer_id = ca.customer_id
    
;

-- @block [SUM() + HAVING] Customers with purchases over $1.000
"""
    Prompt:
    A retail database has a table Purchases(customer_id, purchase_amount, purchase_date).
    
    Task:
    Which customers have spent more than $1000 in total across all purchases?

"""

    -- SUM / HAVING Solution
    SELECT customer_id, SUM(purchase_amount) AS total_spent
    FROM Purchases
    GROUP BY customer_id
    HAVING SUM(purchase_amount) > 1000 --';'
;

-- @block [COUNT() + MIN()] Drivers, total deliveries and their first
"""
    Prompt:
    You have a table Deliveries(driver_id, delivery_time).
    
    Task:
    For each driver, show the number of deliveries made and the time of their first delivery.

"""

    -- COUNT/MIN Solution
    SELECT 
        driver_id,
        COUNT(*) AS total_deliveries,
        MIN(delivery_time) AS first_delivery
    FROM Deliveries
    GROUP BY driver_id --';'
;







-- Template
-- @block [...] ...
"""
    Prompt:
    ...
    
    Task:
    ...

"""

    -- ... Solution
    SELECT 
    
    --';'
;



"""
CHALLENGES


"""











