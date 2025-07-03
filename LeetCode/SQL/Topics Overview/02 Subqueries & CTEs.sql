"""
    SUBQUERIES & CTEs

		WITH clause for organizing logic
        Nested subqueries for ranking, filtering, or KPI calculations
    
"""

-- @block [CTE + DATE_TRUNC] Show which customers placed more than 1 order in the same week.

"""
    Prompt:
    Given a table Orders(order_id, customer_id, order_date, total_amount),

    Task:
    Return the customer IDs of those who placed more than one order in the same week.

"""

    -- approach
    WITH weekly_orders AS(
        SELECT 
            DATE_TRUNC('week', order_date) as week,
            customer_id,
            COUNT(order_id) as weekly_orders
        FROM Orders
        GROUP BY week, customer_id
    )

    SELECT week, customer_id, weekly_orders
    FROM weekly_orders
    WHERE weekly_orders > 1 --';'    

;







