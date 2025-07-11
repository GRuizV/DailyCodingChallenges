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

-- @block [CTE + INNER JOIN | ALT: SUBQUERY] Users that have been both assigned and have opened a ticket
"""
    Prompt:
    A table SupportTickets(ticket_id, opened_by, assigned_to, status, open_date) tracks support requests.
    
    Task:
    Return the names of users who have both opened and been assigned at least one ticket.

"""

    -- CTE / INNER JOIN Solution
    WITH users_with_tickets_opened AS (
        SELECT DISTINCT opened_by FROM SupportTickets
    ),

    users_with_tickets_assigned AS (
        SELECT DISTINCT assigned_to FROM SupportTickets
    )

    SELECT DISTINCT opened_by AS user_id
    FROM users_with_tickets_opened uto 
    INNER JOIN users_with_tickets_assigned uta 
        ON uto.opened_by = uta.assigned_to 
    --';'


    -- SUBQUERY Solution
    SELECT DISTINCT opened_by AS user_id
    FROM SupportTickets
    WHERE opened_by IN(
        SELECT assigned_to FROM SupportTickets
    ) --';'

    
;






