"""
    TIME-BASE ANALYSIS

        Filtering by date ranges
        DATE_TRUNC() or EXTRACT() for month/week comparisons
        Calculating growth over time, retention by week, cohort analysis
    
"""
-- @block [DATE() + COUNT(DISTINCT)] Show the distinct users logged daily

"""
    Prompt:
    A table Logins contains records of user logins with columns: user_id, login_time.
    
    Task:
    Return the number of distinct users who logged in each day.

"""

    -- DATE() / GROUP BY approach

    SELECT 
        DATE(l.login_time) AS login_date,
        COUNT(DISTINCT l.user_id) as distinct_logged_users
    FROM Logins l
    GROUP BY login_date
    ORDER BY login_date --';'
;

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

-- @block [DATE_TRUNC + COUNT] Total view by week
"""
    Prompt:
    You're analyzing user engagement from a table Views(user_id, view_time).
    
    Task:
    Show the total number of views per week, ordered chronologically.

"""

    -- SELECT DISTINCT Solution
    SELECT 
        DATE_TRUNC('Week', view_time) AS week_start, 
        COUNT(view_time) as views_per_week
    FROM Views
    GROUP BY week_start
    ORDER BY week_start ASC --';'
;






