"""
    SUBQUERIES & CTEs

		WITH clause for organizing logic
        Nested subqueries for ranking, filtering, or KPI calculations
    
"""

-- @block [CTE + DATE_TRUNC()] Show which customers placed more than 1 order in the same week.

"""
    Prompt:
    Given a table Orders(order_id, customer_id, order_date, total_amount),

    Task:
    Return the customer IDs of those who placed more than one order in the same week.

"""

    -- CTE approach
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

-- @block [CTE + MULTIPLE GROUP BYs] Users visiting the same page more than once per day
"""
    Prompt:
    You have a table PageViews(user_id, page_id, view_time).

    Task:
    Return user IDs that viewed the same page at least twice on the same day.

"""

    -- CTE + MULTIPLE GROUP BYs Solution
    WITH visits_per_day AS (
        SELECT 
            DATE(view_time) AS dates,
            page_id,
            user_id,
            COUNT(user_id) AS visits
        FROM PageViews
        GROUP BY DATE(view_time), page_id, user_id
        ORDER BY DATE(view_time)
    )

    SELECT DISTINCT user_id FROM visits_per_day WHERE visits >= 2 --';'
;

-- @block [CTE + MULTIPLE GROUP BYs | ALT: SUBQUERY + WINDOW FUNC] Users visiting the same page more than once per day
"""
    Prompt:
    You have a table PageViews(user_id, page_id, view_time).

    Task:
    Return user IDs that viewed the same page at least twice on the same day.

"""

    -- CTE + MULTIPLE GROUP BYs Solution
    WITH visits_per_day AS (
        SELECT 
            DATE(view_time) AS dates,
            page_id,
            user_id,
            COUNT(user_id) AS visits
        FROM PageViews
        GROUP BY DATE(view_time), page_id, user_id
        ORDER BY DATE(view_time)
    )

    SELECT DISTINCT user_id FROM visits_per_day WHERE visits >= 2 --';'


    -- WINDOW FUNCTION + SUBQUERY
    SELECT DISTINCT user_id
    FROM (
        SELECT 
            user_id,
            page_id,
            DATE(view_time) AS view_date,
            COUNT(*) OVER (PARTITION BY user_id, page_id, DATE(view_time)) AS daily_page_views
        FROM PageViews
    ) sub
    WHERE daily_page_views >= 2 --';'
;

-- @block [CTE | FU: CTE + WF] Top Three Domains in Users Visits
"""
    Prompt:
        You're reviewing a table that logs user signups, including a column for the email domain (like gmail.com, yahoo.com, etc.).
        A stakeholder wants to know which three domains brought the most users last quarter — but they don't want totals, just the ranked domain names in order of activity.
    
    Approach:

        In a CTE:
            1. Start the SELECTION statement with parsing of the email column with everything after the '@' aliased as 'domain'.
            2. In the SELECTION statement add a coulumn with the COUNT(*) aggregation aliased as 'visits'.
            3. Optional: filter down dates to section only the last quarter, with a WHERE clause.
                (Let's suppose we are at today's date July 28th, 2025 -> This means last Quarter is Q2 [from april 1st to June 30th]and that there is 'visit_date' column in the table with each visit date).
            4. GROUP BY 'domain' column. 
            5. Order the resulting table with ORDER BY 'domain' column Descendingly.
            6. LIMIT the resulting table by 3 results (Which are the top 3).
        
        Now, from the CTE SELECT only the 'domain' column

"""

    -- Simple CTE Solution
    WITH ranked_domains AS( 
        SELECT split_part(email, '@', 2) AS domain, COUNT(*) AS visits
        FROM user_signups
        WHERE visit_date
            BETWEEN DATE '2025-04-01' AND DATE '2025-06-30'
        GROUP BY domain
        ORDER BY visits DESC
    )

    SELECT domain
    FROM ranked_domains
    LIMIT 3--';'


    -- Window Func +  CTE Solution
    WITH ranked_domains AS( 
        SELECT split_part(email, '@', 2) AS domain,
        COUNT(*) AS visits,
        RANK() OVER (ORDER BY COUNT(*) DESC) AS rnk
        FROM user_signups
        WHERE visit_date
            BETWEEN DATE '2025-04-01' AND DATE '2025-06-30'
        GROUP BY domain
    )

    SELECT domain
    FROM ranked_domains
    WHERE rnk <= 3
    ORDER BY rnk DESC --';'
;

-- @block [CTE + SUM()/COUNT() + HAVING] Categories, total revenue, from product with more than 5 distinct buyers
"""
    Prompt:
    A finance analyst wants a summary of total revenue per product category — but only for products that had at least 5 distinct buyers.
    Each sale record contains: product_id, category, buyer_id, and sale_amount.

    CTE approach:
        1. Create a CTE with
            * category,
            * product_id,
            * a SUM() aggregation on the 'sale_amount' AS 'product_revenue',
            * and a COUNT() aggregation on the DISTINCT buyers Aliased as 'distinct_buyers'.
        2. Group everthing by 'product_id' and 'category'.
        3. Filter down the CTE with a HAVING condition with 'COUNT(DISTINCT buyer_id) >= 5'
        4. Make the Outer query calling only on 'category' and a SUM() aggregation on product_revenue from the CTE AS 'total_revenue'.
            * GROUP BY category and ORDER BY total_revenue DESC.
            



"""

    -- CTE solution
    WITH summarized_sales AS(
        SELECT
            product_id,
            category,            
            SUM(sale_amount) AS product_revenue,
            COUNT(DISTINCT buyer_id) AS distinct_buyers
        FROM sales
        GROUP BY product_id, category
        HAVING COUNT(DISTINCT buyer_id) >= 5
    )
    SELECT category, SUM(product_revenue) AS total_revenue
    FROM summarized_sales
    GROUP BY category
    ORDER BY total_revenue DESC --';'

;

-- @block [CTEs + WF] Longest Consecutive Days Worked Streak
"""
    Prompt:
    For each employee in the work_logs table, calculate their longest streak of consecutive working days.
    The table includes employee_id and work_date.
    Return one row per employee with their employee_id and their longest streak.
    

    Solution Explanation + Step-by-Step Walkthrough

        Problem
        We need the longest streak of consecutive working days for each employee from the work_logs table.

        Key Concept
        Consecutive dates form a streak if for each day in the sequence:
     
            current_date = previous_date + 1 day

        - The trick is to turn each consecutive streak into a group key:

            work_date - (row_number * interval '1 day')

        - This difference remains constant across all days in the same streak.
        
        ---
        
        Step-by-Step
        
        Step 1: Add row numbers

            For each employee, sort their work dates and assign row numbers:
        
            WITH ordered_days AS (
                SELECT 
                    employee_id,
                    work_date,
                    ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY work_date) AS rn
                FROM work_logs
            )


        Step 2: Compute anchor key

            Subtract the row number (converted to days) from each date:
        
            , grouped AS (
                SELECT 
                    employee_id,
                    work_date,
                    work_date - rn * INTERVAL '1 day' AS grp_key
                FROM ordered_days
            )

        All dates in the same consecutive streak have the same grp_key.


        Step 3: Count streak lengths

        Count how many dates share the same group key:

            , streaks AS (
                SELECT 
                    employee_id,
                    grp_key,
                    COUNT(*) AS streak_length
                FROM grouped
                GROUP BY employee_id, grp_key
            )


        Step 4: Get the longest streak

        Finally, pick the longest streak per employee:

            SELECT 
                employee_id,
                MAX(streak_length) AS longest_streak
            FROM streaks
            GROUP BY employee_id
            ORDER BY employee_id --";"
        
        
        Why This Works

            - ROW_NUMBER() generates a positional index per employee.
            - work_date - rn * 1 day is constant for consecutive sequences.
            - Grouping by that anchor collapses each streak into one row.
            - Counting and taking MAX() gives the longest streak length.
        
"""

    -- CTE + WF Solution
    
    WITH ordered_days AS (
    SELECT 
        employee_id,
        work_date,
        ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY work_date) AS rn
    FROM work_logs
    ),
    grouped AS (
        SELECT 
            employee_id,
            work_date,
            work_date - rn * INTERVAL '1 day' AS grp_key
        FROM ordered_days
    ),
    streaks AS (
        SELECT 
            employee_id,
            grp_key,
            COUNT(*) AS streak_length
        FROM grouped
        GROUP BY employee_id, grp_key
    )
    SELECT 
        employee_id,
        MAX(streak_length) AS longest_streak
    FROM streaks
    GROUP BY employee_id
    ORDER BY employee_id --';'

    """
    Example of how this looks:
    
    -- 1) Add row numbers per employee sorted by date

        WITH ordered_days AS (
            SELECT 
                employee_id,
                work_date,
                ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY work_date) AS rn
            FROM work_logs
        ),

        -- Example Output:
        -- employee_id | work_date   | rn
        -- ------------+-------------+----
        -- 1           | 2025-07-01  | 1
        -- 1           | 2025-07-02  | 2
        -- 1           | 2025-07-03  | 3
        -- 1           | 2025-07-05  | 4
        -- 2           | 2025-07-10  | 1
        -- 2           | 2025-07-11  | 2
        -- 2           | 2025-07-15  | 3


    -- 2) Compute a "group key" to identify streaks

        grouped AS (
            SELECT 
                employee_id,
                work_date,
                work_date - rn * INTERVAL '1 day' AS grp_key
            FROM ordered_days
        ),

        -- Example Output:
        -- employee_id | work_date   | grp_key
        -- ------------+-------------+---------------------
        -- 1           | 2025-07-01  | 2025-06-30
        -- 1           | 2025-07-02  | 2025-06-30
        -- 1           | 2025-07-03  | 2025-06-30
        -- 1           | 2025-07-05  | 2025-07-01
        -- 2           | 2025-07-10  | 2025-07-09
        -- 2           | 2025-07-11  | 2025-07-09
        -- 2           | 2025-07-15  | 2025-07-12


    -- 3) Count how many days each streak has

        streaks AS (
            SELECT 
                employee_id,
                grp_key,
                COUNT(*) AS streak_length
            FROM grouped
            GROUP BY employee_id, grp_key
        ),

        -- Example Output:
        -- employee_id | grp_key     | streak_length
        -- ------------+-------------+--------------
        -- 1           | 2025-06-30  | 3
        -- 1           | 2025-07-01  | 1
        -- 2           | 2025-07-09  | 2
        -- 2           | 2025-07-12  | 1


    -- 4) Pick the longest streak per employee

        SELECT 
            employee_id,
            MAX(streak_length) AS longest_streak
        FROM streaks
        GROUP BY employee_id
        ORDER BY employee_id --";"

        -- Final Output:
        -- employee_id | longest_streak
        -- ------------+----------------
        -- 1           | 3
        -- 2           | 2

    """
;






