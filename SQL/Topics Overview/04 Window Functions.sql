"""
    WINDOW FUNCTIONS
        ROW_NUMBER(), RANK(), DENSE_RANK()
        SUM() OVER(), AVG() OVER(), partitioned and ordered
    
"""   
  
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

-- @block [CTE + WF] Weekly % Change in users traffic
"""
    Prompt:
    You're asked to analyze weekly traffic trends.
    You have a page_views table with user_id, view_date, and page_url.
    The team wants to know the percentage change in weekly pageviews per user — from one week to the next — only for users who had traffic in consecutive weeks.

"""

    -- CTE + WF Solution
    WITH weekly_counts AS (
    SELECT 
        DATE_TRUNC('week', view_date)::date AS week_start,
        user_id,
        page_url,
        COUNT(*) AS visits
    FROM page_views
    GROUP BY DATE_TRUNC('week', view_date), user_id, page_url
    ),

    week_changes AS (
        SELECT 
            week_start,
            user_id,
            page_url,
            visits,
            LAG(visits) OVER (PARTITION BY user_id, page_url ORDER BY week_start) AS prev_visits
        FROM weekly_counts
    )
    SELECT 
        week_start,
        user_id,
        page_url,
        ROUND( ( ((visits - prev_visits) / NULLIF(prev_visits, 0))* 100.0), 2) AS pct_change
    FROM week_changes
    WHERE prev_visits IS NOT NULL
    ORDER BY user_id, page_url, week_start--';'
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









