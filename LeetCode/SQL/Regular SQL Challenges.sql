'''
CHALLENGES INDEX

    
    176. Second Highest Salary (SUBQUERIES: DISCTINCT, ROW_NUM(), MAX())


'''





-- SELECT

-- @block // 176. Second Highest Salary 

    """
    Challenge Statement

        Base

            Table: Employee
            +-------------+------+
            | Column Name | Type |
            +-------------+------+
            | id          | int  |
            | salary      | int  |
            +-------------+------+
            id is the primary key (column with unique values) for this table.
            Each row of this table contains information about the salary of an employee.
        

        Statement

            Write a solution to find the second highest distinct salary from the Employee table.
            If there is no second highest salary, return null


        Example

            Input: 

            Employee table:
            +----+--------+
            | id | salary |
            +----+--------+
            | 1  | 100    |
            | 2  | 200    |
            | 3  | 300    |
            +----+--------+

            Output: 
            +---------------------+
            | SecondHighestSalary |
            +---------------------+
            | 200                 |
            +---------------------+


            Example 2:

            Input: 

            Employee table:
            +----+--------+
            | id | salary |
            +----+--------+
            | 1  | 100    |
            +----+--------+

            Output: 
            +---------------------+
            | SecondHighestSalary |
            +---------------------+
            | null                |
            +---------------------+

    """

    -- There could be 3 solutions with their pros and cons, All using SUBQUERIES

    "First: 'DISTINCT-LIMIT-OFFSET' APPROACH"
    SELECT (
        SELECT DISTINCT salary FROM Employee
        ORDER BY salary DESC LIMIT 1 OFFSET 1
    ) AS SecondHighestSalary

    /* 
    Notes
    
        This is the most intuitive approach.
    
        This Subquery is necessary, because without it, the Query wouldn't return NULL if there is
        no second highest salary 

        In term of performance it has the 3rd place.
    */


    "Second: 'ROW_NUMBER()' APPROACH"
    SELECT (

        SELECT salary FROM(
            SELECT salary, ROW_NUMBER() OVER(ORDER BY salary DESC) AS rnk
            FROM Employee
            GROUP BY salary
        ) ranked_salaries 
        WHERE rnk = 2

    ) AS SecondHighestSalary

    /* 
    Notes
        
        This is a bit more complex approach since it has nested subqueries (3).

        In term of performance it has the 1st place.
    */
    
    

    "Third: 'MAX() - SUBQUERY' APPROACH"
    SELECT max(salary) FROM Employee AS SecondHighestSalary
    WHERE salary < (SELECT max(salary) FROM Employee)

    /* 
    Notes
        
        This is also quite intuitive but it doesn't work pretty well in large data sets.

        In term of performance it has the 2nd place.
    */
;

