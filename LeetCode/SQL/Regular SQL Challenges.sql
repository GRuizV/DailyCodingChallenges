'''
CHALLENGES INDEX

    
    176. Second Highest Salary (SUBQUERIES: DISCTINCT, ROW_NUM(), MAX())
    181. Employees Earning More Than Their Managers

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

-- @block // 181. Employees Earning More Than Their Managers

    """
    Challenge Statement

        Base

            Table: Employee
            +-------------+---------+
            | Column Name | Type    |
            +-------------+---------+
            | id          | int     |
            | name        | varchar |
            | salary      | int     |
            | managerId   | int     |
            +-------------+---------+
            id is the primary key (column with unique values) for this table.
            Each row of this table indicates the ID of an employee, their name, salary, and the ID of their manager.
        

        Statement

            Write a solution to find the employees who earn more than their managers.

            Return the result table in any order.


        Example

            Input: 

            Employee table:
            +----+-------+--------+-----------+
            | id | name  | salary | managerId |
            +----+-------+--------+-----------+
            | 1  | Joe   | 70000  | 3         |
            | 2  | Henry | 80000  | 4         |
            | 3  | Sam   | 60000  | Null      |
            | 4  | Max   | 90000  | Null      |
            +----+-------+--------+-----------+

            Output: 
            +----------+
            | Employee |
            +----------+
            | Joe      |
            +----------+

            Explanation: Joe is the only employee who earns more than his manager.
    """
    
    SELECT e1.name as Employee
    FROM Employee e1 JOIN Employee e2
        ON e1.managerId = e2.id
    WHERE e1.salary > e2.salary        

;






















