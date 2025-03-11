'''
CHALLENGES INDEX

    
    176. Second Highest Salary  (SUBQUERIES: DISCTINCT, ROW_NUM(), MAX())
    181. Employees Earning More Than Their Managers (SELF JOIN)
    182. Duplicate Emails (HAVING)
    183. Customers Who Never Order (LEFT JOIN)
    
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

-- @block // 182. Duplicate Emails

    """
    Challenge Statement

        Base

            Table: Person
            +-------------+---------+
            | Column Name | Type    |
            +-------------+---------+
            | id          | int     |
            | email       | varchar |
            +-------------+---------+
            id is the primary key (column with unique values) for this table.
            Each row of this table contains an email. The emails will not contain uppercase letters.
        

        Statement

            Write a solution to report all the duplicate emails.
            Note that it's guaranteed that the email field is not NULL.

            Return the result table in any order.


        Example

            Person table:
            +----+---------+
            | id | email   |
            +----+---------+
            | 1  | a@b.com |
            | 2  | c@d.com |
            | 3  | a@b.com |
            +----+---------+

            Output: 
            +---------+
            | Email   |
            +---------+
            | a@b.com |
            +---------+

            Explanation: a@b.com is repeated two times.
    """
    
    -- My First Approach
    SELECT DISTINCT p1.email
    FROM Person p1 JOIN Person p2
        ON p1.email = p2.emails
    WHERE p1.id != p2.id

    """
    Notes: My approach actualy worked but it is inefficient. with GROUP and HAVING
        It'll be more industry standard
    """

    -- Optmial Solution
    SELECT email
    FROM Person
    GROUP BY email
    HAVING count(email) > 1
;

-- @block // 183. Customers Who Never Order

    """
    Challenge Statement

        Base

            Table: Customers
            +-------------+---------+
            | Column Name | Type    |
            +-------------+---------+
            | id          | int     |
            | name        | varchar |
            +-------------+---------+

            id is the primary key (column with unique values) for this table.
            Each row of this table indicates the ID and name of a customer.

            Table: Orders
            +-------------+------+
            | Column Name | Type |
            +-------------+------+
            | id          | int  |
            | customerId  | int  |
            +-------------+------+

            id is the primary key (column with unique values) for this table.
            customerId is a foreign key (reference columns) of the ID from the Customers table.
            Each row of this table indicates the ID of an order and the ID of the customer who ordered it.
        

        Statement

            Write a solution to find all customers who never order anything.

            Return the result table in any order.


        Example

            Customers table:
            +----+-------+
            | id | name  |
            +----+-------+
            | 1  | Joe   |
            | 2  | Henry |
            | 3  | Sam   |
            | 4  | Max   |
            +----+-------+

            Orders table:
            +----+------------+
            | id | customerId |
            +----+------------+
            | 1  | 3          |
            | 2  | 1          |
            +----+------------+

            Output: 
            +-----------+
            | Customers |
            +-----------+
            | Henry     |
            | Max       |
            +-----------+
    """
    
    -- My Approach
    SELECT c.name AS Customers
    FROM Customers c LEFT JOIN Orders o
        ON c.id = o.customerId
    WHERE o.id IS NULL

;





















