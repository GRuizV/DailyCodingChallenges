'''
CHALLENGES INDEX

    *FU: Follow-Up

    
    EASY
    
    181. Employees Earning More Than Their Managers (SELF JOIN)
    182. Duplicate Emails (HAVING)
    183. Customers Who Never Order (LEFT JOIN)
    586. Customer Placing the Largest Number of Orders (ORDER BY: COUNT() / FU: SUBQUERY + MAX())
    511. Game Play Analysis I (SELECT + MIN())
    596. Classes More Than 5 Students (HAVING + COUNT())
    


    MEDIUM

    176. Second Highest Salary  (SUBQUERIES: DISCTINCT, ROW_NUM(), MAX())



'''





-- CHALLENGES

"EASY"

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

-- @block // 586. Customer Placing the Largest Number of Orders

    """
    Challenge Statement

        Base

            Table: Customers
            +-----------------+----------+
            | Column Name     | Type     |
            +-----------------+----------+
            | order_number    | int      |
            | customer_number | int      |
            +-----------------+----------+
            order_number is the primary key (column with unique values) for this table.
            This table contains information about the order ID and the customer ID.             

                   

        Statement

            Write a solution to find the customer_number for the customer who has placed the largest number of orders.

            The test cases are generated so that exactly one customer will have placed more orders than any other customer.

            The result format is in the following example.


        Example

            Orders table:
            +--------------+-----------------+
            | order_number | customer_number |
            +--------------+-----------------+
            | 1            | 1               |
            | 2            | 2               |
            | 3            | 3               |
            | 4            | 3               |
            +--------------+-----------------+

            Output: 
            +-----------------+
            | customer_number |
            +-----------------+
            | 3               |
            +-----------------+

            Explanation: 
            The customer with number 3 has two orders, which is greater than either customer 1 or 2 because each of them only has one order. 
            So the result is customer_number 3.
        

        Follow Up Question:
            What if more than one customer has the largest number of orders, can you find all the customer_number in this case?


    """
    
    -- My Approach
    SELECT customer_number 
    FROM Orders
    GROUP BY customer_number
    ORDER BY COUNT(customer_number) DESC
    LIMIT 1


    -- Follow Up solution: What if ties?
    SELECT customer_number
    FROM Orders
    GROUP BY customer_number
    HAVING COUNT(customer_number) = (
        SELECT MAX(order_count) 
        FROM (
            SELECT customer_number, COUNT(*) AS order_count 
            FROM Orders 
            GROUP BY customer_number
            ) AS counts
    )
;

-- @block // 511. Game Play Analysis I

    """
    Challenge Statement

        Base

            Table: Activity
            +--------------+---------+
            | Column Name  | Type    |
            +--------------+---------+
            | player_id    | int     |
            | device_id    | int     |
            | event_date   | date    |
            | games_played | int     |
            +--------------+---------+

            (player_id, event_date) is the primary key (combination of columns with unique values) of this table.
            This table shows the activity of players of some games.
            Each row is a record of a player who logged in and played a number of games (possibly 0) before logging out on someday using some device.            

                   

        Statement

            Write a solution to find the first login date for each player.

            Return the result table in any order.

            The result format is in the following example.


        Example

            Activity table:
            +-----------+-----------+------------+--------------+
            | player_id | device_id | event_date | games_played |
            +-----------+-----------+------------+--------------+
            | 1         | 2         | 2016-03-01 | 5            |
            | 1         | 2         | 2016-05-02 | 6            |
            | 2         | 3         | 2017-06-25 | 1            |
            | 3         | 1         | 2016-03-02 | 0            |
            | 3         | 4         | 2018-07-03 | 5            |
            +-----------+-----------+------------+--------------+

            Output: 
            +-----------+-------------+
            | player_id | first_login |
            +-----------+-------------+
            | 1         | 2016-03-01  |
            | 2         | 2017-06-25  |
            | 3         | 2016-03-02  |
            +-----------+-------------+


    """
    
    -- My Approach
    SELECT player_id, MIN(event_date) AS first_login
    FROM Activity
    GROUP BY player_id
    ORDER BY player_id ASC

;

-- @block // 596. Classes More Than 5 Students

    """
    Challenge Statement

        Base

            Table:  Courses
            +-------------+---------+
            | Column Name | Type    |
            +-------------+---------+
            | student     | varchar |
            | class       | varchar |
            +-------------+---------+
            (student, class) is the primary key (combination of columns with unique values) for this table.
            Each row of this table indicates the name of a student and the class in which they are enrolled.

                   

        Statement

            Write a solution to find all the classes that have at least five students.

            Return the result table in any order.

            The result format is in the following example.


        Example

            Courses table:
            +---------+----------+
            | student | class    |
            +---------+----------+
            | A       | Math     |
            | B       | English  |
            | C       | Math     |
            | D       | Biology  |
            | E       | Math     |
            | F       | Computer |
            | G       | Math     |
            | H       | Math     |
            | I       | Math     |
            +---------+----------+
            Output: 
            +---------+
            | class   |
            +---------+
            | Math    |
            +---------+


    """
    
    -- My Approach
    SELECT class
    FROM Courses
    GROUP BY class
    HAVING COUNT(student) >= 5;

;







"MEDIUM"

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

















