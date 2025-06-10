'''
CHALLENGES INDEX

    *FU: Follow-Up

    
    EASY (11)
    
    181. Employees Earning More Than Their Managers (SELF JOIN)
    182. Duplicate Emails (HAVING)
    183. Customers Who Never Order (LEFT JOIN)
    586. Customer Placing the Largest Number of Orders (ORDER BY: COUNT() / FU: SUBQUERY + MAX())
    511. Game Play Analysis I (SELECT + MIN())
    596. Classes More Than 5 Students (HAVING + COUNT())
    607. Sales Person (LEFT JOIN + SUBQUERY [LEF JOINT + LIKE])
    610. Triangle Judgement (CASE)
    619. Biggest Single Number (MAX() + SUBQUERY)
    620. Not Boring Movies (CONDITIONALS + ORDER BY)
    196. Delete Duplicate Emails (DML [Deletion]: SUBQUERIES & FU: CTE)


    MEDIUM (2)

    176. Second Highest Salary  (SUBQUERIES: DISCTINCT, ROW_NUM(), MAX())
    177. Nth Higest Salary  (PL/SQL: Procedural Language SQL)



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
    HAVING COUNT(student) >= 5
;

-- @block // 607. Sales Person

    """
    Challenge Statement

        Base

            Table: SalesPerson
            +-----------------+---------+
            | Column Name     | Type    |
            +-----------------+---------+
            | sales_id        | int     |
            | name            | varchar |
            | salary          | int     |
            | commission_rate | int     |
            | hire_date       | date    |
            +-----------------+---------+
            sales_id is the primary key (column with unique values) for this table.
            Each row of this table indicates the name and the ID of a salesperson alongside their salary, commission rate, and hire date.
            

            Table: Company
            +-------------+---------+
            | Column Name | Type    |
            +-------------+---------+
            | com_id      | int     |
            | name        | varchar |
            | city        | varchar |
            +-------------+---------+
            com_id is the primary key (column with unique values) for this table.
            Each row of this table indicates the name and the ID of a company and the city in which the company is located.
            

            Table: Orders
            +-------------+------+
            | Column Name | Type |
            +-------------+------+
            | order_id    | int  |
            | order_date  | date |
            | com_id      | int  |
            | sales_id    | int  |
            | amount      | int  |
            +-------------+------+
            order_id is the primary key (column with unique values) for this table.
            com_id is a foreign key (reference column) to com_id from the Company table.
            sales_id is a foreign key (reference column) to sales_id from the SalesPerson table.
            Each row of this table contains information about one order. This includes the ID of the company, the ID of the salesperson, the date of the order, and the amount paid.

                   

        Statement

            Write a solution to find the names of all the salespersons who did not have any orders related to the company with the name "RED".

            Return the result table in any order.

            The result format is in the following example.


        Example

            SalesPerson table:
            +----------+------+--------+-----------------+------------+
            | sales_id | name | salary | commission_rate | hire_date  |
            +----------+------+--------+-----------------+------------+
            | 1        | John | 100000 | 6               | 4/1/2006   |
            | 2        | Amy  | 12000  | 5               | 5/1/2010   |
            | 3        | Mark | 65000  | 12              | 12/25/2008 |
            | 4        | Pam  | 25000  | 25              | 1/1/2005   |
            | 5        | Alex | 5000   | 10              | 2/3/2007   |
            +----------+------+--------+-----------------+------------+

            Company table:
            +--------+--------+----------+
            | com_id | name   | city     |
            +--------+--------+----------+
            | 1      | RED    | Boston   |
            | 2      | ORANGE | New York |
            | 3      | YELLOW | Boston   |
            | 4      | GREEN  | Austin   |
            +--------+--------+----------+

            Orders table:
            +----------+------------+--------+----------+--------+
            | order_id | order_date | com_id | sales_id | amount |
            +----------+------------+--------+----------+--------+
            | 1        | 1/1/2014   | 3      | 4        | 10000  |
            | 2        | 2/1/2014   | 4      | 5        | 5000   |
            | 3        | 3/1/2014   | 1      | 1        | 50000  |
            | 4        | 4/1/2014   | 1      | 4        | 25000  |
            +----------+------------+--------+----------+--------+

            Output: 
            +------+
            | name |
            +------+
            | Amy  |
            | Mark |
            | Alex |
            +------+

            Explanation: 
            According to orders 3 and 4 in the Orders table, it is easy to tell that only salesperson John and Pam have sales to company RED, so we report all the other names in the table salesperson.


    """
    
    -- My Approach
    SELECT DISTINCT s.name AS name
    FROM SalesPerson salary
    LEFT JOIN Orders o
        ON s.sales_id = o.sales_id
    WHERE s.sales_id NOT IN (

        SELECT o.sales_id
        FROM Orders o
        LEFT JOIN Company C
            ON o.com_id = c.com_id
        WHERE c.name LIKE 'RED'
    )
;

-- @block // 610. Triangle Judgement

    """
    Challenge Statement

        Base

            Table: Triangle
            +-------------+------+
            | Column Name | Type |
            +-------------+------+
            | x           | int  |
            | y           | int  |
            | z           | int  |
            +-------------+------+
            In SQL, (x, y, z) is the primary key column for this table.
            Each row of this table contains the lengths of three line segments.

                   

        Statement

            Report for every three line segments whether they can form a triangle.

            Return the result table in any order.

            The result format is in the following example.


        Example

            Triangle table:
            +----+----+----+
            | x  | y  | z  |
            +----+----+----+
            | 13 | 15 | 30 |
            | 10 | 20 | 15 |
            +----+----+----+

            Output: 
            +----+----+----+----------+
            | x  | y  | z  | triangle |
            +----+----+----+----------+
            | 13 | 15 | 30 | No       |
            | 10 | 20 | 15 | Yes      |
            +----+----+----+----------+


    """
    
    -- My Approach
    SELECT x, y, z, 
        CASE 
            WHEN (x + y > z AND y + z > x AND x + z > y)
            THEN 'Yes' ELSE 'No'
        END AS triangle
    FROM Triangle
;

-- @block // 619. Biggest Single Number

    """
    Challenge Statement

        Base

            Table: MyNumbers

            +-------------+------+
            | Column Name | Type |
            +-------------+------+
            | num         | int  |
            +-------------+------+
            This table may contain duplicates (In other words, there is no primary key for this table in SQL).
            Each row of this table contains an integer.

                   

        Statement

            A single number is a number that appeared only once in the MyNumbers table.

            Find the largest single number. If there is no single number, report null.

            The result format is in the following example.


        Example

            MyNumbers table:
            +-----+
            | num |
            +-----+
            | 8   |
            | 8   |
            | 3   |
            | 3   |
            | 1   |
            | 4   |
            | 5   |
            | 6   |
            +-----+

            Output: 
            +-----+
            | num |
            +-----+
            | 6   |
            +-----+
            Explanation: The single numbers are 1, 4, 5, and 6.
            Since 6 is the largest single number, we return it.


            Example 2:

            MyNumbers table:
            +-----+
            | num |
            +-----+
            | 8   |
            | 8   |
            | 7   |
            | 7   |
            | 3   |
            | 3   |
            | 3   |
            +-----+

            Output: 
            +------+
            | num  |
            +------+
            | null |
            +------+

            Explanation: There are no single numbers in the input table so we return null.


    """
    
    -- My Approach
    SELECT MAX(num) AS num FROM (
        SELECT num FROM MyNumbers
        GROUP BY num
        HAVING COUNT(num) = 1
        ) AS mynumber
;

-- @block // 619. Biggest Single Number

    """
    Challenge Statement

        Base

            Table: Cinema

            +----------------+----------+
            | Column Name    | Type     |
            +----------------+----------+
            | id             | int      |
            | movie          | varchar  |
            | description    | varchar  |
            | rating         | float    |
            +----------------+----------+
            id is the primary key (column with unique values) for this table.
            Each row contains information about the name of a movie, its genre, and its rating.
            rating is a 2 decimal places float in the range [0, 10]

                   

        Statement

            Write a solution to report the movies with an odd-numbered ID and a description that is not "boring".

            Return the result table ordered by rating in descending order.

            The result format is in the following example.


        Example

           Cinema table:
            +----+------------+-------------+--------+
            | id | movie      | description | rating |
            +----+------------+-------------+--------+
            | 1  | War        | great 3D    | 8.9    |
            | 2  | Science    | fiction     | 8.5    |
            | 3  | irish      | boring      | 6.2    |
            | 4  | Ice song   | Fantacy     | 8.6    |
            | 5  | House card | Interesting | 9.1    |
            +----+------------+-------------+--------+

            Output: 
            +----+------------+-------------+--------+
            | id | movie      | description | rating |
            +----+------------+-------------+--------+
            | 5  | House card | Interesting | 9.1    |
            | 1  | War        | great 3D    | 8.9    |
            +----+------------+-------------+--------+

            Explanation: 
            We have three movies with odd-numbered IDs: 1, 3, and 5. The movie with ID = 3 is boring so we do not include it in the answer.


    """
    
    -- My Approach
    SELECT *
    FROM Cinema 
    WHERE id % 2 =1
        AND description NOT LIKE '%boring%'
    ORDER BY rating DESC
;

-- @block // 196. Delete Duplicate Emails

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

            Write a solution to delete all duplicate emails, keeping only one unique email with the smallest id.

            For SQL users, please note that you are supposed to write a DELETE statement and not a SELECT one.

            For Pandas users, please note that you are supposed to modify Person in place.

            After running your script, the answer shown is the Person table. The driver will first compile and run your piece of code and then show the Person table. The final order of the Person table does not matter.

            The result format is in the following example.


        Example

            Person table:
            +----+------------------+
            | id | email            |
            +----+------------------+
            | 1  | john@example.com |
            | 2  | bob@example.com  |
            | 3  | john@example.com |
            +----+------------------+

            Output: 
            +----+------------------+
            | id | email            |
            +----+------------------+
            | 1  | john@example.com |
            | 2  | bob@example.com  |
            +----+------------------+

            Explanation: john@example.com is repeated two times. We keep the row with the smallest Id = 1.


    """
    
    -- Direct Apporach
    DELETE FROM Person
    WHERE id NOT IN (
        SELECT MIN(id) FROM Person
        GROUP BY email
    )


    -- Real life approach: CTE
    WITH to_keep AS (
        SELECT MIN(id) FROM Person
        GROUP BY email
    )

    DELETE FROM Person
    WHERE id NOT IN( 
        SELECT * FROM to_keep
    )
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

-- @block // 177. Nth Higest Salary

    """
    Table: Employee

    +-------------+------+
    | Column Name | Type |
    +-------------+------+
    | id          | int  |
    | salary      | int  |
    +-------------+------+
    id is the primary key (column with unique values) for this table.
    Each row of this table contains information about the salary of an employee.
    

    Problem Statement
    Write a solution to find the nth highest distinct salary from the Employee table. If there are less than n distinct salaries, return null.

    The result format is in the following example.

    

    Example 1:

    Input: 
    Employee table:
    +----+--------+
    | id | salary |
    +----+--------+
    | 1  | 100    |
    | 2  | 200    |
    | 3  | 300    |
    +----+--------+
    n = 2
    Output: 
    +------------------------+
    | getNthHighestSalary(2) |
    +------------------------+
    | 200                    |
    +------------------------+
    Example 2:

    Input: 
    Employee table:
    +----+--------+
    | id | salary |
    +----+--------+
    | 1  | 100    |
    +----+--------+
    n = 2
    Output: 
    +------------------------+
    | getNthHighestSalary(2) |
    +------------------------+
    | null                   |
    +------------------------+

    """

    CREATE OR REPLACE FUNCTION NthHighestSalary(N INT)
    RETURNS TABLE (Salary INT) AS $$

    BEGIN
        RETURN QUERY
        SELECT DISTINCT e.salary
        FROM Employee e
        ORDER BY e.salary DESC
        OFFSET N - 1 LIMIT 1;
    
    END; 
    $$ LANGUAGE plpgsql;

-- @block // xx.















