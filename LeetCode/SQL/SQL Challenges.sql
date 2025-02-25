'''
CHALLENGES INDEX

    SELECT
        1757. Recyclable and Low Fat Products
        584. Find Customer Referee
        595. Big Countries
        1148. Article Views I
        1683. Invalid Tweets [The length of the tweets]
    .   


    JOIN
        1581. Customer Who Visited but Did Not Make Any Transactions (LEFT JOIN)
        1068. Product Sales Analysis I (INNER JOIN)
        1378. Replace Employee ID With The Unique Identifier (LEFT JOIN)




'''





-- SELECT

-- @block // 1757. Recyclable and Low Fat Products

    """
    Challenge Statement

        Base
            Table: Products
            +-------------+---------+
            | Column Name | Type    |
            +-------------+---------+
            | product_id  | int     |
            | low_fats    | enum    |
            | recyclable  | enum    |
            +-------------+---------+
            product_id is the primary key (column with unique values) for this table.
            low_fats is an ENUM (category) of type ('Y', 'N') where 'Y' means this product is low fat and 'N' means it is not.
            recyclable is an ENUM (category) of types ('Y', 'N') where 'Y' means this product is recyclable and 'N' means it is not.
        

        Statement

            Write a solution to find the ids of products that are both low fat and recyclable.

            Return the result table in any order.

            The result format is in the following example.


        Example

            Input: 
            Products table:
            +-------------+----------+------------+
            | product_id  | low_fats | recyclable |
            +-------------+----------+------------+
            | 0           | Y        | N          |
            | 1           | Y        | Y          |
            | 2           | N        | Y          |
            | 3           | Y        | Y          |
            | 4           | N        | N          |
            +-------------+----------+------------+

            Output: 
            +-------------+
            | product_id  |
            +-------------+
            | 1           |
            | 3           |
            +-------------+

            Explanation: Only products 1 and 3 are both low fat and recyclable.

    """

    -- Solution
    SELECT product_id FROM Products WHERE low_fats = 'Y' AND recyclable = 'Y'
;

-- @block // 584. Find Customer Referee

    """
    Challenge Statement

        Base

            Table: Customer
            +-------------+---------+
            | Column Name | Type    |
            +-------------+---------+
            | id          | int     |
            | name        | varchar |
            | referee_id  | int     |
            +-------------+---------+
            In SQL, id is the primary key column for this table.
            Each row of this table indicates the id of a customer, their name, and the id of the customer who referred them.
        

        Statement

            Find the names of the customer that are not referred by the customer with id = 2.

            Return the result table in any order.

            The result format is in the following example.


        Example

            Input: 
            Customer table:
            +----+------+------------+
            | id | name | referee_id |
            +----+------+------------+
            | 1  | Will | null       |
            | 2  | Jane | null       |
            | 3  | Alex | 2          |
            | 4  | Bill | null       |
            | 5  | Zack | 1          |
            | 6  | Mark | 2          |
            +----+------+------------+
            Output: 
            +------+
            | name |
            +------+
            | Will |
            | Jane |
            | Bill |
            | Zack |
            +------+

    """

    -- Solution
    SELECT c.name FROM Customer AS c
    WHERE c.referee_id != 2 OR c.referee_id IS NULL
;

-- @block // 595. Big Countries

    """
    Challenge Statement

        Base

            Table: World
                +-------------+---------+
                | Column Name | Type    |
                +-------------+---------+
                | name        | varchar |
                | continent   | varchar |
                | area        | int     |
                | population  | int     |
                | gdp         | bigint  |
                +-------------+---------+
                name is the primary key (column with unique values) for this table.
                Each row of this table gives information about the name of a country, the continent to which it belongs, its area, the population, and its GDP value.
        

        Statement

            A country is big if:
                - it has an area of at least three million (i.e., 3000000 km2), or
                - it has a population of at least twenty-five million (i.e., 25000000).
            
            Write a solution to find the name, population, and area of the big countries.

            Return the result table in any order.

            The result format is in the following example.


        Example

            Input: 
            World table:
            +-------------+-----------+---------+------------+--------------+
            | name        | continent | area    | population | gdp          |
            +-------------+-----------+---------+------------+--------------+
            | Afghanistan | Asia      | 652230  | 25500100   | 20343000000  |
            | Albania     | Europe    | 28748   | 2831741    | 12960000000  |
            | Algeria     | Africa    | 2381741 | 37100000   | 188681000000 |
            | Andorra     | Europe    | 468     | 78115      | 3712000000   |
            | Angola      | Africa    | 1246700 | 20609294   | 100990000000 |
            +-------------+-----------+---------+------------+--------------+
            Output: 
            +-------------+------------+---------+
            | name        | population | area    |
            +-------------+------------+---------+
            | Afghanistan | 25500100   | 652230  |
            | Algeria     | 37100000   | 2381741 |
            +-------------+------------+---------+

    """

    -- Solution
    SELECT w.name, w.population, w.area FROM World as w
    WHERE w.area >= 3000000 OR w.population >= 25000000
;

-- @block // 1148. Article Views I
        
    """
    Challenge Statement

        Base

            Table: Views
                +---------------+---------+
                | Column Name   | Type    |
                +---------------+---------+
                | article_id    | int     |
                | author_id     | int     |
                | viewer_id     | int     |
                | view_date     | date    |
                +---------------+---------+
                There is no primary key (column with unique values) for this table, the table may have duplicate rows.
                Each row of this table indicates that some viewer viewed an article (written by some author) on some date. 
                Note that equal author_id and viewer_id indicate the same person.
        

        Statement

            Write a solution to find all the authors that viewed at least one of their own articles.

            Return the result table sorted by id in ascending order.

            The result format is in the following example.


        Example

            Input: 
            Views table:
            +------------+-----------+-----------+------------+
            | article_id | author_id | viewer_id | view_date  |
            +------------+-----------+-----------+------------+
            | 1          | 3         | 5         | 2019-08-01 |
            | 1          | 3         | 6         | 2019-08-02 |
            | 2          | 7         | 7         | 2019-08-01 |
            | 2          | 7         | 6         | 2019-08-02 |
            | 4          | 7         | 1         | 2019-07-22 |
            | 3          | 4         | 4         | 2019-07-21 |
            | 3          | 4         | 4         | 2019-07-21 |
            +------------+-----------+-----------+------------+
            Output: 
            +------+
            | id   |
            +------+
            | 4    |
            | 7    |
            +------+

    """

    -- Solution
    SELECT Views.author_id AS id FROM Views
    WHERE Views.author_id = Views.viewer_id
    GROUP BY id ORDER BY id ASC
;

-- @block // 1683. Invalid Tweets

    """
    Challenge Statement

        Base

            Table: Tweets
            +----------------+---------+
            | Column Name    | Type    |
            +----------------+---------+
            | tweet_id       | int     |
            | content        | varchar |
            +----------------+---------+
            tweet_id is the primary key (column with unique values) for this table.
            content consists of characters on an American Keyboard, and no other special characters.
            This table contains all the tweets in a social media app.
        

        Statement

            Write a solution to find the IDs of the invalid tweets. The tweet is invalid if the number of characters used in the content of the tweet is strictly greater than 15.

            Return the result table in any order.

            The result format is in the following example.


        Example

            Input: 
            Tweets table:
            +----------+-----------------------------------+
            | tweet_id | content                           |
            +----------+-----------------------------------+
            | 1        | Let us Code                       |
            | 2        | More than fifteen chars are here! |
            +----------+-----------------------------------+
            Output: 
            +----------+
            | tweet_id |
            +----------+
            | 2        |
            +----------+
            Explanation: 
            Tweet 1 has length = 11. It is a valid tweet.
            Tweet 2 has length = 33. It is an invalid tweet.

    """

    -- Solution
    SELECT tweet_id
    FROM Tweets
    WHERE LENGTH(content) > 15
;





-- JOIN

-- @block // 1581. Customer Who Visited but Did Not Make Any Transactions

    """
    Challenge Statement

        Base

            Table:Visits

                +-------------+---------+
                | Column Name | Type    |
                +-------------+---------+
                | visit_id    | int     |
                | customer_id | int     |
                +-------------+---------+
                visit_id is the column with unique values for this table.
                This table contains information about the customers who visited the mall.
        

            Table: Transactions

                +----------------+---------+
                | Column Name    | Type    |
                +----------------+---------+
                | transaction_id | int     |
                | visit_id       | int     |
                | amount         | int     |
                +----------------+---------+
                transaction_id is column with unique values for this table.
                This table contains information about the transactions made during the visit_id.


        Statement

            Write a solution to find the IDs of the users who visited without making any transactions and the number of times they made these types of visits.

            Return the result table sorted in any order.

            The result format is in the following example.


        Example

            Input: 
            Visits
            +----------+-------------+
            | visit_id | customer_id |
            +----------+-------------+
            | 1        | 23          |
            | 2        | 9           |
            | 4        | 30          |
            | 5        | 54          |
            | 6        | 96          |
            | 7        | 54          |
            | 8        | 54          |
            +----------+-------------+

            Transactions
            +----------------+----------+--------+
            | transaction_id | visit_id | amount |
            +----------------+----------+--------+
            | 2              | 5        | 310    |
            | 3              | 5        | 300    |
            | 9              | 5        | 200    |
            | 12             | 1        | 910    |
            | 13             | 2        | 970    |
            +----------------+----------+--------+
           
            Output: 
            +-------------+----------------+
            | customer_id | count_no_trans |
            +-------------+----------------+
            | 54          | 2              |
            | 30          | 1              |
            | 96          | 1              |
            +-------------+----------------+

            Explanation: 
            Customer with id = 23 visited the mall once and made one transaction during the visit with id = 12.
            Customer with id = 9 visited the mall once and made one transaction during the visit with id = 13.
            Customer with id = 30 visited the mall once and did not make any transactions.
            Customer with id = 54 visited the mall three times. During 2 visits they did not make any transactions, and during one visit they made 3 transactions.
            Customer with id = 96 visited the mall once and did not make any transactions.
            As we can see, users with IDs 30 and 96 visited the mall one time without making any transactions. Also, user 54 visited the mall twice and did not make any transactions.

    """

    -- Solution
    SELECT v.customer_id AS customer_id, 
        count(v.customer_id) AS count_no_trans
    FROM Visits v
    LEFT JOIN Transactions t ON v.visit_id = t.visit_id
    WHERE t.visit_id IS NULL
    GROUP BY v.customer_id
    ORDER BY count_no_trans
;

-- @block // 1068. Product Sales Analysis I

    """
    Challenge Statement

        Base

            Table: Sales

            +-------------+-------+
            | Column Name | Type  |
            +-------------+-------+
            | sale_id     | int   |
            | product_id  | int   |
            | year        | int   |
            | quantity    | int   |
            | price       | int   |
            +-------------+-------+
            (sale_id, year) is the primary key (combination of columns with unique values) of this table.
            product_id is a foreign key (reference column) to Product table.
            Each row of this table shows a sale on the product product_id in a certain year.
            Note that the price is per unit.
            

            Table: Product

            +--------------+---------+
            | Column Name  | Type    |
            +--------------+---------+
            | product_id   | int     |
            | product_name | varchar |
            +--------------+---------+
            product_id is the primary key (column with unique values) of this table.
            Each row of this table indicates the product name of each product.
        .


        Statement

            Write a solution to report the product_name, year, and price for each sale_id in the Sales table.

            Return the resulting table in any order.

            The result format is in the following example.
        .

        Example

            Input: 
            Sales table:
            +---------+------------+------+----------+-------+
            | sale_id | product_id | year | quantity | price |
            +---------+------------+------+----------+-------+ 
            | 1       | 100        | 2008 | 10       | 5000  |
            | 2       | 100        | 2009 | 12       | 5000  |
            | 7       | 200        | 2011 | 15       | 9000  |
            +---------+------------+------+----------+-------+

            Product table:
            +------------+--------------+
            | product_id | product_name |
            +------------+--------------+
            | 100        | Nokia        |
            | 200        | Apple        |
            | 300        | Samsung      |
            +------------+--------------+


            Output: 
            +--------------+-------+-------+
            | product_name | year  | price |
            +--------------+-------+-------+
            | Nokia        | 2008  | 5000  |
            | Nokia        | 2009  | 5000  |
            | Apple        | 2011  | 9000  |
            +--------------+-------+-------+

            Explanation: 
            From sale_id = 1, we can conclude that Nokia was sold for 5000 in the year 2008.
            From sale_id = 2, we can conclude that Nokia was sold for 5000 in the year 2009.
            From sale_id = 7, we can conclude that Apple was sold for 9000 in the year 2011.

    """

    -- Solution
    SELECT p.product_name, s.year, s.price
    FROM Product p lEFT JOIN Sales s ON p.product_id = s.product_id
    WHERE s.year IS NOT NULL
    ORDER BY s.year

    -- Alternatively and More Efficient (INNER JOIN)
    SELECT p.product_name, s.year, s.price
    FROM Product p INNER JOIN Sales s ON p.product_id = s.product_id
    ORDER BY s.year

;

-- @block // 1378. Replace Employee ID With The Unique Identifier

    """
    Challenge Statement

        Base

            Table: Employees

            +---------------+---------+
            | Column Name   | Type    |
            +---------------+---------+
            | id            | int     |
            | name          | varchar |
            +---------------+---------+
            id is the primary key (column with unique values) for this table.
            Each row of this table contains the id and the name of an employee in a company.
            

            Table: EmployeeUNI

            +---------------+---------+
            | Column Name   | Type    |
            +---------------+---------+
            | id            | int     |
            | unique_id     | int     |
            +---------------+---------+
            (id, unique_id) is the primary key (combination of columns with unique values) for this table.
            Each row of this table contains the id and the corresponding unique id of an employee in the company.
        .


        Statement

            Write a solution to show the unique ID of each user, If a user does not have a unique ID replace just show null.

            Return the result table in any order.

            The result format is in the following example.
        .

        Example

            Input: 
            Employees table:
            +----+----------+
            | id | name     |
            +----+----------+
            | 1  | Alice    |
            | 7  | Bob      |
            | 11 | Meir     |
            | 90 | Winston  |
            | 3  | Jonathan |
            +----+----------+
            EmployeeUNI table:
            +----+-----------+
            | id | unique_id |
            +----+-----------+
            | 3  | 1         |
            | 11 | 2         |
            | 90 | 3         |
            +----+-----------+
            Output: 
            +-----------+----------+
            | unique_id | name     |
            +-----------+----------+
            | null      | Alice    |
            | null      | Bob      |
            | 2         | Meir     |
            | 3         | Winston  |
            | 1         | Jonathan |
            +-----------+----------+
            Explanation: 
            Alice and Bob do not have a unique ID, We will show null instead.
            The unique ID of Meir is 2.
            The unique ID of Winston is 3.
            The unique ID of Jonathan is 1.

    """

    -- Solution
    SELECT eu.unique_id, e.name
    FROM Employees e LEFT JOIN EmployeeUNI eu ON e.id = eu.id
    ORDER BY e.name


;













































