'''
CHALLENGES INDEX

    SELECT
        1757. Recyclable and Low Fat Products
        584. Find Customer Referee
        595. Big Countries





'''







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



















































