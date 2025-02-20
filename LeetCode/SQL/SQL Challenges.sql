'''
CHALLENGES INDEX

    SELECT
        1757. Recyclable and Low Fat Products
        584. Find Customer Referee
        595. Big Countries
        1148. Article Views I
        1683. Invalid Tweets





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
    SELECT Views.author_id AS id FROM Views
    WHERE Views.author_id = Views.viewer_id
    GROUP BY id ORDER BY id ASC
;
















































