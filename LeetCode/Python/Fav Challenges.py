'''
CHALLENGES INDEX

ARRAYS
[D] 11. Container With Most Water (Array)
->  31. Next Permutation (Array) (TP)
    42. Trapping Rain Water (Array)
    46. Permutations (Array)
    45. Jump Game II (Array) (GRE) (DP)
    55. Jump Game (Array) (DP) (GRE)
    88. Merge Sorted Array (Array) (TP) (Sorting)
    118. Pascal's Triangle (Array) (DP)
    121. Best Time to Buy and Sell Stock (Array) (DP)
    122. Best Time to Buy and Sell Stock II (Array) (DP) (GRE)
    128. Longest Consecutive Sequence (Array) (Hash Table)
    134. Gas Station (Array) (GRE)
    153. Find Minimum in Rotated Sorted Array (Array) (BS)
    204. Count Primes (Array) (Others)
    215. Kth Largest Element in an Array (Array) (Heap) (DQ) (Sorting)
    239. Sliding Window Maximum (Array) (SW)
    283. Move Zeroes (Array) (TP)
    287. Find the Duplicate Number (FCD) (Array) (TP)
    560. Subarray Sum Equals K (Array) (PS)


HASH TABLES
[D] 12. Integer to Roman (Hash Table) (GRE)
[D] 13. Roman to Integer (Hash Table)
    17. Letter Combinations of a Phone Number (Hash Table) (BT)
    76. Minimum Window Substring (Hash Table) (SW)
    127. Word Ladder (Hast Table) (BFS)
    142. Linked List Cycle II (Hash Table) (LL) (TP) (FCD)
    202. Happy Number (Hash Table) (TP) (Others)
    208. Implement Trie (Hast Table) (Tree)
    567. Permutation in String (Hash Table) (SW)


MATRICES
    36. Valid Sudoku (Array) (Hash Table) (Matrix)
    48. Rotate Image (Matrix)
    130. Surrounded Regions (Matrix) (BFS) (DFS)
    200. Number of Islands (Matrix) (DFS)
    240. Search a 2D Matrix II (Matrix) (DQ) (BS)
    289. Game of Life (Matrix)
    994. Rotting Oranges (Matrix) (BFS) *DIFFERENTIAL COORDINATES


DFS & BFS
    98. Validate Binary Search Tree (Tree) (DFS)
    101. Symmetric Tree (Tree) (BFS) (DFS)
    102. Binary Tree Level Order Traversal (Tree) (BFS) (DFS)
    104. Maximum Depth of Binary Tree (Tree) (BFS) (DFS)
    114. Flatten Binary Tree to Linked List (LL) (DFS) (Tree)
    116. Populating Next Right Pointers in Each Node (BFS) (DFS) (Tree)
    199. Binary Tree Right Side View (Tree) (DFS) (RC)
    207. Course Schedule (DFS) (Topological Sort)
    226. Invert Binary Tree (Tree) (DFS)
    329. Longest Increasing Path in a Matrix (Matrix) (DFS) (MEM) (RC)
    341. Flatten Nested List Iterator (DFS)


DYNAMIC PROGRAMMING
    5. Longest Palindromic Substring (DP) (TP)
    22. Generate Parentheses (DP) (BT)
    32. Longest Valid Parentheses (Stack) (DP)
    62. Unique Paths (DP)
    70. Climbing Stairs (DP)
    72. Edit Distance (DP) 'Levenshtein Distance'
    91. Decode Ways (DP)
    139. Word Break (DP)
    198. House Robber (Array) (DP)
    279. Perfect Squares (DP)
    300. Longest Increasing Subsequence (Array) (DP)
    322. Coin Change (DP)
    392. Is Subsequence (TP) (DP)
    1143. Longest Common Subsequence (DP)


HEAPS, STACKS & QUEUES
    20. Valid Parentheses (Stack)
    150. Evaluate Reverse Polish Notation (Stack)
    155. Min Stack (Stack)
    227. Basic Calculator II (Stack)
    295. Find Median from Data Stream (Heap) (Sorting)
    347. Top K Frequent Elements (Array) (Heaps) (Sorting)
    378. Kth Smallest Element in a Sorted Matrix (Matrix) (Heaps)
    32. Longest Valid Parentheses (Stack) (DP)
    394. Decode String (RC) (Stack)
    739. Daily Temperatures (Array) (Stack) [Monotonic Stack]


LINKED-LISTS
    2. Add Two Numbers (LL) (RC)
    19. Remove Nth Node From End of List (LL) (TP)
    21. Merge Two Sorted Lists (LL) (RC)
    138. Copy List with Random Pointer (Hash Table) (LL)
    141. Linked List Cycle (TP) (LL)
    148. Sort List (TP) (LL)
    160. Intersection of Two Linked Lists (TP) (LL)
    206. Reverse Linked List (LL) (RC)
    234. Palindrome Linked List (LL) (RC) (TP)
    237. Delete Node in a Linked List (LL)
    328. Odd Even Linked List (LL)


OTHERS
    8. String to Integer (atoi) (Others)
    14. Longest Common Prefix (Others)
    38. Count and Say (Others)
    171. Excel Sheet Column Number (Others)
    172. Factorial Trailing Zeroes (Others)
    XX. Minimal Balls Move (SW) [AgileEngine]




*LL: Linked-Lists
*BS: Binary Search
*DP: Dynamic Programming
*RC: Recursion
*TP: Two-pointers
*FCD: Floyd's cycle detection (Hare & Tortoise approach)
*PS: Preffix-sum
*SW: Sliding-Window
*MEM: Memoization
*GRE: Greedy
*DQ: Divide and Conquer
*BT: Backtracking
*BFS & DFS: Breadth-First Search & Depth-First Search
*Arrays, Hash Tables & Matrices
*Sorting
*Heaps, Stacks & Queues
*Graphs & Trees
*Others


(87)
'''




'ARRAYS'

'11. Container With Most Water'
# def x():

#     from typing import Optional

#     # Input
#     # Case 1
#     height = [1,8,6,2,5,4,8,3,7]
#     # Output: 49

#     # Case 2
#     height = [1,1]
#     # Output: 1

#     # Case 3
#     height = [4,3,2,1,4]
#     # Output: 16

#     # Case 4
#     height = [1,2,1]
#     # Output: 2

#     # Case 5
#     height = [1,2,4,3]
#     # Output: 4

#     '''
#     My Approach (Two Pointers)

#         Intuition:
            
#             - Define two pointers 'left' and 'right' initialized at 0 and len(height)-1 respectively.
#             - Initialize a 'max_amount' holder at 0 to hold the results.
#             - In a While Loop (while left < right):
#                 + Update the max_amount with the max between its current value and the current area [min(left, right)*(right-left)]
#                 + Update left/right pointers with the condition left+=1 if intervals[left] < intervals[right] else right -= 1
#             - Return the max_amount
#     '''

#     def maxArea(height: list[int]) -> int:

#         # Define the two pointers
#         left, right = 0, len(height)-1

#         # Initialize the result holder
#         max_amount = 0

#         # Process the input
#         while left < right:

#             current_water = min(height[left],height[right])*(right-left)
#             max_amount = max(max_amount, current_water)

#             if height[left] < height[right]:
#                 left +=1 
#             else:
#                 right-=1
               
#         # Return the result
#         return max_amount

#     # Testing
#     print(maxArea(height=height))

#     '''Note: This solution beated submissions by 77% in Runtime and 95% in Memory'''








'HASH TABLES'

'''12. Integer to Roman'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     num = 3749
#     # Output: MMMDCCXLIX

#     # Case 2
#     num = 58
#     # Output: LVIII

#     # Case 3
#     num = 1994
#     # Output: MCMXCIV
    
#     # # Custom Case
#     # num = 0
#     # # Output: -

#     '''
#     My Approach

#         Intuition:
            
#             - Parse the input by digit in reverse.
#             - Assign a roman number according to if its units, tens, hundred or thousands
#             - Revert the order of the result built elements
#             - Join and return
#     '''

#     def intToRoman(num: int) -> str:

#         # Parse the input to separate by units, tens, hundreds and thousands but reversed
#         places = [int(x) for x in str(num)[::-1]]

#         # Initialize a Result Holder
#         res: list = []

#         # Process the input
#         for i in range(len(places)):

#             if i == 3:
#                 res.append('M'*places[i])
            
#             else:                                
#                 if places[i] != 0:

#                     num = places[i]*(10**i)
#                     first_dig = places[i]

#                     if first_dig in range(1,4):                    
#                         if i == 0:
#                             res.append('I'*first_dig)

#                         elif i == 1:
#                             res.append('X'*first_dig)
                        
#                         elif i == 2:
#                             res.append('C'*first_dig)


#                     elif first_dig == 4:
#                         if i == 0:
#                             res.append('IV')

#                         elif i == 1:
#                             res.append('XL')
                        
#                         elif i == 2:
#                             res.append('CD')


#                     elif first_dig == 5:
#                         if i == 0:
#                             res.append('V')

#                         elif i == 1:
#                             res.append('L')
                        
#                         elif i == 2:
#                             res.append('D')
                    

#                     elif first_dig in range(6,9):
#                         if i == 0:
#                             res.append('V'+'I'*(first_dig-5))

#                         elif i == 1:
#                             res.append('L'+'X'*(first_dig-5))
                        
#                         elif i == 2:
#                             res.append('D'+'C'*(first_dig-5))


#                     else:
#                         if i == 0:
#                             res.append('IX')

#                         elif i == 1:
#                             res.append('XC')
                        
#                         elif i == 2:
#                             res.append('CM')

#         # Reverse back and join the 'res' holder
#         res = ''.join(res[::-1])
        
#         return res


#     # Testing
#     print(intToRoman(num=num))

#     '''Note: While this approach works, it a bit verbose and could be confusing compared to the Greedy approach'''




#     '''
#     Greedy Approach
    
#         Explanation:

#         Roman Numeral Mapping:
#             Use a list of tuples to map integer values to their corresponding Roman numeral symbols. The list is ordered from largest to smallest.
        
#         Iterate Through the Map:
#             For each (value, symbol) in the map, repeatedly subtract value from num and append symbol to the result string until num is smaller than value.
        
#         Return the Result:
#             Once all values have been processed, the accumulated result contains the Roman numeral representation.
#     '''

#     def intToRoman(num: int) -> str:

#         # Define a list of tuples mapping Roman numeral values to their symbols
#         roman_map = [
#             (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
#             (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
#             (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
#         ]
        
#         # Initialize the result
#         result = ""
        
#         # Process the integer
#         for value, symbol in roman_map:

#             # Append the Roman numeral symbol while the value fits into num
#             while num >= value:
#                 result += symbol
#                 num -= value
        
#         return result
    

#     # Testing
#     print(intToRoman(num=num))

#     '''Note: Done'''

'''13. Roman to Integer'''
def x():

    from typing import Optional

    # Input
    # Case 1
    s = 'III'
    # Output: 3

    # Case 2
    s = 'LVIII'
    # Output: 58

    # Case 3
    s = 'MCMXCIV'
    # Output: 1994
    
    # Custom Case
    s = 'DCXXI'
    # Output: 621

    '''
    My Approach

        Substraction exceptions:
        - I can be placed before V (5) and X (10) to make 4 and 9. 
        - X can be placed before L (50) and C (100) to make 40 and 90. 
        - C can be placed before D (500) and M (1000) to make 400 and 900.
    '''

    def romanToInt(s: str) -> int:

        # Aux Dict creation
        dic = {
            'I':1, 'IV':4, 'V':5, 'IX':9, 
            'X':10, 'XL':40, 'L':50, 'XC':90,
            'C':100, 'CD':400, 'D':500, 'CM':900,
            'M':1000
            }

        # Initialize a Result Holder
        res: int = 0

        # Initialize an Index holder to better handle the positions
        i = 0

        # Numbers that substract
        subs = ('I', 'X', 'C')

        # Process the input
        while i < len(s):

            if i == len(s)-1:
                res += dic[s[i]]
                i += 1
            
            else:

                if s[i] in subs and dic[s[i+1]] > dic[s[i]]:
                    res += dic[s[i:i+2]]
                    i += 2
                
                else:
                    res += dic[s[i]]
                    i += 1
        
        return res


    # Testing
    print(romanToInt(s=s))

    '''Note: Done'''




    '''ChatGPT's Approach'''
    def romanToInt(s: str) -> int:

        roman_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total = 0
        prev_value = 0

        for char in s[::-1]:    #Reverse to simplify the process
            
            curr_value = roman_dict[char]

            if curr_value < prev_value:
                total -= curr_value
            
            else:
                total += curr_value
                prev_value = curr_value
        
        return total
    
    # Testing
    print(romanToInt(s=s))

    '''Note: Done'''






















'OTHERS'


'AgileEngine: Minimal Balls Move'
# def min_moves_balls(buckets: str) -> int:

#     # Step 1: Count number of balls ('B')
#     ball_count = buckets.count('B')
    
#     # Corner case: No balls
#     if ball_count == 0:
#         return 0  # No balls, no moves needed
    
#     # Corner case: No enought spaces
#     if ball_count > buckets.count('.'):
#         return -1  # It's impossible to arrange with valid spacing
    
#     # Step 2: Initialize sliding window
#     start_index = 0
#     end_index = 2 * ball_count - 2  # Window size (the '-2' actually is just '-1' but since it'll be an index)
#     min_shifts = ball_count  # Start with max possible shifts
    
#     # Step 3: Slide the window across the string
#     while end_index < len(buckets):
        
#         ball_correct_pos = 0
#         window = buckets[start_index:end_index+1]   #This line is just to follow through the solution to understand it
        
#         # Count balls in correct positions (with gaps of 2 between them)
#         for i in range(start_index, end_index + 1, 2):
#             if buckets[i] == 'B':
#                 ball_correct_pos += 1
        
#         # Calculate the number of shifts needed for this window
#         shifts = ball_count - ball_correct_pos
#         min_shifts = min(min_shifts, shifts)
        
#         # Move the window
#         start_index += 1
#         end_index += 1
    
#     return min_shifts










