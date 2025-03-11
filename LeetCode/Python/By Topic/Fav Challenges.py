'''
CHALLENGES INDEX

[D]: Done

ARRAYS
[D] 11. Container With Most Water (Array)
->  31. Next Permutation (Array) (TP)
    42. Trapping Rain Water (Array)
    46. Permutations (Array)
    45. Jump Game II (Array) (GRE) (DP)
    55. Jump Game (Array) (DP) (GRE)
[D] 56. Merge Intervals (Array) [Intervals]
[D] 57. Insert Interval (Array) [Intervals]
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
[D] 452. Minimum Number of Arrows to Burst Balloons (Array) [Intervals]
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
[D] 112. Path Sum (Tree) (DFS)    


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
[D] 20. Valid Parentheses (Stack)
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


(91)
'''




#Template
"xxx"
"""xxx"""
def x():
    
    from typing import Optional

    # Input
    # Case 1
    head = None
    # Output: None

    '''
    My Approach

        Intuition:
            
            -...
    '''

    def y() -> int:

        # Handle Corner case: ...
        if not head:
            return
                
        # Return ...
        return 

    # Testing
    print(y())

    '''Note: Done'''


# Binary Tree Node Definition
class TreeNode:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

# List Node Definition
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# Pretty Print Function
def pretty_print_bst(node:TreeNode, prefix="", is_left=True):

    if not node:
        return

    if node.right is not None:
        pretty_print_bst(node.right, prefix + ("│   " if is_left else "    "), False)

    print(prefix + ("└── " if is_left else "┌── ") + str(node.val))

    if node.left is not None:
        pretty_print_bst(node.left, prefix + ("    " if is_left else "│   "), True)








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


'''56. Merge Intervals'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     intervals = [[1,3],[2,6],[8,10],[15,18]]
#     # Output: [[1,6],[8,10],[15,18]]

#     # Case 7
#     intervals = [[1,4],[0,2],[3,5],[6,7]]
#     # Output: [[0,5],[6,7]]

#     # Case 88
#     intervals = [[2,3],[2,2],[3,3],[1,3],[5,7],[2,2],[4,6]]
#             # [[1, 3], [2, 3], [2, 2], [2, 2], [3, 3], [4, 6], [5, 7]]
#     # Output: [[1,3],[4,7]]


#     '''
#     My Approach

#         Intuition:
            
#             - Handle corner case: Single Item Input.
#             - Sort the input list by the 'start' of each interval.
#             - Initialize a 'i' index at 0 to handle the while loop.
#             - In a While Loop (while i < len(intervals)):
#                 + if intervals[i][1] >= intervals[i+1][0]:
#                     - Initialize a new interval 'n_int' at [min(intervals[i][0], intervals[i+1][0]), max(intervals[i][1], intervals[i+1][1])]
#                     - Redefine intervals as: intervals = [n_int]+intervals[i+2:]
#                 + Increase i in 1.
#             - Return 'intervals'.
                
#     '''

#     def merge(intervals: list[list[int]]) -> list[list[int]]:

#         # Handle Corner case: Single Item Input
#         if len(intervals) == 1:
#             return intervals
        
#         # Sort the input list by the 'start' element of each interval.
#         intervals.sort(key=lambda x: x[0])

#         # Initialize a 'result' holder at the first interval
#         result = [intervals.pop(0)]

#         # Process the input
#         for inter in intervals:
            
#             if result[-1][1] >= inter[0]:

#                 # Pull the last interval in the result holder
#                 last = result.pop()

#                 n_int = [min(last[0], inter[0]), max(last[1], inter[1])]
#                 result.append(n_int)
            
#             else:
#                 result.append(inter)
        
#         # Return result
#         return result

#     # Testing
#     print(merge(intervals=intervals))

#     '''Note: This approach beated by 74.25% in Runtime, 42.2% in Memory'''

'''57. Insert Interval'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     intervals = [[1,3],[6,9]]
#     newInterval = [2,5]
#     # Output: [[1,5],[6,9]]

#     # Case 2
#     intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]]
#     newInterval = [4,8]
#     # Output: [[1,2],[3,10],[12,16]]

#     # Case 3
#     intervals = []
#     newInterval = [5,7]
#     # Output: [[5,7]]

#     # Case 4
#     intervals = [[1,5]]
#     newInterval = [2,3]
#     # Output: [[1,5]]

#     # Case 5
#     intervals = [[1,5]]
#     newInterval = [2,7]
#     # Output: [[1,7]]

#     # Case 6
#     intervals = [[1,5]]
#     newInterval = [6,8]
#     # Output: [[1,5],[6,8]]

#     # Case 7
#     intervals = [[1,5]]
#     newInterval = [0,0]
#     # Output: [[0,0],[1,5]]

#     # Custom Case
#     intervals = [[2,6],[7,9]]
#     newInterval = [15,18]
#     # Output: [[2,6],[7,9],[15,18]]


#     '''
#     My Approach

#         Intuition:
            
#             - Handle Corner Case: Empty intervals.
#             - Insert the new interval into intervals.
#             - Sort the intervals by each item first element.  
#             - Create a result holder initialized in the element popped out from intervals in its first index
#             - In a for loop (for inter in intervals):
#                 + Compare the last element of result's last item with the first element of inter, if they overlap, merge them.
#                 + Otherwise, simply add inter to result.
#             - Return result.
#     '''         
  
#     def insert(intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:

#         # Handle Corner case: Empty intervals
#         if not intervals:
#             return [newInterval]
        
#         intervals.append(newInterval)
#         intervals.sort(key=lambda x: x[0])

#         # Initialize the 'result' holder in the first interval elements
#         result = [intervals.pop(0)]

#         for inter in intervals:

#             if result[-1][1] >= inter[0]:
#                 result[-1] = [min(result[-1][0], inter[0]), max(result[-1][1], inter[1])]
            
#             else:
#                 result.append(inter)     
               
#         # Return result
#         return result

#     # Testing
#     # print(insert(intervals=intervals, newInterval=newInterval))

#     '''Note: Done'''



    
#     '''
#     Another more elegant solution

#     Intuition:        
#         - Add all the intervals smaller that the new one.
#         - Merge all the intervals overlapping with the new one.
#         - Add any remaining interval.
#     '''         
  
#     def insert(intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:

#         # Capture the input length
#         n = len(intervals)

#         # Initialize the result holder and the index
#         result = []
#         i = 0

#         # Add all the intervals smaller that the new one.   
#         while i < n and newInterval[0] > intervals[i][1]:
#             result.append(intervals[i])
#             i += 1

#         # Merge all the overlapping intervals
#         while i < n and newInterval[1] >= intervals[i][0]:
#             newInterval[0] = min(newInterval[0], intervals[i][0])
#             newInterval[1] = max(newInterval[1], intervals[i][1])
#             i += 1

#         # Add the merged intervals
#         result.append(newInterval)

#         # Add any remaining intervals not processed
#         while i < n:
#             result.append(intervals[i])
#             i+=1

#         # Return result
#         return result

#     # Testing
#     print(insert(intervals=intervals, newInterval=newInterval))

#     'Notes: Done'

'''452. Minimum Number of Arrows to Burst Balloons'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     points = [[10,16],[2,8],[1,6],[7,12]]
#     # Output: 2
#     # Explanation: One way is to shoot one arrow for example at x = 6 (bursting the balloons [2,8] and [1,6])

#     # Case 2
#     points = [[1,2],[3,4],[5,6],[7,8]]
#     # Output: 4
#     # Explanation: Since the balloons don't overlap, we only need 4 arrows.

#     # Case 3
#     points = [[1,2],[2,3],[3,4],[4,5]]
#     # Output: 2
#     # Explanation: One way is to shoot one arrow for example at x = 4 (bursting the balloons [1,2] and [3,4])

#     # Case 4
#     points = [[1,2]]
#     # Output: 1
#     # Explanation: You can burst the balloon by shooting the arrow at the end of the balloon.

#     # Case 5
#     points = [[2,3],[2,3]]
#     # Output: 1
#     # Explanation: You can burst the balloon by shooting the arrow at the end of the balloon.

#     # Case 6
#     points = [[1,2],[1,2],[1,2]]
#     # Output: 1
#     # Explanation: You can burst the balloon by shooting the arrow at the end of the balloon.

#     # Case 7
#     points = [[1,2],[2,3]]
#     # Output: 1
#     # Explanation: You can burst the balloon by shooting the arrow at the end of the balloon.

#     # Case 8
#     points = [[1,2],[2,3],[3,4]]
#     # Output: 2

#     # Custom Case
#     points = [[3,9],[7,12],[3,8],[6,8],[9,10],[2,9],[0,9],[3,9],[0,6],[2,8]]
#             #[(0, 6), (2, 9), (2, 8), (3, 8), (3, 9), (6, 8), (7, 12), (9, 10)]
#     # Output: 2



#     '''
#     My Approach

#         Analysis:

#             - This problem could be rethink as finding the unique point and the overlapping intervals. 
        
            
#         Intuition:
            
#             - Make a set out of the input.
#             - Handle corner case: 1 element input / several elements on the same spot.
#             - Sort the set to have a ascendingly ordered list by element's first item.
#             - Initialize a 'visited' holder, to hold the visited ballons initialized in the popped first item of balloons.
#             - Create an 'arrows' counter initilized in 1.   
#             - In a forloop (point in points):
#                 + Initialize a 'last' holder in the last element of 'visited'.
#                 + if 'last' do not overlap with the current 'point':
#                     * add 1 up to arrows.
#                     * append 'point' to 'visited'.
#                 + else:
#                     * Redefine 'last' as the overlapping range.
#                     * Append 'last' to 'visited'.
#             - Return 'Arrows'.

#     '''

#     def findMinArrowShots(points: list[list[int]]) -> int:

#         # Make a set out of the input
#         points = set(tuple(elem) for elem in points)

#         # Handle Corner case: element input / several elements on the same spot.
#         if len(points) == 1:
#             return 1

#         # Sort the set to have a ascendingly ordered list
#         points = sorted(points, key=lambda x:x[0])

#         # Initilize a visited holder
#         visited = [points.pop(0)]

#         # Create an result counter
#         arrows = 1  # Because if there's at least one element after the corner case guard, at least 1 arrow will be thrown

#         # Process each balloon
#         for point in points:

#             last = visited[-1]

#             if last[1] < point[0]:
#                 arrows += 1
#                 visited.append(point)
            
#             else:
#                 last = [max(last[0], point[0]), min(last[1], point[1])]
#                 visited.append(last)

#         # Return the arrows counted
#         return arrows

#     # Testing
#     print(findMinArrowShots(points=points))

#     '''Note: 
#         This solves the probelm beating submissions by 7% in Runtime and 6% in Memory. 
        
#         Meaning is not as efficient, but there is a more elegant and efficient way to do it.'''




#     '''
#     Greedy approach

#         Explanation
            
#             1. Sort the intervals by their ending points (point[1]).
#             2. Use a greedy approach to count the minimum number of arrows:
#                 - Start with the first balloon and shoot an arrow at its end point (point[1]).
#                 - For every subsequent balloon, if its start point (point[0]) is greater than the current arrow position, shoot a new arrow.

#     '''

#     def findMinArrowShots(points: list[list[int]]) -> int:

#         points.sort(key=lambda x: x[1])
#         arrows = 1
#         arrow_pos = points[0][1]

#         for point in points[1:]:
#             if point[0]> arrow_pos:
#                 arrows += 1
#                 arrow_pos = point[1]

#         # Return the arrows counted
#         return arrows

#     # Testing
#     # print(findMinArrowShots(points=points))

#     '''Note: This approach beated submissions by 72% in Runtime and 20% in Memory'''








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
# def x():

#     from typing import Optional

#     # Input
#     # Case 1
#     s = 'III'
#     # Output: 3

#     # Case 2
#     s = 'LVIII'
#     # Output: 58

#     # Case 3
#     s = 'MCMXCIV'
#     # Output: 1994
    
#     # Custom Case
#     s = 'DCXXI'
#     # Output: 621

#     '''
#     My Approach

#         Substraction exceptions:
#         - I can be placed before V (5) and X (10) to make 4 and 9. 
#         - X can be placed before L (50) and C (100) to make 40 and 90. 
#         - C can be placed before D (500) and M (1000) to make 400 and 900.
#     '''

#     def romanToInt(s: str) -> int:

#         # Aux Dict creation
#         dic = {
#             'I':1, 'IV':4, 'V':5, 'IX':9, 
#             'X':10, 'XL':40, 'L':50, 'XC':90,
#             'C':100, 'CD':400, 'D':500, 'CM':900,
#             'M':1000
#             }

#         # Initialize a Result Holder
#         res: int = 0

#         # Initialize an Index holder to better handle the positions
#         i = 0

#         # Numbers that substract
#         subs = ('I', 'X', 'C')

#         # Process the input
#         while i < len(s):

#             if i == len(s)-1:
#                 res += dic[s[i]]
#                 i += 1
            
#             else:

#                 if s[i] in subs and dic[s[i+1]] > dic[s[i]]:
#                     res += dic[s[i:i+2]]
#                     i += 2
                
#                 else:
#                     res += dic[s[i]]
#                     i += 1
        
#         return res


#     # Testing
#     print(romanToInt(s=s))

#     '''Note: Done'''




#     '''ChatGPT's Approach'''
#     def romanToInt(s: str) -> int:

#         roman_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
#         total = 0
#         prev_value = 0

#         for char in s[::-1]:    #Reverse to simplify the process
            
#             curr_value = roman_dict[char]

#             if curr_value < prev_value:
#                 total -= curr_value
            
#             else:
#                 total += curr_value
#                 prev_value = curr_value
        
#         return total
    
#     # Testing
#     print(romanToInt(s=s))

#     '''Note: Done'''







"MATRICES"








"DFS & BFS"

'''112. Path Sum'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     root = [5,4,8,11,None,13,4,7,2,None,None,None,1]
#     targetSum = 22
#     root = TreeNode(5)
#     root.left = TreeNode(4)
#     root.right = TreeNode(8)
#     root.left.left = TreeNode(11)
#     root.right.left = TreeNode(13)
#     root.right.right = TreeNode(4)
#     root.left.left.left = TreeNode(7)
#     root.left.left.right = TreeNode(2)
#     root.right.right.right = TreeNode(1)
#     # Output: True

#     # Case 2
#     root = [1,2,3]
#     targetSum = 5
#     root = TreeNode(1)
#     root.left = TreeNode(2)
#     root.right = TreeNode(3)
#     # Output: False

#     # Case 3
#     root = [1,2]
#     targetSum = 0
#     root = TreeNode(1)
#     root.left = TreeNode(2)
#     # Output: False

#     '''
#     My Approach (Depth-Fisrt Search)

#         Intuition:
            
#             - Traverse iteratively the BT having a running sum for each path
#     '''

#     def hasPathSum(root: Optional[TreeNode], targetSum: int) -> bool:

#         # No node guard
#         if not root:
#             return False
            
#         # Initialize the stack to hold the running path sum
#         stack = [(root, root.val)]

#         # Traverse the tree
#         while stack:

#             # Take the last element in the stack
#             node, path_sum = stack.pop()

#             # If it's a leaf node, compare it to the target
#             if not node.left and not node.right:
#                 if path_sum == targetSum:
#                     return True

#             # Push right and left children onto the stack
#             if node.right:
#                 stack.append((node.right, node.right.val + path_sum))

#             if node.left:
#                 stack.append((node.left, node.left.val + path_sum))

#         # If the code gets up to here, means it didn't find the target
#         return False

#     # Testing
#     print(hasPathSum(root=root, targetSum=targetSum))

#     '''Note: Done'''








'HEAPS, STACKS & QUEUES'

'20. Valid Parentheses'
# def x():

#     from typing import Optional

#     # Input
#     # Case 1
#     s = "()"
#     # Output: True

#     # Case 2
#     s = "()[]{}"
#     # Output: True

#     # Case 3
#     s = "(]"
#     # Output: False

#     # Case 4
#     s = "([])"
#     # Output: True

#     # Custom Case
#     s = "))"
#     # Output: False

#     '''
#     My Approach (Stack)

#         Intuition:
            
#             - Hanlde corner case: If input string length is not even
#             - Initialize a 'stack' holder to store the closing parenthesis.
#             - Initialize a 'par' dictionary at each opening parentesis char as key and its correspondent closing as value.
#             - Iterate from right to left:
#                 + Pop each element.
#                 + if the element is a closing parenthesis inserted as first value in 'stack'.
#                 + else:
#                     + The first 'stack' element should correspond (with 'par's help) to the last popped, if it doesn't:
#                         * Return False
#                         * Otherwise, pop the first element of 'stack' and continue the iterations.
            
#             - If the code gets to this point, return True.

#     '''

#     def isValid(s: str) -> bool:
        
#         # Turn the input into a list
#         s = list(s)

#         # Handle Corner case: Odd lengthed input
#         if len(s)%2 != 0:
#             return False
        
#         # Initialize a 'stack' holder to store the closing parenthesis.
#         stack = []

#         # Initialize a 'par' dictionary
#         par = {
#             '(':')',
#             '{':'}',
#             '[':']',
#         }

#         for i in range(len(s)-1,-1,-1):

#             char = s[i]

#             if char in ')}]':
#                 stack.insert(0, char)
            
#             else:

#                 if not stack or par[char] != stack[0]:
#                     return False

#                 else:
#                     stack.pop(0)

#         # Return True if it gets to this point
#         return True if not stack else False

#     # Testing
#     print(isValid(s=s))

#     'Notes: it works!'














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










