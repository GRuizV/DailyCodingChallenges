'''
CHALLENGES INDEX

ARRAYS
11. Container With Most Water (Array)
31. Next Permutation (Array) (TP)
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
12. Integer to Roman (Hash Table) (GRE)
13. Roman to Integer (Hash Table)
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










