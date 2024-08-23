'''
CHALLENGES INDEX

50. Pow(x, n) (RC)
53. Maximum Subarray (DP)
54. Spiral Matrix (Matrix)  
55. Jump Game (DP)
56. Merge Intervals
62. Unique Paths (DP)
66. Plus One
69. Sqrt(x)
70. Climbing Stairs (DP)
73. Set Matrix Zeroes (Matrix)
75. Sort Colors (TP)
76. Minimum Window Substring
78. Subsets
79. Word Search (Matrix)
88. Merge Sorted Array (TP)
91. Decode Ways (DP) 
98. Validate Binary Search Tree (DFS)
101. Symmetric Tree (BFS) (DFS)
102. Binary Tree Level Order Traversal (BFS)
103. Binary Tree Zigzag Level Order Traversal (BFS) (DFS)
104. Maximum Depth of Binary Tree (BFS)
105. Construct Binary Tree from Preorder and Inorder Traversal
108. Convert Sorted Array to Binary Search Tree
116. Populating Next Right Pointers in Each Node (BFS) (DFS)

*DP: Dynamic Programming
*RC: Recursion
*TP: Two-pointers

(25)

'''




'50. Pow(x, n)'

# Input

# # Case 1
# x = 2.00000
# n = 10
# # Output: 1024.00000

# # Case 2
# x = 2.10000
# n = 3
# # Output: 9.26100

# # Case 3
# x = 2.00000
# n = -2
# # Output: 0.25000

# # Custom Case
# x = 0.00001
# n = 2147483647
# # Output: ...


# My Approach
# def myPow(x, n):

#     if x == 0:
#         return 0
    
#     if n == 0:
#         return 1

#     res = 1

#     for _ in range(abs(n)):
#         res *= x

#     if n > 0:
#         return f'{res:.5f}'
    
#     else:        
#         return f'{(1/res):.5f}'


# print(myPow(x, n))

'Notes: it works, but broke memory with the case: x = 0.00001, n=2147483647, it is 95% of the solution'


# # Another Approach
# def myPow(x: float, n: int) -> float:

#     b = n

#     if x == 0:
#         return 0
    
#     elif b == 0:
#         return 1
    
#     elif b < 0:
#         b = -b
#         x = 1 / x
    

#     a = 1

#     while b > 0:

#         if b % 2 == 0:
#             x = x * x
#             b = b // 2

#         else:
#             b = b - 1
#             a = a * x
            
#     return a

# print(myPow(x, n))

'''
Notes: 
    This solution takes advantage of the property x^(2n) = (x^2)^n, 
    saving a lot of time reducing in half the calculations each time the exponent is even.
'''




'53. Maximum Subarray'

# Input

# # Case 1
# nums = [-2,1,-3,4,-1,2,1,-5,4]
# # Output: 6 / [4,-1,2,1]

# # Case 2
# nums = [1]
# # Output: 1 / [1]

# # Case 3
# nums = [5,4,-1,7,8]
# # Output: 23 / [5,4,-1,7,8]



# My approach

'''
Intuition: Brute forcing
    1. Store the sum of the max array
    2. From the max len down to len = 1, evaluate with a sliding window each sub array.
    3. Return the max sum
'''


# def maxSubArray(nums):

#     max_sum = sum(nums)

#     # Handling the special case len = 1
#     if len(nums) == 1:
#         return max_sum
    
    
#     max_element = max(nums)
#     nums_len = len(nums)-1
#     idx = 0

#     while nums_len > 1:

#         if idx + nums_len > len(nums):
#             nums_len -= 1
#             idx = 0
#             continue
            
#         sub_array = nums[idx:idx+nums_len]
#         sub_array_sum = sum(sub_array)

#         if sub_array_sum > max_sum:
#             max_sum = sub_array_sum
            
#         idx += 1

#     # Handling the case where one element is greater than any subarray sum
#     if max_element > max_sum:
#         return max_element
    
#     return max_sum


# print(maxSubArray(nums))
        

'Notes: My solution worked 94,7% of the cases, the time limit was reached.'




# Kadane's Algorythm (for max subarray sum)

# def maxSubArray(nums):

#     max_end_here = max_so_far = nums[0]

#     for num in nums[1:]:

#         max_end_here = max(num, max_end_here + num)
#         max_so_far = max(max_so_far, max_end_here)
    
#     return max_so_far


# print(maxSubArray(nums))

'Notes: Apparently it was a classic problem'





'54. Spiral Matrix'

# Input

# # Case 1
# matrix = [[1,2,3],[4,5,6],[7,8,9]]
# # Output: [1,2,3,6,9,8,7,4,5]

# # Case 2
# matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
# # Output: [1,2,3,4,8,12,11,10,9,5,6,7]

# # Custom Case  / m = 1
# matrix = [[2,1,3]]
# # Output: [2,1,3]

# # Custom Case / n = 1
# matrix = [[2],[1],[3],[5],[4]]
# # Output: [2,1,3,5,4]

# # Custom Case
# matrix = [[1,2],[3,4]]
# # Output: [1,2,4,3]

# # Custom Case
# matrix = [[2,3,4],[5,6,7],[8,9,10],[11,12,13],[14,15,16]]
# # Output: [2,3,4,7,10,13,16,15,14,11,8,5,6,9,12]




# My approach

'''
Intuition:
    1. Handle Special Cases: m = 1 / n = 1 

    2. Make a while loop that runs until the input has no elements:
        a. Take all the subelements from the first element and append them individually to the result.
        b. Make a for loop and take the last subelement of each element and append them individually to the result
        c. Take all the subelements from the last element and append them in a reverse order individually to the result.
        d. Make a for loop and take the first subelement of each element and append them in a reverse order individually to the result.

     3. Return the result
'''

# def spiralOrder(matrix):

#     if len(matrix) == 1:
#         return matrix[0]
    
#     if len(matrix[0]) == 1:
#         return [num for vec in matrix for num in vec]
    

#     result = []

#     while len(matrix) != 0:

#         first_element = matrix.pop(0)
#         result += first_element

#         if len(matrix) == 0:
#             break

#         second_element = []
#         for elem in matrix:
#             second_element.append(elem.pop(-1))
#         result += second_element

#         third_element = matrix.pop(-1)
#         result += reversed(third_element)
        
#         if len(matrix) > 0 and len(matrix[0]) == 0:
#             break

#         fourth_element = []
#         for elem in matrix:
#             fourth_element.append(elem.pop(0))
#         result += reversed(fourth_element)

#     return result

# print(spiralOrder(matrix))


'Notes: it works up to 76% of the cases, but from here seems more like patching something that could be better designed'


# Another Approach

# def spiralOrder(matrix):
        
#         result = []

#         while matrix:
#             result += matrix.pop(0) # 1

#             if matrix and matrix[0]: # 2 
#                 for line in matrix:
#                     result.append(line.pop())

#             if matrix: # 3
#                 result += matrix.pop()[::-1]

#             if matrix and matrix[0]: # 4

#                 for line in matrix[::-1]:
#                     result.append(line.pop(0))

#         return result


'Notes: Same logic, better executed'





'55. Jump Game'

# Input

# # Case 1
# nums = [2,3,1,1,4]
# # Output: True

# # Case 2
# nums = [3,2,1,0,4]
# # Output: False

# # Custom Case
# nums = [2,0,0]
# # Output: 


# My Approach

'''
Intuition: Brute Force

    I will check item by item to determine if the end of the list is reachable

'''


# def canJump(nums:list[int]) -> bool:

#     # Corner case: nums.lenght = 1 / nums[0] = 0
#     if len(nums) == 1 and nums[0] == 0:
#         return True
    
#     idx = 0

#     while True:

#         idx += nums[idx]

#         if idx >= len(nums)-1:
#             return True
        
#         if nums[idx] == 0 and idx < len(nums)-1:
#             return False
        

# print(canJump(nums))

'Notes: This solution suffice 91,2% of the case'


#Backtrack Approach

# def canJump(nums: list[int]) -> bool:

#     if len(nums)==1:
#         return True  

#     #Start at num[-2] since nums[-1] is given
#     backtrack_index = len(nums)-2 
#     #At nums[-2] we only need to jump 1 to get to nums[-1]
#     jump =1  

#     while backtrack_index>0:
#         #We can get to the nearest lily pad
#         if nums[backtrack_index]>=jump: 
#             #now we have a new nearest lily pad
#             jump=1 
#         else:
#             #Else the jump is one bigger than before
#             jump+=1 
#         backtrack_index-=1
    
#     #Now that we know the nearest jump to nums[0], we can finish
#     if jump <=nums[0]: 
#         return True
#     else:
#         return False 

'Notes: Right now I am not that interested in learning bactktracking, that will be for later'





'56. Merge Intervals'

#Input

# # Case 1
# intervals = [[1,3],[2,6],[8,10],[15,18]]
# # Output: [[1,6],[8,10],[15,18]]

# # Case 2
# intervals = [[1,4],[4,5]]
# # Output: [[1,5]]

# # Custom Case
# intervals = [[1,4],[0,0]]
# # Output: [...]


# My Approach

'''
Intuition:

    - Check the second item of the element and the first of the next, 
    if they coincide, merge.
        (Through a While Loop)
'''

# def merge(intervals:list[list[int]]) -> list[list[int]]:

#     #Handling the corner case
#     if len(intervals) == 1:
#         return intervals

#     intervals.sort(key=lambda x: x[0])

#     idx = 0

#     while idx < len(intervals)-1:

#         if intervals[idx][1] >= intervals[idx+1][0]:

#             merged_interval = [[min(intervals[idx][0], intervals[idx+1][0]), max(intervals[idx][1], intervals[idx+1][1])]]
#             intervals = intervals[:idx] + merged_interval + intervals[idx+2:]
#             idx = 0

#         else:
#             idx += 1

#     return intervals

# print(merge(intervals))

'Note: My solution works but is not efficient, since it has to go over the whole array again'



# Some other Approach

# def merge(intervals):
#     """
#     :type intervals: List[List[int]]
#     :rtype: List[List[int]]
#     """
#     intervals.sort()

#     merge_intervals = []
#     curr_interval = intervals[0]

#     for interval in intervals[1:]:

#         if curr_interval[1] < interval[0]:
#             merge_intervals.append(curr_interval)
#             curr_interval = interval

#         else:
#             curr_interval[1] = max(curr_interval[1], interval[1])

#     merge_intervals.append(curr_interval)

#     return merge_intervals


# print(merge(intervals))





'62. Unique Paths'

'''
*1st Dynamic programming problem

Notes:

This problem was pretty similar to the one on Turing's tests, althought here is requested
to find a bigger scale of thar problem. The classic 'How many ways would be to get from x to y',

if the problem were only set to m = 2, it'd be solved with fibonacci, but sadly that was not the case,
here, Dynamic Programming was needed.

The problem is graphically explained here: https://www.youtube.com/watch?v=IlEsdxuD4lY

But the actual answer I rather take it from the leetCode's solutions wall, since is more intuitive to me.

'''

# Input

# # Case 1:
# m, n = 3, 7
# # Output: 28

# # Case 2:
# m, n = 3, 2
# # Output: 3


# Solution

# def uniquePaths(m: int, n: int) -> int:

#     # Handling the corner case in which any dimention is 0
#     if n == 0 or m == 0:
#         return 0


#     # Here the grid is initialized
#     result = [[0]*n for _ in range(m)]

#     # The first column of the grid is set to 1, since there is only (1) way to get to each cell of that column
#     for row in range(m):
#         result[row][0] = 1

#     # The first row of the grid is set to 1, since there is only (1) way to get to each cell of that row
#     for col in range(n):
#         result[0][col] = 1


#     # Here all the grid is traversed summing up the cells to the left and up, since are the only ways to get to the current cell
#     # The range starts in 1 since all the first column and row are populated, so the traversing should start in [1,1]
#     for i in range(1, m):

#         for j in range(1, n):

#             result[i][j] = result[i-1][j] + result[i][j-1]
    

#     # The bottom right cell will store all the unique ways to get there
#     return result[-1][-1]


# print(uniquePaths(m, n))





'66. Plus One'

# Input

# # Case 1
# digits = [1,2,3]
# # Output: [1,2,4]

# # Case 2
# digits = [4,3,2,1]
# # Output: [4,3,2,2]

# # Case 3
# digits = [9]
# # Output: [1,0]

# # Custom Case
# digits = [9,9,9]
# # Output: [1,0,0,0]


# My approach

'''
Intuition:
    - The case is simple, the catch is to handle the case "[9,9,9]"
'''

# def plusOne(digits: list[int]) -> list[int]:

#     idx = -1

#     while abs(idx) <= len(digits):
        
#         if abs(idx) == len(digits) and digits[idx] == 9:

#             digits[idx] = 1
#             digits.append(0)
#             break

#         if digits[idx] != 9:

#             digits[idx] += 1
#             break

#         digits[idx] = 0
#         idx -= 1

#     return digits


# print(plusOne(digits=digits))

'''
Notes: 
While this code works, there was an even cleverer approach - To convert the digits into a int, add 1 and return as a list of ints
this way, is avoided the handling of cases
'''

# A different Approach

# def plusOne(digits: list[int]) -> list[int]:

#     number = int(''.join([str(x) for x in digits]))
#     number += 1
    
#     return [int(x) for x in str(number)]


# print(plusOne(digits=digits))





'69. Sqrt(x)'

# Input

# # Case 1
# x = 4
# # Output: 2

# # Case 2
# x = 8
# # Output: 2

# # Custom Case
# x = 399
# # Output: ..




# My Approach

# limit = 46341

# # Auxiliary Eratosthenes' sieve function
# def primes(cap):  

#     primes = []
#     not_primes = []

#     for i in range(2, cap+1):

#         if i not in not_primes:
#             primes.append(i)
#             not_primes.extend([x for x in range(i*i, cap+1, i)])

#     return primes


# def mySqrt(x:int) -> int:

#     #Setting a limit for calculating primes
#     limit = x//2

#     prime_nums = primes(limit)

#     squares = list(map(lambda x: x*x, prime_nums))


#     #proning in the squares the correct range to make a range to evaluate
#     root_range = []
#     for idx, v in enumerate(squares):

#         if x <= v:
#             root_range = [prime_nums[idx-1], prime_nums[idx]]
#             break


#     #Calculating manually the square of each root in range to select the floor-root for the value
#     for root in range(root_range[0], root_range[1]+1):
        
#         if root*root >= x:
#             return result
        
#         result = root


# print(mySqrt(x))

'Notes: This approach was too complicated and actually no as efficient. Apparently with the notion of binary search is easier to solve'


# # Binary Search Approach

# def mySqrt(x):

#     left = 0
#     right = x

#     while left <= right:

#         mid = (left + right)//2

#         if mid*mid < x:
#             left = mid + 1

#         elif mid*mid > x: 
#             right = mid -1

#         else:
#             return mid
    
#     return right
    

# print(mySqrt(x))






'70. Climbing Stairs'

# Input

# # Case 1
# n = 2
# # Output = 2

# # Case 2
# n = 3
# # Output = 3

# # Case 2
# n = 5
# # Output = 3

# Solution

# def climbStairs(n:int) -> int:

#     res = [1,1]

#     for i in range(2, n+1):
#         res.append(res[i-2]+res[i-1])
    

#     return res[-1]


# print(climbStairs(n))

'Notes: The recursive solution, while elegant and eyecatching, is not as efficient as an iterative one'






'73. Set Matrix Zeroes'

# Input

# # Case 1
# matrix = [[1,1,1],[1,0,1],[1,1,1]]
# # Output: [[1,0,1],[0,0,0],[1,0,1]]

# # Case 2
# matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
# # Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]

# # Custom Case
# matrix = [...]
# # Output: [...]



# My Approach

'''
Intuition:
    - Locate the indexes of every 0 present
    - Try to overwrite the values for the row and the column of each occurrence
    - Look up if the col and row are already 0 to optimize

'''

# def setZeroes(matrix: list[list[int]]) -> list[list[int]]:

#     m, n = len(matrix), len(matrix[0])

#     occurrences = []

#     for i, row in enumerate(matrix):

#         for j, col in enumerate(row):

#             if 0 not in row:
#                 continue
            
#             if col == 0:
#                 occurrences.append((i,j))


#     for pair in occurrences:

#         matrix[pair[0]] = [0] * n

#         for row in range(m):
#             matrix[row][pair[1]] = 0

    
#     return matrix


# for i in setZeroes(matrix):
#     print(i)

'''
Notes: It actually passed! :D
'''






'75. Sort Colors'

# Input

# # Case 1
# nums = [2,0,2,1,1,0]
# # Output: [0,0,1,1,2,2]

# # Case 2
# nums = [2,0,1]
# # Output: [0,1,2]


# My approach

'''
Intuition:
    Since the solution requires the sorting be in place, 
    perhaps Bubblesort would do the trick.
'''


# def sortColors(nums:list[int]) -> list[int]:

#     swapped = True
    

#     while swapped != False:

#         swapped = False
#         i = 0

#         while True:

#             if i == len(nums)-1:
#                 break

#             if nums[i] > nums[i+1]:
#                 nums[i], nums[i+1] = nums[i+1], nums[i]
#                 swapped = True

#             i += 1


# sortColors(nums)

# print(nums)

'Notes: Done!'




'76. Minimum Window Substring'

# Input

# # Case 1
# s, t = 'ADOBECODEBANC', 'ABC'
# # Output: "BANC"

# # Case 2
# s, t = 'a', 'a'
# # Output: "a"

# # Case 3
# s, t = 'a', 'aa'
# # Output: "abbbbbcdd"

# # Custom case
# s, t = 'aaaaaaaaaaaabbbbbcdd', 'abcdd'
# # Output: "abbbbbcdd"


# My approach

# def minWindow(s:str, t:str) -> str:

#     if len(t) > len(s):
#         return ''
    
#     if t == s:
#         return t
    

#     for i in range(len(t), len(s) + 1):

#         for j in range((len(s)-i) + 1):
            
#             if all([char in s[j:j+i] for char in t]):
#                 return s[j:j+i]
            
#     return ''

'Notes: This solution works up to 57%'


# # With an improvement
# def minWindow(s:str, t:str) -> str:

#     from collections import Counter

#     if len(t) > len(s):
#         return ''
    
#     if t == s:
#         return t
    
#     count_t = Counter(t).items()

#     for i in range(len(t), len(s) + 1):

#         for j in range((len(s)-i) + 1):
            
#             subs = s[j:j+i]
#             count_subs = Counter(subs)

#             if all( (x[0] in count_subs.keys() and x[1] <= count_subs[x[0]]) for x in count_t):
#                 return s[j:j+i]
            
#     return ''

'Notes: This solution works up to 93% and hit the time limit'


# Another solution

# def minWindow(s, t):    

#     if not s or not t:
#         return ""


#     from collections import defaultdict

#     dictT = defaultdict(int)
#     for c in t:
#         dictT[c] += 1

#     required = len(dictT)
#     l, r = 0, 0
#     formed = 0

#     windowCounts = defaultdict(int)
#     ans = [-1, 0, 0]

#     while r < len(s):
#         c = s[r]
#         windowCounts[c] += 1

#         if c in dictT and windowCounts[c] == dictT[c]:
#             formed += 1

#         while l <= r and formed == required:
#             c = s[l]

#             if ans[0] == -1 or r - l + 1 < ans[0]:
#                 ans[0] = r - l + 1
#                 ans[1] = l
#                 ans[2] = r

#             windowCounts[c] -= 1
#             if c in dictT and windowCounts[c] < dictT[c]:
#                 formed -= 1

#             l += 1

#         r += 1

#     return "" if ans[0] == -1 else s[ans[1]:ans[2] + 1]
        

# print(minWindow(s,t))




'78. Subsets'

# Input

# # Case 1
# nums = [1,2,3]
# # Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

# # Case 2
# nums = [0]
# # Output: [[],[0]]



# My Approach

'''
Intuition:
    - Perhaps with itertool something might be done
'''

# def subsets(nums:list[int]) -> list[list[int]]:

#     from itertools import combinations

#     result = []

#     for i in range(len(nums)+1):

#         result.extend(list(map(list, combinations(nums,i))))

#     return result


# print(subsets(nums=nums))

'Notes: It actually worked'

# Another more algorithmic Approach

# def subsets(nums: list[int]) -> list[list[int]]:

#     arr = [[]]

#     for j in nums:

#         temp = []

#         for i in arr: 

#             temp.append(i+[j])
        
#         arr.extend(temp)
    
#     return arr 

# print(subsets(nums))

'Notes: The guy who came up with this is genius'




'79. Word Search'

# Input

# # Case 1
# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = 'ABCCED'
# # Output: True


# # Case 2
# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = 'SEE'
# # Output: True


# # Case 3
# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = 'ABCB'
# # Output: False


'''
Intuition:

    The problem can be solved by traversing the grid and performing a depth-first search (DFS) for each possible starting position. 
    At each cell, we check if the current character matches the corresponding character of the word. 
    If it does, we explore all four directions (up, down, left, right) recursively until we find the complete word or exhaust all possibilities.

    Approach

        1. Implement a recursive function backtrack that takes the current position (i, j) in the grid and the current index k of the word.
        2. Base cases:
            - If k equals the length of the word, return True, indicating that the word has been found.
            - If the current position (i, j) is out of the grid boundaries or the character at (i, j) does not match the character at index k of the word, return False.
        3. Mark the current cell as visited by changing its value or marking it as empty.
        4. Recursively explore all four directions (up, down, left, right) by calling backtrack with updated positions (i+1, j), (i-1, j), (i, j+1), and (i, j-1).
        5. If any recursive call returns True, indicating that the word has been found, return True.
        6. If none of the recursive calls returns True, reset the current cell to its original value and return False.
        7. Iterate through all cells in the grid and call the backtrack function for each cell. If any call returns True, return True, indicating that the word exists in the grid. Otherwise, return False.
        
'''


# # Backtracking (Recursive) Approach
# def exist(board: list[list[str]], word: str) -> bool:


#     def backtrack(i, j, k):

#         if k == len(word):
#             return True
        
#         if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[k]:
#             return False
        
#         temp = board[i][j]
#         board[i][j] = ''
        
#         if backtrack(i+1, j, k+1) or backtrack(i-1, j, k+1) or backtrack(i, j+1, k+1) or backtrack(i, j-1, k+1):
#             return True
        
#         board[i][j] = temp

#         return False
        


#     for i in range(len(board)):

#         for j in range(len(board[0])):

#             if backtrack(i, j, 0):

#                 return True
            

#     return False
        

# print(exist(board, word))




'88. Merge Sorted Array'

# Input

# # Case 1
# nums1 = [1,2,3,0,0,0]
# m = 3
# nums2 = [2,5,6]
# n = 3
# # Output: [1,2,2,3,5,6]

# # Case 2
# nums1 = [1]
# m = 1
# nums2 = []
# n = 0
# # Output: [1]

# # Case 3
# nums1 = [0]
# m = 0
# nums2 = [1]
# n = 1
# # Output: [1]

# # Custom case
# nums1 = [0,2,0,0,0,0,0]
# m = 2
# nums2 = [-1,-1,2,5,6]
# n = 5
# # Output: [1]

# # Custom case
# nums1 = [-1,1,0,0,0,0,0,0]
# m = 2
# nums2 = [-1,0,1,1,2,3]
# n = 6
# # Output: [1]



# Solution

# def merge(nums1, m, nums2, n):

#     if m == 0:
#         for i in range(n):
#             nums1[i] = nums2[i]

#     elif n != 0:

#         m = n = 0

#         while n < len(nums2):

#             if nums2[n] < nums1[m]:

#                 nums1[:m], nums1[m+1:] = nums1[:m] + [nums2[n]], nums1[m:-1]

#                 n += 1
#                 m += 1
            
#             else:

#                 if all([x==0 for x in nums1[m:]]):
#                     nums1[m] = nums2[n]
#                     n += 1
                    
#                 m += 1


# merge(nums1,m,nums2,n)

# print(nums1)




'91. Decode Ways'  

# Input

# # Case 1:
# s = '12'
# # Output: 2

# # Case 2:
# s = '226'
# # Output: 3

# # Case 3:
# s = '06'
# # Output: 0

# # Custom Case:
# s = '112342126815'
# # Output: 11



# My apporach

# def fib(n):

#     res = [1,1]

#     for _ in range(n-1):
#         res.append(res[-2] + res[-1])
          
#     return res[1:]


# def numDecodings(s:str) -> int:

#     if s[0] == '0':
#         return 0
    
#     if len(s) == 1:
#         return 1

#     substrings = []
#     subs = ''

#     if s[0] in ['1', '2']:
#         subs += s[0]

#     for i in range(1, len(s)+1):

#         if i == len(s):
#             if subs != '':
#                 substrings.append(subs)

#         elif (s[i] in ['1', '2']) or (s[i-1] in ['1', '2'] and s[i] <= '6'):
#             subs += s[i]

#         else:
#             substrings.append(subs)
#             subs = ''

#     cap = len(max(substrings, key=len))
#     possibilities = fib(cap)

#     res = 0

#     for i in substrings:

#         if i in '10' or '20':
#             res += 1

#         else:
#             res += possibilities[len(i)-1] 
    
#     return res


# print(numDecodings(s))


'''
Notes: 
    This solution met 48% of expected results, there are a couple of cases I left unanalyzed.
    Nevertheless, the logic of fibonaccying the parsing numbers works, perhaps with more time
    a solution through this approach could work.

'''

# Dynamic Programming Approach

# def numDecodings(self, s):
    
#     dp = {len(s):1}

#     def backtrack(i):

#         if i in dp:
#             return dp[i]

#         if s[i]=='0':
#             return 0

#         if i==len(s)-1:
#             return 1

#         res = backtrack(i+1)

#         if int(s[i:i+2])<27:
#             res+=backtrack(i+2)
            
#         dp[i]=res

#         return res

#     return backtrack(0)




'98. Validate Binary Search Tree' 

# Base

# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right



# Input

# # Case 1
# root_layout = [2,1,3]
# root = TreeNode(val=2, left=TreeNode(val=1), right=TreeNode(val=3))
# # Output: True


# # Case 2
# root_layout  = [5,1,4,None, None, 3, 6]
# left = TreeNode(val=1)
# right = TreeNode(val=4, left=TreeNode(val=3), right=TreeNode(val=6)) 
# root = TreeNode(val=5, left=left, right=right)
# # Output: False


# # Custom Case 1
# root_layout  = [4,2,5,1,8,5,9,3,10,2,15]

# root = TreeNode(val=4)
# first_left, first_right = TreeNode(val=2), TreeNode(val=5)

# fl_left = TreeNode(val=1)
# fl_right = TreeNode(val=8, left=TreeNode(val=5), right=TreeNode(val=9)) 
# fr_left = TreeNode(val=3)
# fr_right = TreeNode(val=10, left=TreeNode(val=2), right=TreeNode(val=15)) 

# first_left.left, first_left.right = fl_left, fl_right
# first_right.left, first_right.right = fr_left, fr_right

# root.left, root.right = first_left, first_right
# # Output: True


# # Custom Case 2
# root_layout  = [10,9,11,3,4,7,15,8,4,13,16,12,21]

# root = TreeNode(val=10)
# first_left, first_right = TreeNode(val=9), TreeNode(val=11)

# fl_left = TreeNode(val=3, left=TreeNode(val=4), right=TreeNode(val=7))
# fl_right = TreeNode(val=15)
# fr_left = TreeNode(val=8, left=TreeNode(val=4), right=TreeNode(val=13))
# fr_right = TreeNode(val=16, left=TreeNode(val=12), right=TreeNode(val=21)) 

# first_left.left, first_left.right = fl_left, fl_right
# first_right.left, first_right.right = fr_left, fr_right

# root.left, root.right = first_left, first_right
# # Output: False


# # Custom Case 3
# root_layout  = [2,2,2]
# root = TreeNode(val=2, left=TreeNode(val=2), right=TreeNode(val=2))
# # Output: False


# My approach

'''
Intuition:
    traverse with DFS and check each (root-child) group,
    if balanced, check the next group, else, return False.

    if we get to the end of the tree and there were no imbalance, return True.

'''

# def dfs(root:TreeNode):

#     stack = [root]

#     while stack:

#         node = stack.pop()
#         ndv = node.val

#         if node.left or node.right:

#             if node.left:

#                 ndlv = node.left.val
                
#                 if node.left.val > node.val:
#                    return False
                
#                 stack.append(node.left)
            

#             if node.right:

#                 ndrv = node.right.val
                                                
#                 if node.right.val < node.val:
#                    return False
                
#                 stack.append(node.right)

#             if node.val == node.right.val and node.val == node.left.val:
#                 return False
            
#     return True


# print(dfs(root))

'Note: My solution works up to 78% of the cases'


# Inorder Tree Traversal Approach

# path = []

# def inorder(root:TreeNode, route:list):

#     if root is None:
#         return
    
#     inorder(root.left, route)
#     route.append(root.val)
#     inorder(root.right, route)


# inorder(root=root, route=path)


# print(path)

'Note: The Trick here is that the inorder traversal, basically returns a sorted list if is balanced!'




'101. Symmetric Tree' 

# Base

# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# Input

# # Case 1
# root_layout = [1,2,2,3,4,4,3]

# root = TreeNode(val=1)
# first_left= TreeNode(val=2, left=TreeNode(val=3), right=TreeNode(val=4))
# first_right = TreeNode(val=2, left=TreeNode(val=4), right=TreeNode(val=3))

# root.left, root.right = first_left, first_right
# # Output: True

# # Case 2
# root_layout = [1,2,2,None,3,None,3]

# root = TreeNode(val=1)
# first_left= TreeNode(val=2, right=TreeNode(val=3))
# first_right = TreeNode(val=2, right=TreeNode(val=3))

# root.left, root.right = first_left, first_right
# # Output: False

# # Custom Case 1
# root_layout = [1,2,2,2,None,2]

# root = TreeNode(val=1)
# first_left= TreeNode(val=2, left=TreeNode(val=2))
# first_right = TreeNode(val=2, left=TreeNode(val=2))

# root.left, root.right = first_left, first_right
# # Output: False



# My approach

'''
Intuition:
    Return a inorder-traversal list of the trees from the first left and right node,
    and one should be the reverse of the other.

    Handling corner cases:
    - If only a root: True
    - If only a root with two leaves, if the leaves are equal: True
    - If the number of nodes is even: False
'''

# def isSymetric(root:TreeNode):

#     tree_nodes = []

#     def inorder(root):

#         if root == None:
#             return 
        
#         inorder(root.left)
#         tree_nodes.append(root.val)
#         inorder(root.right)

#     inorder(root=root)

    
#     if len(tree_nodes) == 1:
#         return True
    
#     # If there are an even number of nodes, it can be symetrical
#     if len(tree_nodes)%2 == 0:
#         return False   
    
#     if len(tree_nodes) == 3:
#         if root.left.val == root.right.val:
#             return True

#     mid = len(tree_nodes)//2 
#     left_tree = tree_nodes[:mid]
#     right_tree = tree_nodes[mid+1:]
    
#     return left_tree == list(reversed(right_tree))


# print(isSymetric(root))

'Note: This solution works for cases where all node are identical, since it didnt distinguish between left and right'



# Recursive Approach

# def is_mirror(self, n1, n2):

#     if n1 is None and n2 is None:
#         return True
    
#     if (n1 is None) or (n2 is None) or (n1.val != n2.val):
#         return False

#     return self.is_mirror(n1.left, n2.right) and self.is_mirror(n1.right, n2.left)


# def isSymmetric(self, root):

#     return self.is_mirror(n1=root.left, n2=root.right)

# 'This solution works perfectly'




'102. Binary Tree Level Order Traversal' 

# Input

# Base

# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# Input

# # Case 1
# root_layout = [3,9,20,None,None,15,7]

# root = TreeNode(val=3)
# first_left= TreeNode(val=9)
# first_right = TreeNode(val=2, left=TreeNode(val=15), right=TreeNode(val=7))

# root.left, root.right = first_left, first_right
# # Output: [[3],[9,20],[15,7]]

# # Case 2
# root_layout = [1]
# root = TreeNode(val=1)
# # Output: [[1]]

# # Case 3
# root_layout = []
# # Output: []


# My Approach

'''
Intuition:

    With bread-first search, I can pull the values in order by levels.

    Given that Binary tree are binary, with the powers of 2
    it could be calculated how many nodes exist in each level.

    and with the l = 1 + floor(log_2(n)), the number of levels can
    be known just having the number of nodes.

    
'''
# from collections import deque
# from math import floor, log2

# def bfs(root:TreeNode):

#     queue = deque()
#     queue.append(root)

#     path = []

#     while queue:

#         node = queue.popleft()

#         if node not in path:

#             path.append(node)

#             if node.left:
#                 queue.append(node.left)

#             if node.right:
#                 queue.append(node.right)

#     return [x.val for x in path]

# nodes_list = bfs(root=root)

# n_levels = 1 + floor(log2(len(nodes_list)))

# result = []

# for i in range(n_levels):

#     temp = []

#     for j in range(pow(2, i)):

#         if nodes_list:
#             temp.append(nodes_list.pop(0))
    
#     result.append(temp)
    

# print(result)


'Notes: This solution works but the leetcode interpreter didnt recognized the log2 function'


# A Simplier Approach

# def levelsOrder(root:TreeNode):

#     from collections import deque
    
#     queue = deque()
#     queue.append(root)    
#     result = []

#     while queue:

#         queue_len = len(queue)
#         level = [] 
        
#         for i in range(queue_len):

#             node = queue.popleft()

#             if node is not None:

#                 level.append(node.val)
#                 queue.append(node.left)
#                 queue.append(node.right)

#         if level:   
#             result.append(level)

#     return result

# print(levelsOrder(root=root))

'Done'




'103. Binary Tree Zigzag Level Order Traversal' 

# Base

# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# Input

# # Case 1
# root_layout = [3,9,20,None,None,15,7]

# root = TreeNode(val=3)
# first_left= TreeNode(val=9)
# first_right = TreeNode(val=20, left=TreeNode(val=15), right=TreeNode(val=7))

# root.left, root.right = first_left, first_right
# # Output: [[3],[20,9],[15,7]]

# # Case 2
# root_layout = [1]
# root = TreeNode(val=1)
# # Output: [[1]]

# # Case 3
# root_layout = []
# # Output: []


# My Approach

'''
Notes:
    This will go apparently the same as the level order, but in the other way arround
    and this time is alternating depending of the level
'''


# def zigzagLevelOrder(root:TreeNode) -> list[list[int]]:

#     from collections import deque

#     queue = deque()
#     queue.append(root)
#     result = []
#     level = 1

#     while queue:

#         len_q = len(queue)
#         level_nodes = []
      
#         for i in range(len_q):

#             node = queue.popleft()

#             if node is not None:

#                 queue.append(node.left)
#                 queue.append(node.right)
#                 level_nodes.append(node.val)

#         if len(level_nodes) != 0:

#             if level % 2 == 0:
#                 level_nodes = list(reversed(level_nodes))
            
#             result.append(level_nodes)
        
#         level += 1
    
#     return result

# print(zigzagLevelOrder(root=root))

'It worked!'




'104. Maximum Depth of Binary Tree'

# Base

# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# Input

# # Case 1
# root_layout = [3,9,20,None,None,15,7]

# root = TreeNode(val=3)
# first_left= TreeNode(val=9)
# first_right = TreeNode(val=20, left=TreeNode(val=15), right=TreeNode(val=7))

# root.left, root.right = first_left, first_right
# # Output: 3

# # Case 2
# root_layout = [1, None, 2]

# root = TreeNode(val=1, right=TreeNode(val=2))
# # Output: 2


# My approach

'''
Notes:
    Here could be to ways (or more) to solve it:
        1. Implement the BFS by level listing (like the challenges prior to this one) and count the elements of the result
        2. Simply list through DFS or BFS and apply l = 1 + floor(log_2(n)), to know the number of levels, but probably leetcode won't have 
           the log2 function in its math module, so I'll the first way.
'''

# def maxDepth(root:TreeNode) -> int:

#     from collections import deque

#     queue = deque()
#     queue.append(root)
#     result = []

#     while queue:

#         queue_len = len(queue)
#         level = []

#         for _ in range(queue_len):

#             node = queue.popleft()

#             if node is not None:

#                 queue.append(node.left)
#                 queue.append(node.right)

#                 level.append(node.val)

#         if level:
#             result.append(level)
    
#     return result

# print(maxDepth(root))

'Done!'




'105. Construct Binary Tree from Preorder and Inorder Traversal'

# # Base

# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# # Input

# # Case 1
# preorder, inorder = [3,9,20,15,7],[9,3,15,20,7]
# # Output: [3,9,20,None,None,15,7]


# def buildTree(preorder, inorder):

#     if inorder:

#         idx = inorder.index(preorder.pop(0))
#         root = TreeNode(val = inorder[idx])
#         root.left = buildTree(preorder=preorder, inorder=inorder[:idx])
#         root.right = buildTree(preorder=preorder, inorder=inorder[idx+1:])

#         return root


'Done'




'108. Convert Sorted Array to Binary Search Tree'

# # Base

# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# Input

# # Case 1
# nums = [-10,-3,0,5,9]
# # Output: [0,-3,9,-10,None,5] | [0,-10,5,None,-3,None,9]

# # Case 2
# nums = [1,3]
# # Output: [3,1] | [1,None,-3]


# My Approach

'''
Intuition:
    Learnt for the prior exercise, the middle node will be taken as the root.
    from there, it can recursively built the solution.
        base case = when len(nums) = 0
'''

# def sortedArrayToBST(nums:list[int]) -> TreeNode:

#     nums_len = len(nums)

#     if nums_len:

#         idx = nums_len // 2

#         return TreeNode(val = nums[idx], left = sortedArrayToBST(nums=nums[:idx]), right = sortedArrayToBST(nums=nums[idx+1:]))



# node = sortedArrayToBST(nums=nums)
# print(node)

'Done'




'116. Populating Next Right Pointers in Each Node'

# Base

# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None, next=None):
#         self.val = val
#         self.left = left
#         self.right = right
#         self.next = next


# Input

# # Case 1
# tree_lauout = [1,2,3,4,5,6,7]

# left = TreeNode(val=2, left=TreeNode(val=4), right=TreeNode(val=5))
# right = TreeNode(val=3, left=TreeNode(val=6), right=TreeNode(val=7))
# root = TreeNode(val=1, left=left, right=right)
# # Output: [1,#,2,3,#,4,5,6,7,#]


#My Approach

'''
Intuition:
    This could be solved with the BFS modified to catch nodes by level,
    and with the level picked from each loop, modify its pointers in that order 
'''

# def connect(root:TreeNode) -> TreeNode:
    
#     #Start
#     queue = [root]
    
#     while queue:

#         q_len = len(queue)
#         level = []

#         for i in range(q_len):

#             node = queue.pop(0)

#             if node:

#                 queue.extend([node.left, node.right])
#                 level.append(node)
        
#         if level:

#             for i in range(len(level)):

#                 if i != len(level)-1:

#                     level[i].next = level[i+1]
    
#     return root

'Worked right way, YAY! :D'