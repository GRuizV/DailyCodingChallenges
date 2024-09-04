'''
CHALLENGES INDEX

*Arrays
1. Two Sum (Array) (Hash Table)
4. Median of Two Sorted Arrays (Array)
11. Container With Most Water (Array)
15. 3Sum (Array)
34. Find First and Last Position of Element in Sorted Array (Array)
36. Valid Sudoku (Array) (Hash Table) (Matrix)
42. Trapping Rain Water (Array)
46. Permutations (Array)
48. Rotate Image (Array) (Matrix)
49. Group Anagrams (Array) (Hash Table) (Sorting)
53. Maximum Subarray (Array) (DQ) (DP)
55. Jump Game (Array) (DP) (GRE)
75. Sort Colors (Array) (TP) (Sorting)
78. Subsets (Array) (BT)
88. Merge Sorted Array (Array) (TP) (Sorting)
118. Pascal's Triangle (Array) (DP)
121. Best Time to Buy and Sell Stock (Array) (DP)
122. Best Time to Buy and Sell Stock II (Array) (DP) (GRE)
128. Longest Consecutive Sequence (Array) (Hash Table)
134. Gas Station (Array) (GRE)
152. Maximum Product Subarray (Array) (DP)
162. Find Peak Element (Array)
179. Largest Number (Array) (Sorting) (GRE)
189. Rotate Array (Array) (TP)
198. House Robber (Array) (DP)
204. Count Primes (Array) (Others)
215. Kth Largest Element in an Array (Array) (Heap) (DQ) (Sorting)
238. Product of Array Except Self (PS)
239. Sliding Window Maximum (Array) (SW)
283. Move Zeroes (Array) (TP)
287. Find the Duplicate Number (FCD) (Array) (TP)
300. Longest Increasing Subsequence (Array) (DP)
315. Count of Smaller Numbers After Self - Partially solved (Array) (DQ)
334. Increasing Triplet Subsequence (Array) (GRE)
347. Top K Frequent Elements (Array) (Heaps) (Sorting)
350. Intersection of Two Arrays II (Array) (TP)
378. Kth Smallest Element in a Sorted Matrix (Matrix) (Heaps)
384. Shuffle an Array (Array) (Others)
454. 4Sum II (Arrays) (Others)


*Hash Tables
3. Longest Substring Without Repeating Characters (Hash Table) (SW)
13. Roman to Integer (Hash Table)
17. Letter Combinations of a Phone Number (Hash Table) (BT)
73. Set Matrix Zeroes (Matrix) (Hash Table)
76. Minimum Window Substring (Hash Table) (SW)
127. Word Ladder (Hast Table) (BFS)
138. Copy List with Random Pointer (Hash Table) (LL)
146. LRU Cache (Hash Table)
166. Fraction to Recurring Decimal (Hash Table) (Others)
202. Happy Number (Hash Table) (TP) (Others)
208. Implement Trie (Hast Table) (Tree)
380. Insert Delete GetRandom O(1) (Hash Table) (Others)


*Matrices
54. Spiral Matrix (Matrix)  
79. Word Search (Matrix) (BT)
130. Surrounded Regions (Matrix) (BFS) (DFS)
200. Number of Islands (Matrix) (DFS)
212. Word Search II (Array) (DFS) (BT) (Matrix)
240. Search a 2D Matrix II (Matrix) (DQ) (BS)
289. Game of Life (Matrix)


*Others
7. Reverse Integer (Others)
8. String to Integer (atoi) (Others)
14. Longest Common Prefix (Others)
29. Divide Two Integers (Others)
38. Count and Say (Others)
66. Plus One (Others)
69. Sqrt(x) (Others)
149. Max Points on a Line (Others)
171. Excel Sheet Column Number (Others)
172. Factorial Trailing Zeroes (Others)
326. Power of Three (RC) (Others)




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


(XX)
'''




'### ARRAYS ###'

'1. Two Sum'
# def x():
#     # 1st approach
#     nums = [3,2,4]
#     target = 6
#     result = []

#     for i in range(len(nums)):

#         for j in range(i+1, len(nums)):

#             if nums[i] + nums[j] == target:
#                 result = [i, j]
#                 break
            
#         if result:
#             break
        
#     print(result)


#     # 2nd approach
#     nums = [3,2,4]
#     target = 6
#     hashmap = {v:k for k,v in enumerate(nums)}
#     result = []

#     for i in range(len(nums)):

#         comp = target - nums[i]

#         if comp in hashmap and i != hashmap[comp]:
#             result = [i, hashmap[comp]]
#             break
    
    
#     print(result)


#     # 3rd approach
#     nums = [3,2,4]
#     target = 6
#     hashmap = {}
#     result = []

#     for i in range(len(nums)):

#         comp = target - nums[i]

#         if comp in hashmap:
#             result = [i, hashmap[comp]]
#             break

#         hashmap[nums[i]] = i
        
#     print(result)

'4. Median of Two Sorted Arrays'
# def x():

#     # Input
#     nums1 = [1,3]
#     nums2 = [2]


#     # My Approach
#     nums_total = sorted(nums1 + nums2)

#     if len(nums_total) % 2 != 0:
        
#         median_idx = len(nums_total) // 2 
#         median = float(nums_total[median_idx])
#         print(f'{median:.5f}')

#     else:
#         median_idx = len(nums_total) // 2 
#         print(nums_total[median_idx-1], nums_total[median_idx] )
#         median = (nums_total[median_idx-1] + nums_total[median_idx]) / 2
#         print(f'{median:.5f}')


#     '''
#     Note: My solution actually worked, but don't know why is not working in LeetCode.
#             Apparently, they want a solution with mergesort.
#     '''


#     # Mergesort Solution with pointers
#     def findMedianSortedArrays(nums1, nums2) -> float:
            
#             m, n = len(nums1), len(nums2)
#             p1, p2 = 0, 0
            
#             # Get the smaller value between nums1[p1] and nums2[p2].
#             def get_min():
#                 nonlocal p1, p2
#                 if p1 < m and p2 < n:
#                     if nums1[p1] < nums2[p2]:
#                         ans = nums1[p1]
#                         p1 += 1
#                     else:
#                         ans = nums2[p2]
#                         p2 += 1
#                 elif p2 == n:
#                     ans = nums1[p1]
#                     p1 += 1
#                 else:
#                     ans = nums2[p2]
#                     p2 += 1
#                 return ans
            
#             if (m + n) % 2 == 0:
#                 for _ in range((m + n) // 2 - 1):
#                     _ = get_min()
#                 return (get_min() + get_min()) / 2
#             else:
#                 for _ in range((m + n) // 2):
#                     _ = get_min()
#                 return get_min()
            
#     print(findMedianSortedArrays(nums1, nums2))

#     '''
#     Note: Honestly I feel this is kind of ego lifting.
#     '''

'11. Container With Most Water'
# def x():
#     # Input
#     heights = [1,8,6,2,5,4,8,3,7]


#     # My Approach
#     max_area = 0

#     for i in range(len(heights)):

#         for j in range(i+1, len(heights)):

#             height = min(heights[i], heights[j])
#             width = j-i
#             area = height * width

#             max_area = max(max_area, area)

#     print(max_area)


#     '''
#     Note:
#         While this approach works, its complexity goes up to O(n), and is required to be more efficient
#     '''


#     # Two-pointer solution

#     left = 0
#     right = len(heights)-1
#     max_area = 0

#     while left < right:

#         h = min(heights[left], heights[right])
#         width = right - left
#         area = h * width

#         max_area = max(max_area, area)

#         if heights[left] <= heights [right]:
#             left += 1
        
#         else:
#             right -= 1


#     print(max_area)

'15. 3Sum'
# def x():
#     import itertools

#     # Input
#     nums = [0,0,0]


#     # My approach

#     '''
#     Rationale:
        
#         1) Build all combinations caring for the order.
#         2) Filter down those who met sum(subset) = 0
#         3) Make sure there is no duplicates & return.

#     '''
#     comb = list(itertools.combinations(nums,3))

#     comb = [sorted(x) for x in comb if sum(x) == 0]

#     res = []

#     for i in comb:

#         if i not in res:
#             res.append(i)

#     print(res)

#     '''
#     Notes:

#         This solution actually works, but breaks when a big enough input is passed.
#     '''

#     # Two-Pointers approach solution
#     def threeSum(self, nums):
            
#             nums.sort()
#             answer = []
            
#             # if the inputs have less than 3 items
#             if len(nums) < 3:
#                 return answer
            
#             for i in range(len(nums)):

#                 # Since is a sorted input, if first element is positive, there is no way it'll sum up to 0
#                 if nums[i] > 0:
#                     break
                
#                 # Apart from the first element, if the following is the same, jump to the next iteration to avoid returning duplicates
#                 if i > 0 and nums[i] == nums[i - 1]:
#                     continue
                
#                 # Pointers setting    
#                 low, high = i + 1, len(nums) - 1

#                 while low < high:

#                     s = nums[i] + nums[low] + nums[high]

#                     if s > 0:
#                         high -= 1

#                     elif s < 0:
#                         low += 1

#                     else:

#                         answer.append([nums[i], nums[low], nums[high]])
#                         lastLowOccurrence, lastHighOccurrence = nums[low], nums[high]
                        
#                         while low < high and nums[low] == lastLowOccurrence:
#                             low += 1
                        
#                         while low < high and nums[high] == lastHighOccurrence:
#                             high -= 1
            
#             return answer

'34. Find First and Last Position of Element in Sorted Array'
# def x():

#     # Input

#     # case 1
#     nums = [5,7,7,8,8,10]
#     target = 8  # Expected Output: [3,4]

#     # case 2
#     nums = [5,7,7,8,8,10]
#     target = 6  # Expected Output: [-1,-1]

#     # case 3
#     nums = []
#     target = 0  # Expected Output: [-1,-1]


#     # My Approach
#     def searchRange(nums:list, target:int) -> int:
        
#         if target in nums:

#             starting_position = nums.index(target)

#             # The ending positions is calculated as of: (number of indices) - the relative position if the list is reversed
#             Ending_position = (len(nums)-1) - nums[::-1].index(target)

#             return [starting_position, Ending_position]

#         else:
#             return [-1,-1]

#     print(searchRange(nums, target))

#     'Notes: It worked!'

'36. Valid Sudoku'
# def x():

#     # Input

#     # Case 1
#     board = [
#     ["5","3",".",".","7",".",".",".","."]
#     ,["6",".",".","1","9","5",".",".","."]
#     ,[".","9","8",".",".",".",".","6","."]
#     ,["8",".",".",".","6",".",".",".","3"]
#     ,["4",".",".","8",".","3",".",".","1"]
#     ,["7",".",".",".","2",".",".",".","6"]
#     ,[".","6",".",".",".",".","2","8","."]
#     ,[".",".",".","4","1","9",".",".","5"]
#     ,[".",".",".",".","8",".",".","7","9"]
#     ]

#     # Case 2
#     board = [
#     ["8","3",".",".","7",".",".",".","."]
#     ,["6",".",".","1","9","5",".",".","."]
#     ,[".","9","8",".",".",".",".","6","."]
#     ,["8",".",".",".","6",".",".",".","3"]
#     ,["4",".",".","8",".","3",".",".","1"]
#     ,["7",".",".",".","2",".",".",".","6"]
#     ,[".","6",".",".",".",".","2","8","."]
#     ,[".",".",".","4","1","9",".",".","5"]
#     ,[".",".",".",".","8",".",".","7","9"]
#     ]


#     # My Approach
#     '''
#     Rationale:
#         1. Pull out all the columns, rows and sub-boxes to be evaluated.
#         2. Filter down empty colums, rows and sub-boxes.
#         3. Cast set on each element on the 3 groups and 
#             if one of them have less items than before the casting, return False. Otherwise, return True
#     '''

#     def isValidSudoku(board: list[list[str]]) -> bool:

#         rows = board
#         columns = [list(x) for x in zip(*board)]


#         # Bulding the sub-boxes directly into the list
#             # Did it this way to save time complexity.
                
#         sub_boxes = [
#             [board[0][0:3],board[1][0:3],board[2][0:3]],
#             [board[0][3:6],board[1][3:6],board[2][3:6]],
#             [board[0][6:9],board[1][6:9],board[2][6:9]],
#             [board[3][0:3],board[4][0:3],board[5][0:3]],
#             [board[3][3:6],board[4][3:6],board[5][3:6]],
#             [board[3][6:9],board[4][6:9],board[5][6:9]],
#             [board[6][0:3],board[7][0:3],board[8][0:3]],
#             [board[6][3:6],board[7][3:6],board[8][3:6]],
#             [board[6][6:9],board[7][6:9],board[8][6:9]],
#         ]


#         # Validating rows
#         for row in rows:

#             row_wo_dot = [num for num in row if num != '.']

#             if len(row_wo_dot) != len(set(row_wo_dot)):
#                 return False


#         # Validating columns
#         for col in columns:

#             col_wo_dot = [num for num in col if num != '.']

#             if len(col_wo_dot) != len(set(col_wo_dot)):
#                 return False


#         # Validating Sub-boxes
#         for subb in sub_boxes:

#             plain_subb = [num for li in subb for num in li if num != '.']

#             if len(plain_subb) != len(set(plain_subb)):
#                 return False


#         return True


#     print(isValidSudoku(board))

#     'Notes: It works perfectly, but could be less verbose'


#     # Another Approach
#     import collections

#     def isValidSudoku(self, board):

#         rows = collections.defaultdict(set)
#         cols = collections.defaultdict(set)
#         subsquares = collections.defaultdict(set)

#         for r in range(9):

#             for c in range(9):

#                 if(board[r][c] == "."):
#                     continue

#                 if board[r][c] in rows[r] or board[r][c] in cols[c] or board[r][c] in subsquares[(r//3, c//3)]:
#                     return False
                
#                 rows[r].add(board[r][c])
#                 cols[c].add(board[r][c])
#                 subsquares[(r//3,c//3)].add(board[r][c])

#         return True

#     '''
#     Notes: 
#         This solution was much more elegant. And essentially the difference lays in this solution could be more scalable 
#         since it builds the data holder while iterating.
#     '''

'42. Trapping Rain Water'
# def x():

#     # Input

#     # case 1
#     height = [0,1,0,2,1,0,1,3,2,1,2,1]  # Exp. Out: 6

#     # case 2
#     height = [4,2,0,3,2,5]  # Exp. Out: 9


#     'Solution'
#     def trap(height):

#         if not height:
#             return 0
        

#         left, right = 0, len(height)-1
#         left_max, right_max = 0, 0
#         result = 0

#         while left < right:

#             if height[left] < height[right]:

#                 if height[left] >= left_max:
#                     left_max = height[left]

#                 else:
#                     result += left_max - height[left]

#                 left += 1
            
#             else:

#                 if height[right] >= right_max:
#                     right_max = height[right]

#                 else:
#                     result += right_max - height[right]

#                 right -= 1
        
#         return result

#     # Testing
#     print(trap([3,0,2]))

#     'Done'

'46. Permutations'
# def x():

#     # Input

#     # Case 1
#     nums = [1,2,3] # Exp. Out: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

#     # Case 2
#     nums = [0,1] # Exp. Out: [[0,1],[1,0]]

#     # Case 3
#     nums = [1] # Exp. Out: [[1]]


#     # Solution
#     def permute(nums: list[int]) -> list[list[int]]:
        
#         if len(nums) == 0:
#             return []
        
#         if len(nums) == 1:
#             return [nums]
        
#         l = []

#         for i in range(len(nums)):

#             num = nums[i]
#             rest = nums[:i] + nums[i+1:]

#             for p in permute(rest):
#                 l.append([num] + p)
            
#         return l

#     'Done'

'48. Rotate Image'
# def x():

#     # Input

#     # Case 1
#     matrix = [
#         [1,2,3],
#         [4,5,6],
#         [7,8,9]
#         ]
#     # Exp. Out: [[7,4,1],[8,5,2],[9,6,3]]

#     # Case 2
#     matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
#     # Exp. Out: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]


#     'My approach'
#     def rotate(matrix: list[list[int]]):

#         n = len(matrix)

#         for i in range(n):

#             rot_row = []

#             for j in range(n):  # Since is given that is an squared matrix
#                 rot_row.insert(0, matrix[j][i])

#             matrix.append(rot_row)
    
#         for i in range(n):
#             matrix.pop(0)

#     # Testing
#     rotate(matrix)
#     print(matrix)

#     'Notes: It worked, but seems a little unorthodox'


#     'Another Approach'
#     def rotate(matrix):

#         # reverse
#         l = 0
#         r = len(matrix) -1

#         while l < r:
#             matrix[l], matrix[r] = matrix[r], matrix[l]
#             l += 1
#             r -= 1

#         x = ''
#         matrix = matrix[::-1]
#         x=0

#         # transpose 
#         for i in range(len(matrix)):
#             for j in range(i):
#                 matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


#     # Testing
#     rotate(matrix)
#     print(matrix)

#     'Notes: This one looks much more like a canon type solution'

'49. Group Anagrams'
# def x():

#     # Input

#     # Case 1
#     strs = ["eat","tea","tan","ate","nat","bat"]
#     #Exp. Out: [["bat"],["nat","tan"],["ate","eat","tea"]]

#     # Case 2
#     strs = [""]
#     #Exp. Out: [[""]]

#     # Case 3
#     strs = ["a"]
#     # Exp. Out: [["a"]]

#     # Custom Case
#     strs = ["ddddddddddg","dgggggggggg"]
#     # Expected: [["dgggggggggg"],["ddddddddddg"]]



#     'My Approach'

#     '''
#     Intuition:
#         1. Take the first element of the input and make a list with all element that contains the same characters
#         2. Erase the taken elements from the input.
#         3. Reiterate steps 1 & 2 until the input is exhausted

#     '''

#     def groupAnagrams(strs:list):
        
#         if len(strs) == 1:
#             return[strs]

#         # Auxiliary anagram checker
#         def is_anagram(ref:list, string:list):

#             if len(ref) != len(string):
#                 return False

#             for char in ref:
                
#                 if ref.count(char) != string.count(char):   
#                     return False

#             return True
        
#         # Creating Flag to manage repetitions
#         strs = [[word, False] for word in strs]


#         result = []

#         for word in strs:
                
#             if word[1] == False:

#                 anagrams = []
#                 anagrams.append(word[0])            
#                 word[1] = True

#                 for rest in strs:

#                     if rest[1] == False:

#                         if is_anagram(word[0], rest[0]):
#                             anagrams.append(rest[0])
#                             rest[1] = True
            
#                 result.append(anagrams)

#         return result
    
#     # Testing
#     print(groupAnagrams(strs))

#     '''
#     Notes: 
#         It passed 72/126 cases, the case below broke the code: 
#             strs = ["ddddddddddg","dgggggggggg"] / Output: [["ddddddddddg","dgggggggggg"]], Expected: [["dgggggggggg"],["ddddddddddg"]]

#         After the fixture, it works but beat no one in efficiency
#     '''


#     'Another Approach'
#     def groupAnagrams(strs):
        
#         freq = {}

#         for word in strs:

#             newWord = ''.join(sorted(word))

#             if newWord not in freq:
#                 freq[newWord] = []
            
#             freq[newWord].append(word)

#         return list(freq.values())

#     # Testing
#     print(groupAnagrams(strs))

#     '''
#     Notes: Absolutely more elegant solution
#     '''

'53. Maximum Subarray'
# def x():

#     # Input

#     # Case 1
#     nums = [-2,1,-3,4,-1,2,1,-5,4]
#     # Output: 6 / [4,-1,2,1]

#     # Case 2
#     nums = [1]
#     # Output: 1 / [1]

#     # Case 3
#     nums = [5,4,-1,7,8]
#     # Output: 23 / [5,4,-1,7,8]
    
#     '''
#     My Approach

#     Intuition: Brute forcing
#         1. Store the sum of the max array
#         2. From the max len down to len = 1, evaluate with a sliding window each sub array.
#         3. Return the max sum
#     '''
#     def maxSubArray(nums):

#         max_sum = sum(nums)

#         # Handling the special case len = 1
#         if len(nums) == 1:
#             return max_sum
        
        
#         max_element = max(nums)
#         nums_len = len(nums)-1
#         idx = 0

#         while nums_len > 1:

#             if idx + nums_len > len(nums):
#                 nums_len -= 1
#                 idx = 0
#                 continue
                
#             sub_array = nums[idx:idx+nums_len]
#             sub_array_sum = sum(sub_array)

#             if sub_array_sum > max_sum:
#                 max_sum = sub_array_sum
                
#             idx += 1

#         # Handling the case where one element is greater than any subarray sum
#         if max_element > max_sum:
#             return max_element
        
#         return max_sum


#     print(maxSubArray(nums))
    
#     'Notes: My solution worked 94,7% of the cases, the time limit was reached.'


#     '''Kadane's Algorythm (for max subarray sum)'''

#     def maxSubArray(nums):

#         max_end_here = max_so_far = nums[0]

#         for num in nums[1:]:

#             max_end_here = max(num, max_end_here + num)
#             max_so_far = max(max_so_far, max_end_here)
        
#         return max_so_far


#     print(maxSubArray(nums))

#     'Notes: Apparently it was a classic problem'

'55. Jump Game'
# def x():

#     # Input
#     # Case 1
#     nums = [2,3,1,1,4]
#     # Output: True

#     # Case 2
#     nums = [3,2,1,0,4]
#     # Output: False

#     # Custom Case
#     nums = [2,0,0]
#     # Output: 

#     '''
#     My Approach

#         Intuition - Brute Force:

#             - I will check item by item to determine if the end of the list is reachable

#     '''
#     def canJump(nums:list[int]) -> bool:

#         # Corner case: nums.lenght = 1 / nums[0] = 0
#         if len(nums) == 1 and nums[0] == 0:
#             return True
        
#         idx = 0

#         while True:

#             idx += nums[idx]

#             if idx >= len(nums)-1:
#                 return True
            
#             if nums[idx] == 0 and idx < len(nums)-1:
#                 return False
            
#     # Testing
#     print(canJump(nums))

#     'Notes: This solution suffice 91,2% of the case'


#     'Backtrack Approach'
#     def canJump(nums: list[int]) -> bool:

#         if len(nums)==1:
#             return True  

#         #Start at num[-2] since nums[-1] is given
#         backtrack_index = len(nums)-2 
#         #At nums[-2] we only need to jump 1 to get to nums[-1]
#         jump =1  

#         while backtrack_index>0:
#             #We can get to the nearest lily pad
#             if nums[backtrack_index]>=jump: 
#                 #now we have a new nearest lily pad
#                 jump=1 
#             else:
#                 #Else the jump is one bigger than before
#                 jump+=1 
#             backtrack_index-=1
        
#         #Now that we know the nearest jump to nums[0], we can finish
#         if jump <=nums[0]: 
#             return True
#         else:
#             return False 

#     'Notes: Right now I am not that interested in learning bactktracking, that will be for later'

'75. Sort Colors'
# def x():

#     # Input
#     # Case 1
#     nums = [2,0,2,1,1,0]
#     # Output: [0,0,1,1,2,2]

#     # Case 2
#     nums = [2,0,1]
#     # Output: [0,1,2]

#     '''
#     My approach

#         Intuition:
#             Since the solution requires the sorting be in place, 
#             perhaps Bubblesort would do the trick.
#     '''

#     def sortColors(nums:list[int]) -> list[int]:

#         swapped = True       

#         while swapped != False:

#             swapped = False
#             i = 0

#             while True:

#                 if i == len(nums)-1:
#                     break

#                 if nums[i] > nums[i+1]:
#                     nums[i], nums[i+1] = nums[i+1], nums[i]
#                     swapped = True

#                 i += 1

#     # Testing
#     sortColors(nums)
#     print(nums)

#     'Notes: Done!'

'78. Subsets'
# def x():

#     # Input
#     # Case 1
#     nums = [1,2,3]
#     # Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

#     # Case 2
#     nums = [0]
#     # Output: [[],[0]]

#     '''
#     My Approach

#         Intuition:
#             - Perhaps with itertool something might be done
#     '''

#     def subsets(nums:list[int]) -> list[list[int]]:

#         from itertools import combinations

#         result = []

#         for i in range(len(nums)+1):

#             result.extend(list(map(list, combinations(nums,i))))

#         return result

#     # Testing
#     print(subsets(nums=nums))

#     'Notes: It actually worked'


#     'Another more algorithmic Approach'
#     def subsets(nums: list[int]) -> list[list[int]]:

#         arr = [[]]

#         for j in nums:

#             temp = []

#             for i in arr: 

#                 temp.append(i+[j])
            
#             arr.extend(temp)
        
#         return arr 

#     # Testing
#     print(subsets(nums))

#     'Notes: The guy who came up with this is genius'

'88. Merge Sorted Array'
# def x():

#     # Input
#     # Case 1
#     nums1 = [1,2,3,0,0,0]
#     m = 3
#     nums2 = [2,5,6]
#     n = 3
#     # Output: [1,2,2,3,5,6]

#     # Case 2
#     nums1 = [1]
#     m = 1
#     nums2 = []
#     n = 0
#     # Output: [1]

#     # Case 3
#     nums1 = [0]
#     m = 0
#     nums2 = [1]
#     n = 1
#     # Output: [1]

#     # Custom case
#     nums1 = [0,2,0,0,0,0,0]
#     m = 2
#     nums2 = [-1,-1,2,5,6]
#     n = 5
#     # Output: [1]

#     # Custom case
#     nums1 = [-1,1,0,0,0,0,0,0]
#     m = 2
#     nums2 = [-1,0,1,1,2,3]
#     n = 6
#     # Output: [1]


#     'Solution'
#     def merge(nums1, m, nums2, n):

#         if m == 0:
#             for i in range(n):
#                 nums1[i] = nums2[i]

#         elif n != 0:

#             m = n = 0

#             while n < len(nums2):

#                 if nums2[n] < nums1[m]:

#                     nums1[:m], nums1[m+1:] = nums1[:m] + [nums2[n]], nums1[m:-1]

#                     n += 1
#                     m += 1
                
#                 else:

#                     if all([x==0 for x in nums1[m:]]):
#                         nums1[m] = nums2[n]
#                         n += 1
                        
#                     m += 1

#     # Testing
#     merge(nums1,m,nums2,n)
#     print(nums1)

'''118. Pascal's Triangle'''
# def x():

#     # Input
#     # Case 1
#     numRows = 5
#     #Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]

#     # Case 2
#     numRows = 1
#     #Output: [[1]]
      

#     '''
#     My Approach

#         Intuition:
#             initialize a preset solution to [[1],[1,1]] and according to the
#             parameter passed in the function, start to sum and populate this sums to a list
#             like [1]+[resulting_sums]+[1] and return that back to the preset solution, to operate over that
#             new element,

#                 The number of loops will be numRows - 2 (given the 2 initial elements)
#     '''

#     def generate(numRows:int) -> list[list[int]]:

#         result = [[1],[1,1]]

#         if numRows == 1:
#             return [result[0]]
        
#         if numRows == 2:
#             return result
        

#         for i in range(1, numRows-1):

#             new_element = []

#             for j in range(i):
#                 new_element.append(result[-1][j]+result[-1][j+1])

#             if new_element:
#                 result.append([1]+new_element+[1])

#         return result

#     # Testing
#     print(generate(numRows=5))

#     'It worked!'

'''121. Best Time to Buy and Sell Stock'''
# def x():

#     # Input
#     #Case 1
#     prices = [7,1,5,3,6,4]
#     #Output: 5

#     #Case 2
#     prices = [7,6,4,3,1]
#     #Output: 0


#     '''
#     My approach
#         Intuition
#             - Corner Case: if is a ascendingly sorted list, return 0.
            
#             - Pick the first item and set the profit as the max between the current profit and the difference between the first element
#             the max value from that item forward.
            
#             Do this in a while loop until len(prices) = 1.
#     '''

#     def maxProfit(prices: list[int]) -> int:

#         profit = 0

#         if prices == sorted(prices, reverse=True):
#             return profit        

#         while len(prices) > 1:

#             purchase = prices.pop(0)
#             profit = max(profit, max(prices)-purchase)
        
#         return profit

#     # Testing
#     print(maxProfit(prices=prices))

#     'This approach met 94% of the results'


#     '''Kadane's Algorithm'''
#     def maxProfit(prices: list[int]) -> int:

#         buy = prices[0]
#         profit = 0

#         for num in prices[1:]:

#             if num < buy:
#                 buy = num
            
#             elif num-buy > profit:
#                 profit = num - buy
        
        
#         return profit

#     # Testing
#     print(maxProfit(prices=prices))

#     'Done'

'''122. Best Time to Buy and Sell Stock II'''
# def x():

#     #Input
#     #Case 1
#     prices = [7,1,5,3,6,4]
#     #Output: 7

#     #Case 2
#     prices = [1,2,3,4,5]
#     #Output: 4

#     #Case 3
#     prices = [7,6,4,3,1]
#     #Output: 0

#     #Custom Case
#     prices = [3,3,5,0,0,3,1,4]
#     #Output: 0


#     'My approach'
#     def maxProfit(prices:list[int]) -> int:

#         if prices == sorted(prices, reverse=True):
#             return 0
        
#         buy = prices[0]
#         buy2 = None
#         profit1 = 0
#         profit2 = 0
#         total_profit = 0

#         for i in range(1, len(prices)):

#             if prices[i] < buy:
#                 buy = prices[i]
            
#             elif prices[i] - buy >= profit1:            
#                 profit1 = prices[i] - buy
#                 buy2 = prices[i] 

#                 for j in range(i+1, len(prices)):

#                     if prices[j] < buy2:
#                         buy2 = prices[j]

#                     elif prices[j] - buy2 >= profit2:
#                         profit2 = prices[j] - buy2
#                         total_profit = max(total_profit, profit1 + profit2)
            
#             total_profit = max(total_profit, profit1)

#         return total_profit

#     # Testing
#     print(maxProfit(prices=prices))

#     'This solution went up to solve 83% of the cases, the gap was due to my lack of understanding of the problem'


#     '''Same Kadane's but modified'''
#     def maxProfit(prices:list[int]) -> int:

#         max = 0 
#         start = prices[0]
#         len1 = len(prices)

#         for i in range(0 , len1):

#             if start < prices[i]: 
#                 max += prices[i] - start

#             start = prices[i]

#         return max

#     # Testing
#     print(maxProfit(prices=prices))

#     'My mistake was to assume it can only be 2 purchases in the term, when it could be as many as it made sense'

'''128. Longest Consecutive Sequence'''
# def x():

#     #Input
#     #Case 1
#     nums = [100,4,200,1,3,2]
#     #Output: 4

#     #Case 2
#     nums = [0,3,7,2,5,8,4,6,0,1]
#     #Output: 9


#     'My approach'
#     def longestConsecutive(nums:list)->int:

#         if not nums:
#             return 0
    
#         nums.sort()

#         sequences = {}

#         for i in range(len(nums)):

#             curr_seqs = [x for elem in sequences.values() for x in elem]

#             if nums[i] not in curr_seqs:

#                 sequences[nums[i]] = [nums[i]]

#                 for j in range(i+1,len(nums)):
                    
#                     criteria = range( min(sequences[nums[i]])-1, max(sequences[nums[i]])+2)
#                     if nums[j] in criteria:
#                         sequences[nums[i]].append(nums[j])

#         result = max(sequences.values(), key=len)

#         return len(set(result))

#     # Testing
#     print(longestConsecutive(nums=nums))

#     'This solution went up to 83% of the cases'


#     'Another Approach'
#     def longestConsecutive (nums):

#         if not nums:
#             return 0
        
#         num_set = set(nums)

#         longest = 1

#         for num in nums:

#             count = 1

#             if num-1 not in num_set:

#                 x = num

#                 while x+1 in num_set:
                
#                     count+=1
#                     x+=1

#             longest = max(longest, count)

#         return longest

#     # Testing
#     print(longestConsecutive(nums=nums))

#     'Done'

'''134. Gas Station'''
# def x():

#     # Input
#     #Case 1
#     gas, cost = [1,2,3,4,5], [3,4,5,1,2]
#     #Output = 3

#     #Case 2
#     gas, cost = [2,3,4], [3,4,3]
#     #Output = -1

#     # #Custom Case 
#     gas, cost = [3,1,1], [1,2,2]
#     #Output = 0


#     '''
#     My Approach

#         Intuition:
#             - Handle the corner case where sum(gas) < sum(cos) / return -1
#             - Collect the possible starting point (Points where gas[i] >= cost[i])
#             - Iterate to each starting point (holding it in a placeholder) to check 
#                 if a route starting on that point completes the lap:
                
#                 - if it does: return that starting point
#                 - if it doesn't: jump to the next starting point

#             - If no lap is completed after the loop, return -1.
#     '''

#     def canCompleteCircuit(gas:list[int], cost:list[int]) -> int:
        
#         # Handle the corner case
#         if sum(gas) < sum(cost):
#             return -1
        
#         # Collect the potential starting stations
#         stations = [i for i in range(len(gas)) if gas[i] >= cost[i]]

#         # Checking routes starting from each collected station
#         for i in stations:

#             station = i
#             tank = gas[i]

#             while tank >= 0:
                
#                 # Travel to the next station
#                 tank = tank - cost[station] 

#                 # Check if we actually can get to the next station with current gas
#                 if tank < 0:
#                     break
                    
#                 # If we are at the end of the stations (clockwise)
#                 if station + 1 == len(gas):
#                     station = 0
                            
#                 else:
#                     station += 1
                            
#                 #If we success in making the lap
#                 if station == i:
#                     return i
            
#                 # Refill the tank
#                 tank = tank + gas[station]

#         # in case no successful loop happens, return -1
#         return -1

#     # Testing
#     print(canCompleteCircuit(gas=gas, cost=cost))

#     'Note: My solution met 85% of the test cases'


#     'Another approach'
#     def canCompleteCircuit(gas:list[int], cost:list[int]) -> int:
        
#         # Handle the corner case
#         if sum(gas) < sum(cost):
#             return -1
        
#         current_gas = 0
#         starting_index = 0

#         for i in range(len(gas)):

#             current_gas += gas[i] - cost[i]

#             if current_gas < 0:
#                 current_gas = 0
#                 starting_index = i + 1
                
#         return starting_index
    
#     # Testing
#     print(canCompleteCircuit(gas=gas, cost=cost))

#     'Note: This simplified version prooved to be more efficient'

'''152. Maximum Product Subarray'''
# def x():

#     # Input
#     # Case 1
#     input = [2,3,-2,4]
#     # Output: 6 / [2,3] has the largest product

#     # Case 2
#     input = [-2,0,-1]
#     # Output: 0 / all products are 0

#     # Custom Case
#     input = [-2,3,-4]
#     # Output: 0 / all products are 0


#     '''
#     My approach

#         Intuition

#             This is a variation of Kadane's Algorithm, and may be solve same way
#             as the original
#     '''

#     def maxProduct(nums:list[int]) -> int:

#         if len(nums) == 1:
#             return nums[0]

#         max_ends_here, max_so_far = nums[0]

#         for num in nums[1:]:
        
#             max_ends_here = max(num, max_ends_here * num)
#             max_so_far = max(max_so_far, max_ends_here)

#         return max_so_far

#     # Testing
#     print(maxProduct(nums=input))

#     '''
#     Original Kadane's modified to compute product solved 51% of the cases. 
#     But, apparently with capturing the min_so_far and having a buffer to hold the max_so_far to not interfere with the
#         min_so_far calculation, the problem is solved
#     '''

#     '''Another Kadane's Mod. Approach'''
#     def maxProduct(nums:list[int]) -> int:

#         if len(nums) == 1:
#             return nums[0]

#         max_so_far = min_so_far = result = nums[0]

#         for num in nums[1:]:
        
#             temp_max = max(num, max_so_far * num, min_so_far * num)
#             min_so_far = min(num, max_so_far * num, min_so_far * num)
#             max_so_far = temp_max

#             result = max(result, max_so_far)

#         return result

#     # Testing
#     print(maxProduct(nums=input))

#     'Done'

'''162. Find Peak Element'''
# def x():

#     # Input
#     # Case 1
#     nums = [1,2,3,1]
#     # Output: 2

#     # Case 2
#     nums = [1,2,1,3,5,6,4]
#     # Output: 1 | 5


#     'Solution'
#     def find_peak_element(nums:list[int]) -> int:

#         if len(nums) == 1 or nums[0] > nums[1]:
#             return 0
        
#         if nums[-1] > nums[-2]:
#             return len(nums)-1

#         left, right = 0, len(nums)-1


#         while left < right:

#             mid = (left + right)//2

#             if nums[mid-1] < nums[mid] > nums[mid+1]:
#                 return mid

#             if nums[mid] < nums[mid+1]:
#                 left = mid + 1

#             else:
#                 right = mid - 1
        
#         # Doesn't actually matter if is left or right, because at the end of the loop they're equals
#         return left

#     # Testing
#     print(find_peak_element(nums=nums))

#     'Notes: ChatGPT helped me understanding the conditions and guided through the solution'

'''179. Largest Number'''
# def x():

#     # Input
#     # Case 1
#     nums = [20,1]
#     # Output: "201"

#     # Case 2
#     nums = [3,30,34,5,9]
#     # Output: "9534330"

#     # Custom Case
#     nums = [8308,8308,830]
#     # Output: "83088308830"


#     'My 1st Approach'
#     def largestNumber(nums: list[int]) -> str: 

#         nums = [str(x) for x in nums]
    
#         res = sorted(nums, key= lambda x: int(x[0]), reverse=True)  


#         # Mergesort
#         def mergesort(seq: list) -> list:

#             if len(seq) <= 1:
#                 return seq

#             mid = len(seq)//2

#             left_side, right_side = seq[:mid], seq[mid:]

#             left_side = mergesort(left_side)
#             right_side = mergesort(right_side)

#             return merge(left=left_side, right=right_side)

#         # Auxiliary merge for Mergesort
#         def merge(left: list, right: list) -> list:

#             res = []
#             zeros = []
#             i = j = 0

#             while i < len(left) and j < len(right):

#                 if left[i][-1] == '0':
#                     zeros.append(left[i])
#                     i+=1

#                 elif right[j][-1] == '0':
#                     zeros.append(right[j])
#                     j+=1
                
#                 elif left[i][0] == right[j][0]:

#                     if left[i]+right[j] > right[j]+left[i]:
#                         res.append(left[i])
#                         i+=1

#                     else:
#                         res.append(right[j])
#                         j+=1                

#                 elif int(left[i][0]) > int(right[j][0]):
#                     res.append(left[i])
#                     i+=1
                
#                 else:
#                     res.append(right[j])
#                     j+=1
            

#             while i < len(left):
#                 res.append(left[i])
#                 i+=1

            
#             while j < len(right):
#                 res.append(right[j])
#                 j+=1


#             # Deal with the elements with '0' as last digit
#             zeros.sort(key=lambda x: int(x), reverse=True)

#             return res+zeros          

#         result = mergesort(seq=res)
        
#         return ''.join(result)

#     # Testing
#     print(largestNumber(nums=nums))

#     'Note: This approach cleared 57% of cases '


#     'My 2nd Approach'
#     def largestNumber(nums: list[int]) -> str: 

#         res = [str(x) for x in nums]    
#         # res = sorted(nums, key= lambda x: int(x[0]), reverse=True)  

#         # Mergesort
#         def mergesort(seq: list) -> list:

#             if len(seq) <= 1:
#                 return seq

#             mid = len(seq)//2

#             left_side, right_side = seq[:mid], seq[mid:]

#             left_side = mergesort(left_side)
#             right_side = mergesort(right_side)

#             return merge(left=left_side, right=right_side)

#         # Auxiliary merge for Mergesort
#         def merge(left: list, right: list) -> list:

#             res = []        
#             i = j = 0

#             while i < len(left) and j < len(right):

#                 if left[i]+right[j] > right[j]+left[i]:
#                     res.append(left[i])
#                     i += 1

#                 else:
#                     res.append(right[j])
#                     j += 1
            
#             while i < len(left):
#                 res.append(left[i])
#                 i += 1
                            
#             while j < len(right):
#                 res.append(right[j])
#                 j += 1

#             return res        

#         result = mergesort(seq=res)
        
#         return ''.join(result)

#     # Testing
#     print(largestNumber(nums=nums))

#     'Note: This one did it!'

'''189. Rotate Array'''
# def x():

#     'Input'
#     # Case 1
#     nums, k = [1,2,3,4,5,6,7], 3
#     # Output: [5,6,7,1,2,3,4]

#     # Case 2
#     nums, k = [-1,-100,3,99], 2
#     # Output: [3,99,-1,-100]

#     # My approach
#     def rotate(nums: list[int], k: int) -> None:

#         if len(nums) == 1:
#             return
        
#         rot = k % len(nums)

#         dic = {k:v for k, v in enumerate(nums)}

#         for i in range(len(nums)):

#             n_idx = (i+rot)%len(nums)
#             nums[n_idx] = dic[i]

#     'Note:It actually worked!'

'''198. House Robber'''
# def x():

#     # Input
#     # Case 1
#     nums = [1,2,3,1]
#     # Output: 4

#     # Case 2
#     nums = [2,7,9,3,1]
#     # Output: 12

#     # Custom Case
#     nums = [2,1,1,2]
#     # Output: 12


#     'DP Approach / space: O(n)'
#     def rob(nums: list[int]) -> int:
        
#         # Handling corner cases
#         if len(nums) == 1:
#             return nums[0]
        
#         # Initializing the aux array
#         dp = [0] * len(nums)
#         dp[0] = nums[0]
#         dp[1] = max(dp[0], nums[1])

#         for i in range(2, len(nums)):

#             dp[i] = max(dp[i-1], dp[i-2] + nums[i])

#         return dp[-1]

#     # Testing
#     print(rob(nums=nums))
                    
#     'Note: This could be done in O(1) space'

        
#     'DS Approach / space: O(1)'
#     def rob(nums: list[int]) -> int:
        
#         # Handling corner cases
#         if len(nums) == 1:
#             return nums[0]
        
#         # Initializing the aux array
#         prev_rob = 0
#         max_rob = 0

#         for num in nums:

#             temp = max(max_rob, prev_rob + num)
#             prev_rob = max_rob
#             max_rob = temp
        
#         return max_rob

#     # Testing
#     print(rob(nums=nums))

#     'Done'

'''204. Count Primes'''
# def x():

#     # Input
#     # Case 1
#     n = 10
#     # Output: 4 (2,3,5,7)

#     # Custom Case
#     n = 30
#     # Output: 4 (2,3,5,7)


#     '''
#     My Approach

#         Intuition
#             - Application of Eratosthenes Sieve
#     '''

#     def countPrimes(n: int) -> int:

#         # Handling corner cases
#         if n in range(3):
#             return 0 
        
            
#         primes, non_primes = [], []

#         for num in range(2, n):

#             primes.append(num) if num not in non_primes else None

#             non_primes.extend(x for x in range(num*num, n, num))
        
#         return len(primes)

#     # Testing
#     print(countPrimes(n=n))

#     '''
#     Note: This solution works well for data input in low scales (Worked for 26% of the cases), for big numbers could be quite time complex.

#     After researching a modified version of the Sieve is the way to go, instead of appending numbers to later count them, creating a boolean list to only mark
#     the multiples of other primes is more time and space efficient than storing the actual numbers.

#         But the real hit here is that we will curb the loop of marking the multiples to the square root of the parameter given, because is safe to assume that after the square root
#         other numbers will pretty much be multiples of the range before the SR.

#     '''

#     'Another Approach'
#     def countPrimes(n:int) -> int:

#         if n <= 2:
#             return 0

#         primes = [True]*n

#         primes[0] = primes[1] = False

#         for i in range(2, int(n**0.5)+1):

#             if primes[i]:

#                 for j in range(i*i, n, i):
#                     primes[j] = False
        
#         return sum(primes)

#     'Note:This one did it!'

'''215. Kth Largest Element in an Array'''
# def x():

#     'Solution'
#     import heapq

#     def findKthLargest(self, nums: list[int], k: int) -> int:
#             heap = nums[:k]
#             heapq.heapify(heap)
            
#             for num in nums[k:]:
#                 if num > heap[0]:
#                     heapq.heappop(heap)
#                     heapq.heappush(heap, num)
            
#             return heap[0]

#     'Done'

'''238. Product of Array Except Self'''
# def x():

#     # Input
#     # Case 1
#     nums = [1,2,3,4]
#     # Output: [24,12,8,6]

#     # Case 2
#     nums = [-1,1,0,-3,3]
#     # Output: [0,0,9,0,0]


#     'My Approach / Brute forcing'
#     from math import prod

#     def product_except_self(nums:list[int]) -> list[int]:

#         res = []

#         for i in range(len(nums)):
#             res.append(prod(nums[:i]+nums[i+1:]))

#         return res

#     # Testing
#     print(product_except_self(nums=nums))

#     'Note: This solution suffices 75% of test cases, but resulted inefficient with large inputs'


#     'My Approach v2 - (Trying to carry the result - pseudo-prefix sum)'
#     from math import prod

#     def product_except_self(nums:list[int]) -> list[int]:

#         res = []
#         nums_prod = prod(nums)

#         for i in range(len(nums)):

#             elem = nums_prod//nums[i] if nums[i] != 0 else prod(nums[:i]+nums[i+1:])
#             res.append(elem)


#         return res

#     # Testing
#     print(product_except_self(nums=nums))

#     'Note: It worked and beated 85% in time compl. and 74% in memory compl.'


#     '''
#     Preffix-Suffix product Approach - Customary

#     Intuition:
#         - The core idea of this approach is to build an array of the carrying product of all elements from left to right (Preffix)
#             and build another more array with the sabe but from right to left (suffix).

#         - After having that by combining those products but element by element, the preffix from 0 to n-1 indexed and the suffix from n-1 to 0 indexed
#             and EXCLUDING the current index (i) in the final traversal, the 'self' element is explicitly excluded from the product.
#     '''

#     from itertools import accumulate
#     import operator

#     def product_except_self(nums:list[int]) -> list[int]:

#         res = []

#         # Populate both preffix and suffix
#         preffix = list(accumulate(nums, operator.mul))
#         suffix = list(accumulate(reversed(nums), operator.mul))[::-1]


#         # Combine the results
#         for i in range(len(preffix)):

#             if 0 < i < len(preffix)-1:
#                 res.append(preffix[i-1]*suffix[i+1])
            
#             elif i == 0:
#                 res.append(suffix[i+1])

#             else:
#                 res.append(preffix[i-1])
        
#         return res

#     # Testing
#     print(product_except_self(nums=nums))

#     'Done'

'''239. Sliding Window Maximum'''
# def x():

#     # Input
#     # Case 1
#     nums = [1,3,-1,-3,5,3,6,7]
#     k = 3
#     # Output: [3,3,5,5,6,7]

#     # Case 2
#     nums = [1]
#     k = 1
#     # Output: [1]

#     # Cusom Case
#     nums = [1,3,-1,-3,5,3,6,7]
#     k = 3
#     # Output: [3,3,5,5,6,7]


#     'My approach'
#     def max_sliding_window(nums:list[int], k:int) -> list[int]:

#         if len(nums) == 1:
#             return nums
        
#         if k == len(nums):
#             return [max(nums)]


#         result = []

#         for i in range(len(nums)-k+1):
#             result.append(max(nums[i:i+k]))

#         return result

#     # Testing
#     print(max_sliding_window(nums=nums, k=k))

#     'Note: This approach cleared 73% of test cases, but breaks with large inputs'


#     'Monotonically Decreacing Queue'
#     def max_sliding_window(nums:list[int], k:int) -> list[int]:

#         import collections

#         output = []
#         deque = collections.deque() # nums
#         left = right = 0

#         while right < len(nums):

#             # Pop smaller values from de deque
#             while deque and nums[deque[-1]] < nums[right]:
#                 deque.pop()

#             deque.append(right)

#             # remove the left val from the window
#             if left > deque[0]:
#                 deque.popleft()

#             if (right+1) >= k:
#                 output.append(nums[deque[0]])
#                 left += 1
            
#             right += 1

#         return output

#     # Testing
#     print(max_sliding_window(nums=nums, k=k))

#     'done'

'''283. Move Zeroes'''
# def x():

#     # Input
#     # Case 1
#     nums = [0,1,0,3,12]
#     # Output: [1,3,12,0,0]

#     # Case 2
#     nums = [0]
#     # Output: [0]

#     # Custom Case
#     nums = [2,3,4,0,5,6,8,0,1,0,0,0,9]
#     # Output: [0]


#     '''
#     My Approach

#         Intuition:
#             - Create a new list as a buffer to hold every item in the initial order
#             - Separate the buffer into non-zeroes and zeroes different list and joint them together.
#             - Replace each value of the original list with the order or the buffer list.

#         This solution is more memory expensive than one with a Two-pointer approach, but let's try it
#     '''

#     def move_zeroes(nums:list[int]) -> None:

#         # Handle corner case
#         if len(nums) == 1:
#             return nums
    
#         # Create the buffers to separate the non-zeroes to the zeroes
#         non_zeroes, zeroes = [x for x in nums if x != 0],[x for x in nums if x == 0]

#         # Join the buffers into one single list
#         buffer = non_zeroes + zeroes

#         # Modify the original input with the buffer's order
#         for i in range(len(nums)):
#             nums[i] = buffer[i]
    
#     # Testing
#     move_zeroes(nums=nums)
#     print(nums)

#     'Note: This solution was accepted and beated submissions by 37% in runtime and 87% in memory'


#     'Two-pointers Approach'
#     def move_zeroes(nums:list[int]) -> None:

#         # Initialize the left pointer
#         l = 0

#         # Iterate with the right pointer through the elements of nums
#         for r in range(len(nums)):

#             if nums[r] != 0:

#                 nums[r], nums[l] = nums[l], nums[r]

#                 l += 1

#     # Testing
#     move_zeroes(nums=nums)
#     print(nums)

#     'Done'

'''287. Find the Duplicate Number'''
# def x():

#     # Input
#     # Case 1
#     nums = [1,3,4,2,2]
#     # Output: 2

#     # Case 2
#     nums = [3,1,3,4,2]
#     # Output: 3

#     # Custom Case
#     nums = [3,3,3,3,3]
#     # Output: 3

#     'My approach'

#     def find_duplicate(nums:list[int]) -> int:

#         for num in nums:

#             if nums.count(num) != 1:
#                 return num
    
#     # Testing
#     print(find_duplicate(nums=nums))

#     'Note: This approach cleared 92% of cases but breaks with larger inputs'


#     'Hare & Tortoise Approach'
#     def find_duplicate(nums:list[int]) -> int:

#         # Initialize two pointers directing to the first element in the list
#         slow = fast = nums[0]

#         # Iterate until they coincide (They' found each other in the cycle)
#         while True:
#             slow = nums[slow]
#             fast = nums[nums[fast]]
            
#             if slow == fast:
#                 break
        
#         # Reset the slow to the begining of the list, so they an meet at the repeating number
#         slow = nums[0]

#         # Iterate again but at same pace, they will eventually meet at the repeated number
#         while slow != fast:
#             slow = nums[slow]
#             fast = nums[fast]

#         return fast

#     # Testing
#     print(find_duplicate(nums=nums))

#     'Done'

'''300. Longest Increasing Subsequence'''
# def x():

#     # Input
#     # Case 1
#     nums = [10,9,2,5,3,7,101,18]
#     # Output: 4

#     # Case 2
#     nums = [0,1,0,3,2,3]
#     # Output: 4

#     # Case 3
#     nums = nums = [7,7,7,7,7,7,7]
#     # Output: 1


#     'DP Solution'
#     def lengthOfLIS(nums: list[int]) -> int:    
        
#         # Handle corner case
#         if not nums:
#             return 0        

#         # Initialize the dp array
#         dp = [1] * len(nums)

#         # Iterate through the elements of the list, starting from the second
#         for i in range(1, len(nums)):

#             for j in range(i):

#                 if nums[i] > nums[j]:
#                     dp[i] = max(dp[i], dp[j]+1)

#         return max(dp)

#     'Done'

'''315. Count of Smaller Numbers After Self'''
# def x():

#     # Input
#     # Case 1
#     nums = [5,2,6,1]
#     # Output: [2,1,1,0]

#     # Case 1
#     nums = [-1,-1]
#     # Output: [0,0]


#     'My Approach (Brute forcing)'
#     def count_smaller(nums: list[int]) -> list[int]:

#         # Handle corner case
#         if len(nums) == 1:
#             return [0]
        

#         # Set the min value of the group
#         min_num = min(nums)
        

#         # Initialize the result holder
#         result = []

#         for x,num in enumerate(nums):

#             # corner case: if the number is the smallest of the group or the right most one, no smaller numbers after it
#             if num == min_num or num == nums[-1]:
#                 result.append(0)
            
#             else:

#                 # Define a sublist with all elements to the right of the current one
#                 sublist = nums[x+1:]

#                 # Count how many of those are smaller than the current one
#                 count = len([x for x in sublist if x<num])

#                 # Add that result to the holder
#                 result.append(count)
                
#         return result
        
#     # Testing
#     print(count_smaller(nums=nums))

#     'Note: This approach met up to 79% o the cases'

'''334. Increasing Triplet Subsequence'''
# def x():

#     # Input
#     # Case 1
#     nums = [1,2,3,4,5]
#     # Output: True / Any triplet where i < j < k is valid.

#     # Case 2
#     nums = [5,4,3,2,1]
#     # Output: False / Any triplet where i < j < k is valid.

#     # Case 3
#     nums = [2,1,5,0,4,6]
#     # Output: True / The triplet (3, 4, 5) where [0,4,6] is valid.

#     # Custom Case
#     nums = [1,2,2147483647]
#     # Output: False.


#     '''
#     My approach (Brute forcing) - Iterative looping

#         Intuition:

#             - Handle corner cases: 
#                 + If no input; 
#                 + if input length < 3; 
#                 + If input length = 3 != to sorted(input, reverse = False)
#                 + If input == sorted(input, reverse = True)

#             - In a while loop check one by one, starting from the first index, if next to it is any other element greater than it.
#                 from that element start the search for a greater element than the first greater and 
                
#                 + if found, return True;
#                 + else, move the initial index to the next and start over
#                 + if the initial index gets to the second last element and no triplet has been found, return False.
#     '''

#     def increasingTriplet(nums: list[int]) -> bool:

#         # Handle corner cases
#         if not nums or len(nums) < 3 or (len(nums) == 3 and nums != sorted(nums, reverse=True)) or nums == sorted(nums, reverse=True):
#             return False

#         # Initialize the triplet initial index
#         i = 0

#         # Iterate through the input elements
#         while i < len(nums)-2:

#             for j in range(i+1, len(nums)):

#                 if nums[j] > nums[i]:

#                     for k in range(j+1, len(nums)):

#                         if nums[k] > nums[j]:

#                             return True
                        
#             i += 1
        
#         return False

#     # Testing
#     print(increasingTriplet(nums=nums))

#     'Note: This approach met 90% of test cases, but failed with larger inputs. Time complexity: O(n^3)'


#     '''
#     My approach - Iterative selection

#         Intuition:

#             - Starting from the first index, check with listcomp if there is a larger element present.
#                 + if it does, get its index and do the same but for this second element.
#                     * if there are a larger element present return True,
#                     * else, move the initial input to the next and start over.

#             - Like the prior approach if it reaches the second last element in the input, end the loop and return False
#     '''

#     def increasingTriplet(nums: list[int]) -> bool:

#         # Handle corner cases
#         # if not nums or len(nums) < 3 or (len(nums) == 3 and nums != sorted(nums, reverse=True)) or nums == sorted(nums, reverse=True):
#         #     return False

#         # Initialize the triplet initial index
#         i = 0

#         # Iterate through the input elements
#         while i < len(nums)-2:

#             # Get the next greater element of nums[i]
#             sec_greater = list(filter(lambda x: x>nums[i], nums[i+1:-1]))

#             # if such element exist
#             if sec_greater:    
                
#                 # Iterate again for the rest of the greater elements
#                 for elem in sec_greater:

#                     # Get the idx of the first greater element than nums[i]
#                     j = nums.index(elem, i+1)            

#                     # Find a element greater than nums[j]
#                     third_greater = list(filter(lambda x: x>nums[j], nums[j+1:]))

#                     # if there are greater element than nums[j], return True
#                     if third_greater:
#                         return True       
                            
#             i += 1
        
#         return False

#     # Testing
#     print(increasingTriplet(nums=nums))

#     'Note: This approach met 90% of test cases, but failed with larger inputs. Time complexity: O(n^2*logn)'


#     'Optimized solution O(n)'

#     def increasingTriplet(nums: list[int]) -> bool:

#         first = float('inf')
#         second = float('inf')
        
#         for num in nums:

#             if num <= first:
#                 first = num

#             elif num <= second:
#                 second = num

#             else:
#                 return True
        
#         return False

#     # Testing
#     print(increasingTriplet(nums=nums))

#     'Done'

'''347. Top K Frequent Elements'''
# def x():

#     # Input
#     # Case 1
#     nums = [1,2,2,1]
#     k = 2
#     # Output: [1,2]

#     # Case 2
#     nums = [1]
#     k = 1
#     # Output: [1]


#     '''
#     My approach

#         Intuition:
            
#         Ideas' pool:
#             + A Counter function approach:
#                 - Call a Counter on the input and sort by freq, return in order.

#             + A Heap approach:
#                 - ...    
#     '''

#     def topKFrequent(nums: list[int], k: int) -> list[int]:

#         # Create the result list holder
#         result = []

#         # Import Counter
#         from collections import Counter

#         #  Transform the input into a list sorted by freq
#         nums = sorted(Counter(nums).items(), key=lambda x: x[1], reverse=True)

#         # Populate the result accordingly
#         for i in range(k):
#             result.append(nums[i][0])

#         # Return the result
#         return result

#     # Testing
#     print(topKFrequent(nums=nums, k=k))

#     'Note: This approach worked beating submissions only 20% in Runtime and 61% in Memory'

#     'Done'

'''350. Intersection of Two Arrays II'''
# def x():

#     # Input
#     # Case 1
#     nums1, nums2 = [1,2,2,1], [2,2]
#     # Output: [2,2]

#     # Case 2
#     nums1, nums2 = [4,9,5], [9,4,9,8,4]
#     # Output: [4,9]


#     '''
#     My approach

#         Intuition:
#             - Handle a corner case.
#             - Make a list holder for the result.
#             - Get the largest list.
#             - Collect the common elements and populate the result holder with the lower count from both inputs.
#     '''

#     def intersect(nums1: list[int], nums2: list[int]) -> list[int]:

#         # Handle corner case
#         if not nums1 or not nums2:
#             return []

#         # Create a list holder for the common elements
#         commons = []

#         # Create an iterator with the longest list
#         longest = nums1 if len(nums1)>len(nums2) else nums2
        
#         # Collect the common elements
#         for elem in longest:

#             if elem in nums1 and elem in nums2 and elem not in commons:

#                count = nums1.count(elem) if nums1.count(elem) < nums2.count(elem) else nums2.count(elem)

#                commons.extend([elem]*count)
        
#         return commons

#     # Testing
#     print(intersect(nums1=nums1, nums2=nums2))

#     'Note: This approach worked and beated only 5% in runtine and 93% in memory'


#     'Two pointer approach'
#     def intersect(nums1: list[int], nums2: list[int]) -> list[int]:

#         # Sort both arrays
#         nums1.sort()
#         nums2.sort()
        
#         # Initialize pointers and the result list
#         i, j = 0, 0
#         result = []
        
#         # Traverse both arrays
#         while i < len(nums1) and j < len(nums2):

#             if nums1[i] < nums2[j]:
#                 i += 1

#             elif nums1[i] > nums2[j]:
#                 j += 1

#             else:
#                 result.append(nums1[i])
#                 i += 1
#                 j += 1
        
#         return result

#     # Testing
#     print(intersect(nums1=nums1, nums2=nums2))

#     'Done'

'''378. Kth Smallest Element in a Sorted Matrix'''
# def x():

#     # Input
#     # Case 1
#     matrix = [[1,5,9],[10,11,13],[12,13,15]]
#     k = 8
#     # Output: 13 / The elements in the matrix are [1,5,9,10,11,12,13,13,15], and the 8th smallest number is 13

#     # Case 2
#     matrix = matrix = [[-5]]
#     k = 1
#     # Output: -5


#     '''
#     My Approach

#         Intuition:

#             Ideas' pool:

#                 + Brute forcing: flatten the input, sort and return.
#                 + Heap: Hold a min heap of size k, traverse all items in the matrix and return the last of the heap.

#     '''

#     'Brute force'
#     def kthSmallest(matrix: list[list[int]], k: int) -> int:

#         # Flatten the input
#         matrix = [x for elem in matrix for x in elem]

#         # Sort the resulting matrix
#         matrix.sort()

#         # x=0 

#         # Return the kth element
#         return matrix[k-1]

#     # Testing
#     print(kthSmallest(matrix=matrix, k=k))

#     'Note: This approach works, it has O(nlongn) time complexity and beated other submissions by 89% in Runtine and 22% in Memory'


#     'Min-heap approach'
#     def kthSmallest(matrix: list[list[int]], k: int) -> int:

#         # Capture the matrix dimentions
#         n = len(matrix)

#         # Import the heapq module
#         import heapq
        
#         # Create a min-heap with the first element of each row
#         min_heap = [(matrix[i][0], i, 0) for i in range(n)]
#         heapq.heapify(min_heap)
        
#         # Extract min k-1 times to get to the kth smallest element
#         for _ in range(k - 1):
#             value, row, col = heapq.heappop(min_heap)
#             if col + 1 < n:
#                 heapq.heappush(min_heap, (matrix[row][col + 1], row, col + 1))
        
#         # The root of the heap is the kth smallest element
#         return heapq.heappop(min_heap)[0]

#     # Testing
#     print(kthSmallest(matrix=matrix, k=k))

#     'Note: This solution worked, it has a time complexity of O(klogn) and beated submissions by 50% in Runtime and 34% in Memory.'

#     'Done'

'''384. Shuffle an Array'''
# def x():

#     # Input
#     # Case 1
#     operations = ["Solution", "shuffle", "reset", "shuffle"]
#     inputs = [[[1, 2, 3]], [], [], []]
#     # Output: [None, [3, 1, 2], [1, 2, 3], [1, 3, 2]]

#     # Case 2
#     operations = ["Solution","reset","shuffle","reset","shuffle","reset","shuffle","reset","shuffle"]
#     inputs = [[[-6,10,184]],[],[],[],[],[],[],[],[]]
#     # Output: [[null,[-6,10,184],[-6,10,184],[-6,10,184],[-6,184,10],[-6,10,184],[10,-6,184],[-6,10,184],[10,-6,184]]


#     'My approach'
#     import random

#     class Solution:

#         def __init__(self, nums: list[int]):
#             self.base = nums
                
#         def reset(self) -> list[int]:       
#             return self.base
            
#         def shuffle(self) -> list[int]:
#             nums = self.base[:] # Deepcopy of the list
#             random.shuffle(nums)        
#             return nums
        

#     # Testing
#     for i, op in enumerate(operations):

#         if op == 'Solution':
#             obj = Solution(inputs[i][0])
#             print('Object created')

#         elif op == 'shuffle':
#             print(obj.shuffle())

#         else:
#             print(obj.reset())

#     'Notes: This approached worked beating 88% of submissions in runtime and 25% in memory'

#     'Done'

'''454. 4Sum II'''
# def x():

#     # Input
#     # Case 1
#     nums1 = [1,2]
#     nums2 = [-2,-1]
#     nums3 = [-1,2]
#     nums4 = [0,2]
#     # Output: 2


#     '''
#     My approach / Brute forcing

#         Intuition:
#             - With the help of itertools class 'product' all the possible index combinations will be generated.
#             - Check iteratively which sums up to 0 and count them.
#             - Return the count.
#     '''

#     def fourSumCount( nums1: list[int], nums2: list[int], nums3: list[int], nums4: list[int]) -> int:

#         # Import 'product' from itertool
#         from itertools import product

#         # Capture the size of the arrays
#         n = len(nums1)

#         # Handle Corner Case: if n = 1
#         if n == 1:
#             return 1 if sum(nums1+nums2+nums3+nums4) == 0 else 0
        
#         # Create the count holder
#         count = 0

#         # Generate all possible indexes combinations for the size of the arrays
#         combinations = [x for x in product(range(n), repeat=4)]

#         # Iteratively check each combination to see if they meet the requirement
#         for comb in combinations:

#             tupl = [nums1[comb[0]], nums2[comb[1]], nums3[comb[2]], nums4[comb[3]]]
#             comb_sum = sum(tupl)
#             count += 1 if comb_sum == 0 else 0
        
#         return count

#     # Testing
#     print(fourSumCount(nums1=nums1, nums2=nums2, nums3=nums3, nums4=nums4))

#     '''Note: This apporach was O(n^4) complex and while this approach works, it's very memory and time extensive, so it just met up 14% of the test cases.'''


#     '''
#     An Optimized O(n^2)

#         Intuition:
#             - Check all possible sums of elements between the first two arrays.
#             - Find if there is their complement in the remaining two.
#             - Return the count of the sums that have their complement.
#     '''

#     def fourSumCount( nums1: list[int], nums2: list[int], nums3: list[int], nums4: list[int]) -> int:

#         # Import 'product' from itertool
#         from collections import Counter
            
#         # Initialize the count holder
#         count = 0

#         # Count the sums of all pairs in nums1 and nums2
#         countAB = Counter(a+b for a in nums1 for b in nums2)

#         # Check for the complements of the existing sums in countAB
#         for c in nums3:
#             for d in nums4:

#                 if countAB[-(c+d)]:
#                     count += countAB[-(c+d)]
        
#         return count

#     'Done'








'### HASH TABLE ###'

'3. Longest Substring Without Repeating Characters'
# def x():
#     s = "abcabcbb"


#     # My solution
#     substrings = []

#     i = 0

#     while i < len(s):

#         sub = str()

#         for char in s[i:]:

#             if char in sub:
#                 substrings.append(sub)
#                 break

#             sub += char
        
#         if sub not in substrings:
#             substrings.append(sub)

#         i += 1

#     # print(substrings)

#     max_sub = max(substrings, key = len) if substrings else 0

#     # print(max_sub)

#     print(max_sub, len(max_sub))


#     # Another more efficient solution

#     def lengthOfLongestSubstring(s: str) -> int:
            
#             n = len(s)
#             maxLength = 0
#             charMap = {}
#             left = 0
            
#             for right in range(n):

#                 if s[right] not in charMap or charMap[s[right]] < left:
#                     charMap[s[right]] = right
#                     maxLength = max(maxLength, right - left + 1)

#                 else:
#                     left = charMap[s[right]] + 1
#                     charMap[s[right]] = right
            
#             return maxLength


#     lengthOfLongestSubstring(s)

'13. Roman to Integer'
# def x():

#     '''
#     Substraction exceptions:
#         - I can be placed before V (5) and X (10) to make 4 and 9. 
#         - X can be placed before L (50) and C (100) to make 40 and 90. 
#         - C can be placed before D (500) and M (1000) to make 400 and 900.
#     '''

#     # Input
#     s = 'MCMXCIV'


#     # My approach
#     res = 0
#     rom_to_int_dic = {'I': 1, 'IV': 4, 'V': 5, 'IX': 9, 'X': 10, 'XL': 40, 'L': 50, 'XC': 90, 'C': 100, 'CD': 400, 'D': 500, 'CM': 900, 'M': 1000, }


#     #Substraction Exceptions
#     if 'IV' in s:
#         res += rom_to_int_dic['IV']
#         s = s.replace('IV','')

#     if 'IX' in s:
#         res += rom_to_int_dic['IX']
#         s = s.replace('IX','')

#     if 'XL' in s:
#         res += rom_to_int_dic['XL']
#         s = s.replace('XL','')

#     if 'XC' in s:
#         res += rom_to_int_dic['XC']
#         s = s.replace('XC','')

#     if 'CD' in s:
#         res += rom_to_int_dic['CD']
#         s = s.replace('CD','')

#     if 'CM' in s:
#         res += rom_to_int_dic['CM']
#         s = s.replace('CM','')

#     # Dealing with the Remaining Number
#     if s:
#         for chr in s:
#             res += rom_to_int_dic[chr]

#     else:
#         print(res)


#     print(res)

#     '''
#     Note: This version works, but there is a more concise way
#     '''

#     s = 'MCMXCIV'

#     # ChatGPT's Approach
#     roman_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
#     total = 0
#     prev_value = 0

#     for char in s[::-1]:    #Reverse to simplify the process
        
#         curr_value = roman_dict[char]

#         if curr_value < prev_value:
#             total -= curr_value
        
#         else:
#             total += curr_value
#             prev_value = curr_value

#     print(total)

'17. Letter Combinations of a Phone Number'
# def x():

#     # Input
#     s = '23'

#     '''
#     My Approach

#     Rationale: Brute-forcing 
#         Iterating by the number of characters there are.
#     '''

#     def letterCombinations(digits):

#         dic = {
#             '2':['a', 'b', 'c'],
#             '3':['d', 'e', 'f'],
#             '4':['g', 'h', 'i'],
#             '5':['j', 'k', 'l'],
#             '6':['m', 'n', 'o'],
#             '7':['p', 'q', 'r', 's'],
#             '8':['t', 'u', 'v'],
#             '9':['w', 'x', 'y', 'z'],
#             }


#         lists = []

#         for i in digits:
#             lists.append(dic[i])

        
#         comb = []

#         if not digits:
#             return comb


#         if len(digits) == 4:
#             for i in range(len(lists[0])):
#                 for j in range(len(lists[1])):
#                     for k in range(len(lists[2])):
#                         for l in range(len(lists[3])):
#                             # comb.append(f'{lists[0][i]}{lists[1][j]}{lists[2][k]}{lists[3][l]}')
#                             comb.append(lists[0][i]+lists[1][j]+lists[2][k]+lists[3][l])
    
#         elif len(digits) == 3:
#             for i in range(len(lists[0])):
#                 for j in range(len(lists[1])):
#                     for k in range(len(lists[2])):
#                         comb.append(f'{lists[0][i]}{lists[1][j]}{lists[2][k]}')

#         elif len(digits) == 2:
#             for i in range(len(lists[0])):
#                 for j in range(len(lists[1])):
#                     comb.append(f'{lists[0][i]}{lists[1][j]}')

#         elif len(digits) == 1:
#             for i in range(len(lists[0])):
#                 comb.append(f'{lists[0][i]}')


#         return comb


#     print(letterCombinations(s))

#     'Notes: It works but it could be better'


#     # Recursive Approach

#     def letterCombinations(digits):

#         dic = {
#             '2':['a', 'b', 'c'],
#             '3':['d', 'e', 'f'],
#             '4':['g', 'h', 'i'],
#             '5':['j', 'k', 'l'],
#             '6':['m', 'n', 'o'],
#             '7':['p', 'q', 'r', 's'],
#             '8':['t', 'u', 'v'],
#             '9':['w', 'x', 'y', 'z'],
#             }

#         lists = []

#         for i in digits:
#             lists.append(dic[i])


#         def combine_sublists(lst):

#             if len(lst) == 1:
#                 return lst[0]
            
#             result = []

#             for item in lst[0]:
#                 for rest in combine_sublists(lst[1:]):
#                     result.append(item + rest)
            
#             return result
        
#         if not digits:
#             return []
        
#         else:
#             return combine_sublists(lists)

#     print(letterCombinations(s))

#     'Notes: Works pretty well'


#     # Itertools Approach

#     import itertools

#     def letterCombinations(digits):

#         dic = {
#             '2':['a', 'b', 'c'],
#             '3':['d', 'e', 'f'],
#             '4':['g', 'h', 'i'],
#             '5':['j', 'k', 'l'],
#             '6':['m', 'n', 'o'],
#             '7':['p', 'q', 'r', 's'],
#             '8':['t', 'u', 'v'],
#             '9':['w', 'x', 'y', 'z'],
#             }

#         lists = []

#         for i in digits:
#             lists.append(dic[i])

#         if not digits:
#             return []
        
#         else:
#             return [''.join(comb) for comb in itertools.product(*lists)]
        

#     print(letterCombinations(s))

'73. Set Matrix Zeroes'
# def x():

#     # Input
#     # Case 1
#     matrix = [[1,1,1],[1,0,1],[1,1,1]]
#     # Output: [[1,0,1],[0,0,0],[1,0,1]]

#     # Case 2
#     matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
#     # Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]

#     # Custom Case
#     matrix = [...]
#     # Output: [...]

#     '''
#     My Approach

#         Intuition:
#             - Locate the indexes of every 0 present
#             - Try to overwrite the values for the row and the column of each occurrence
#             - Look up if the col and row are already 0 to optimize
#     '''

#     def setZeroes(matrix: list[list[int]]) -> list[list[int]]:

#         m, n = len(matrix), len(matrix[0])
#         occurrences = []

#         for i, row in enumerate(matrix):

#             for j, col in enumerate(row):

#                 if 0 not in row:
#                     continue
                
#                 if col == 0:
#                     occurrences.append((i,j))

#         for pair in occurrences:

#             matrix[pair[0]] = [0] * n

#             for row in range(m):
#                 matrix[row][pair[1]] = 0
        
#         return matrix

#     # Testing
#     for i in setZeroes(matrix):
#         print(i)

#     '''
#     Notes: It actually passed! :D
#     '''

'76. Minimum Window Substring'
# def x():

#     # Input
#     # Case 1
#     s, t = 'ADOBECODEBANC', 'ABC'
#     # Output: "BANC"

#     # Case 2
#     s, t = 'a', 'a'
#     # Output: "a"

#     # Case 3
#     s, t = 'a', 'aa'
#     # Output: "abbbbbcdd"

#     # Custom case
#     s, t = 'aaaaaaaaaaaabbbbbcdd', 'abcdd'
#     # Output: "abbbbbcdd"


#     'My approach'
#     def minWindow(s:str, t:str) -> str:

#         if len(t) > len(s):
#             return ''
        
#         if t == s:
#             return t
        

#         for i in range(len(t), len(s) + 1):

#             for j in range((len(s)-i) + 1):
                
#                 if all([char in s[j:j+i] for char in t]):
#                     return s[j:j+i]
                
#         return ''

#     'Notes: This solution works up to 57%'


#     'With an improvement'
#     def minWindow(s:str, t:str) -> str:

#         from collections import Counter

#         if len(t) > len(s):
#             return ''
        
#         if t == s:
#             return t
        
#         count_t = Counter(t).items()

#         for i in range(len(t), len(s) + 1):

#             for j in range((len(s)-i) + 1):
                
#                 subs = s[j:j+i]
#                 count_subs = Counter(subs)

#                 if all( (x[0] in count_subs.keys() and x[1] <= count_subs[x[0]]) for x in count_t):
#                     return s[j:j+i]
                
#         return ''

#     'Notes: This solution works up to 93% and hit the time limit'


#     'Another solution'
#     def minWindow(s, t):    

#         if not s or not t:
#             return ""


#         from collections import defaultdict

#         dictT = defaultdict(int)
#         for c in t:
#             dictT[c] += 1

#         required = len(dictT)
#         l, r = 0, 0
#         formed = 0

#         windowCounts = defaultdict(int)
#         ans = [-1, 0, 0]

#         while r < len(s):
#             c = s[r]
#             windowCounts[c] += 1

#             if c in dictT and windowCounts[c] == dictT[c]:
#                 formed += 1

#             while l <= r and formed == required:
#                 c = s[l]

#                 if ans[0] == -1 or r - l + 1 < ans[0]:
#                     ans[0] = r - l + 1
#                     ans[1] = l
#                     ans[2] = r

#                 windowCounts[c] -= 1
#                 if c in dictT and windowCounts[c] < dictT[c]:
#                     formed -= 1

#                 l += 1

#             r += 1

#         return "" if ans[0] == -1 else s[ans[1]:ans[2] + 1]
            
#     # Testing
#     print(minWindow(s,t))

'''127. Word Ladder'''
# def x():

#     # Input
#     #Case 1
#     begin_word, end_word, word_list = 'hit', 'cog', ['hot', 'dot', 'dog', 'lot', 'log', 'cog']
#     #Output: 5

#     #Custom Case
#     begin_word, end_word, word_list = 'a', 'c', ['a', 'b', 'c']
#     #Output: 5
    

#     '''
#     My Approach

#         Intuition:
#             1. handle the corner case: the end_word not in the word_list
#             2. create an auxiliary func that check the word against the end_word: True if differ at most by 1 char, else False.
#             3. create a counter initialized in 0
#             4. start checking the begin_word and the end_word, if False sum 1 to the count, and change to the subquent word in the word_list and do the same.
#     '''

#     def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:

#         if endWord not in wordList:
#             return 0
        
#         def check(word):
#             return False if len([x for x in word if x not in endWord]) > 1 else True
        
#         if beginWord not in wordList:
#             wordList.insert(0,beginWord)
#             count = 0
        
#         else:
#             count = 1
        
#         for elem in wordList:
#             count += 1

#             if check(elem):
#                 return count     
                
#         return 0

#     # Testing
#     print(ladderLength(beginWord=begin_word, endWord=end_word, wordList=word_list))

#     'Note: This solution only went up to the 21% of the cases'


#     'BFS approach'
#     from collections import defaultdict, deque

#     def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:

#         if endWord not in wordList or not endWord or not beginWord or not wordList:
#             return 0

#         L = len(beginWord)
#         all_combo_dict = defaultdict(list)

#         for word in wordList:
#             for i in range(L):
#                 all_combo_dict[word[:i] + "*" + word[i+1:]].append(word) 

#         queue = deque([(beginWord, 1)])
#         visited = set()
#         visited.add(beginWord)

#         while queue:
#             current_word, level = queue.popleft()

#             for i in range(L):
#                 intermediate_word = current_word[:i] + "*" + current_word[i+1:]

#                 for word in all_combo_dict[intermediate_word]:

#                     if word == endWord:
#                         return level + 1

#                     if word not in visited:
#                         visited.add(word)
#                         queue.append((word, level + 1))
                        
#         return 0

#     'Done'

'''138. Copy List with Random Pointer'''
# def x():

#     # Base
#     class Node:
#         def __init__(self, x, next=None, random=None):
#             self.val = int(x)
#             self.next = next
#             self.random = random


#     #Input
#     #Case 1
#     head_map = [[7,None],[13,0],[11,4],[10,2],[1,0]]

#     #Build the relations of the list
#     nodes = [Node(x=val[0]) for val in head_map]

#     for i in range(len(nodes)):
#         nodes[i].next = None if i == len(nodes)-1 else nodes[i+1]
#         nodes[i].random = None if head_map[i][1] is None else nodes[head_map[i][1]]

#     head = nodes[0]
#     # Output: [[7,None],[13,0],[11,4],[10,2],[1,0]]


#     #Case 2
#     head_map = [[1,1],[2,1]]

#     #Build the relations of the list
#     nodes = [Node(x=val[0]) for val in head_map]

#     for i in range(len(nodes)):
#         nodes[i].next = None if i == len(nodes)-1 else nodes[i+1]
#         nodes[i].random = None if head_map[i][1] is None else nodes[head_map[i][1]]

#     head = nodes[0]
#     #Output: [[7,None],[13,0],[11,4],[10,2],[1,0]]


#     #Case 3
#     head_map = [[3,None],[3,0],[3,None]]

#     #Build the relations of the list
#     nodes = [Node(x=val[0]) for val in head_map]

#     for i in range(len(nodes)):
#         nodes[i].next = None if i == len(nodes)-1 else nodes[i+1]
#         nodes[i].random = None if head_map[i][1] is None else nodes[head_map[i][1]]

#     head = nodes[0]
#     #Output: [[7,None],[13,0],[11,4],[10,2],[1,0]]


#     '''
#     My Approach

#         Intuition:
#             - Traverse through the list
#             - Create a copy of each node and store it into a list along side with the content of the random pointer.
#             - Traverse the list linking each node to the next and the random pointer to the position in that list.

#         Thoughts:

#         - It is possible to create the list with a recursive solution but it'll be still necesary to traverse again
#             to collect the content of the random pointer or how else I can point to somewhere at each moment I don't know if it exist. 
#     '''

#     def copyRandomList(head:Node) -> Node:

#         # Handle the corner case where there is a single node list
#         if head.next == None:
#             result = Node(x = head.val, random=result)
#             return result

#         # Initilize a nodes holder dict to collect the new nodes while traversing the list
#         nodes = {}

#         # Initilize a nodes holder list to collect the old nodes values while traversing the list
#         old_nodes_vals = []

#         # Initialize a dummy node to traverse the list
#         current_node = head

#         # Traverse the list
#         while current_node is not None:

#             # Collect the old nodes
#             old_nodes_vals.append(current_node.val)

#             # Check if the node doesn't already exist due to the random pointer handling
#             if current_node.val not in nodes.keys(): 

#                 new_node = Node(x = current_node.val)
#                 nodes[new_node.val] = new_node
            
#             else:
#                 new_node = nodes[current_node.val]


#             # Handle the random pointer 
#             if current_node.random is None:
#                 new_node.random = None

#             else:

#                 # If the randoms does not exist already in the dict, create a new entry in the dict with the random value as key and a node holding that value 
#                 if current_node.random.val not in nodes.keys():
#                     nodes[current_node.random.val] = Node(x = current_node.random.val)
            
#                 new_node.random = nodes[current_node.random.val]


#             # Move to the next node
#             current_node = current_node.next
        

#         # Pull the nodes as a list to link to their next attribute
#         nodes_list = [nodes[x] for x in old_nodes_vals]

#         # Traverse the nodes list
#         for i, node in enumerate(nodes_list):
#             node.next = nodes_list[i+1] if i != len(nodes_list)-1 else None    

#         return nodes_list[0]

#     # Testing
#     result = copyRandomList(head=head)

#     new_copy = []
#     while result is not None:
#         new_copy.append([result.val, result.random.val if result.random is not None else None])
#         result = result.next

#     'Note: My solution works while the values of the list are unique, otherwise a new approach is needed'


#     'Another Approach'
#     def copyRandomList(head:Node):

#         nodes_map = {}

#         current = head

#         while current is not None:

#             nodes_map[current] = Node(x = current.val)
#             current = current.next

        
#         current = head

#         while current is not None:

#             new_node = nodes_map[current]
#             new_node.next = nodes_map.get(current.next)
#             new_node.random = nodes_map.get(current.random)

#             current = current.next
        
#         return nodes_map[head]


#     result = copyRandomList(head=head)


#     new_copy = []
#     while result is not None:
#         new_copy.append([result.val, result.random.val if result.random is not None else None])
#         result = result.next

#     'Done'

'''146. LRU Cache'''
# def x():

#     # Input
#     commands = ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
#     inputs = [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
#     # Output: [null, null, null, 1, null, -1, null, -1, 3, 4]


#     '''
#     My Approach
    
#         Intuition:

#             - The use of 'OrderedDicts' from the Collections module will be useful to keep track of the last recently used values
#     '''

#     class LRUCache(object):   

#         def __init__(self, capacity):
#             """
#             :type capacity: int
#             """     

#             self.capacity = capacity
#             self.capacity_count = 0
#             self.memory = {}
            

#         def get(self, key):
#             """
#             :type key: int
#             :rtype: int
#             """

#             output = self.memory.get(key,-1)

#             if output != -1:

#                 item = (key, self.memory[key])
#                 del self.memory[item[0]]
#                 self.memory[item[0]] = item[1]

#             return output
            

#         def put(self, key, value):
#             """
#             :type key: int
#             :type value: int
#             :rtype: None
#             """

#             existing_key = self.memory.get(key, -1)

#             if existing_key == -1:
#                 self.memory[key] = value

#             else:
#                 self.memory.update({key:value})

#                 item = (key, value)
#                 del self.memory[item[0]]
#                 self.memory[item[0]] = item[1]
            
#             self.capacity_count += 1

#             if self.capacity_count > self.capacity:

#                 del_item = list(self.memory.keys())[0]
#                 del self.memory[del_item]
                
#                 self.capacity_count = self.capacity

#     'Done'

'''166. Fraction to Recurring Decimal'''
# def x():

#     # Input
#     # Case 1
#     num, den = 1, 2
#     # Output: "0.5"

#     # Case 2
#     num, den = 2, 1
#     # Output: "2"

#     # Case 3
#     num, den = 4, 333
#     # Output: "0.(012)"

#     # Custom Case 
#     num, den = 1, 6
#     # Output: "0.1(6)"


#     '''
#     My approach
        
#         Intuition:

#             Here main issue is solving how to identify patterns in a string:
#                 - I'll try with parsing the string with split()
#     '''

#     def fraction_to_decimal(numerator: int, denominator: int) -> str:

#         # If exact division
#         if int(numerator/denominator) == numerator/denominator:
#             return str(int(numerator/denominator))
        
#         division = str(numerator/denominator)

#         whole, decimal = division.split('.')

#         pattern = ''

#         for i in range(len(decimal)-1):

#             pattern += decimal[i]
#             abr = decimal.split(pattern)

#             if not any(abr):
#                 return f'{whole}.({pattern})'            
        
#         return f'{whole}.{decimal}'

#     # Testing
#     print(fraction_to_decimal(numerator=num, denominator=den))

#     '''Note: My solution only solved 50% of the cases because it only works if the whole decimal part is recurring and also didnt considered negatives results'''


#     'Hashmap / Long division Approach'
#     def fraction_to_decimal(numerator: int, denominator: int) -> str:

#         # If exact division
#         if numerator % denominator == 0:
#             return str(numerator//denominator)
        
#         # Determe if is a negative result
#         sign = '-' if numerator * denominator < 0 else None

#         # Work with absolutes to simplify the calculation
#         numerator, denominator = abs(numerator), abs(denominator)

#         # Initialize integer and decimal parts
#         integer_part = numerator // denominator
#         remainder = numerator % denominator

#         decimal_part = ''
#         remainder_dict = {}

#         # Track the position of the decimals
#         position = 0

#         # Build the decimal part
#         while remainder != 0:

#             if remainder in remainder_dict:

#                 repeat_start = remainder_dict[remainder]
#                 non_repeaing_part = decimal_part[:repeat_start]
#                 repeating_part = decimal_part[repeat_start:]
#                 return f'{integer_part}.{non_repeaing_part}({repeating_part})' if not sign else f'-{integer_part}.{non_repeaing_part}({repeating_part})'

#             remainder_dict[remainder] = position
#             remainder *= 10
#             digit = remainder // denominator
#             decimal_part += str(digit)
#             remainder %= denominator
#             position += 1
        
#         return f'{integer_part}.{decimal_part}' if not sign else f'-{integer_part}.{decimal_part}'

#     # Testing
#     print(fraction_to_decimal(numerator=num, denominator=den))

#     '''Note: The final solution were based on understanding how long division works and when to capture the moment when is repeating the remainders'''

'''202. Happy Number'''
# def x():

#     # Input
#     # Case 1
#     n = 19
#     # Output: True

#     # Case 2
#     n = 2
#     # Output: False

#     # Custom Case
#     n = 18
#     # Output: False


#     '''
#     My Approach
    
#         Intuition (Recursive)
            
#             - Recursively separate the digits and check the sum of their squares compared to 1.
#                 - If the stackoverflow is reached, return False.            
#     '''

#     def isHappy(n:int) -> bool:

#         def aux(m:int) -> bool:

#             num = [int(x)**2 for x in str(m)]
#             num = sum(num)

#             if num == 1:
#                 return True
            
#             return aux(m=num)
        
#         try:
#             res = aux(m=n)

#             if res:
#                 return True
        
#         except RecursionError as e:        
#             return False

#     # Testing
#     print(isHappy(n=n))

#     'This approach may work but it exceed time limit: only met 4% of cases'


#     '''
#     Set Approach

#     There are mainly two ways of solving this: The set approach and the Floyd's Cycle detection algorithm

#         - The set approach: Use a set to save the seen numbers and if you end up in one of them, you entered a cycle
#         - The Floyd's Cycle Detection Algorithm: Similar to the case of catching a cycle in a linked list with two pointers: Slow and Fast.
#     '''

#     'Set Approach'
#     def isHappy(n:int) -> bool:

#         def getNum(m:int)->int:
#             return sum(int(x)**2 for x in str(m))

#         seen = set()

#         while n != 1 and n not in seen:
#             seen.add(n)
#             n = getNum(n)
        
#         return n == 1

#     # Testing
#     print(isHappy(n=n))


#     'FDC Approach'
#     def isHappy(n:int) -> bool:

#         def getNum(m:int)->int:
#             return sum(int(x)**2 for x in str(m))

#         slow = n
#         fast = getNum(n)

#         while fast != 1 and slow != fast:
#             slow = getNum(slow)
#             fast = getNum(getNum(fast))
        
#         return fast == 1

#     # Testing
#     print(isHappy(n=n))

#     'Done'

'''208. Implement Trie (Prefix Tree)'''
# def x():

#     # Implementation
#     class TrieNode:

#         def __init__(self, is_word=False):
#             self.values = {}
#             self.is_word = is_word

#     'Solution'
#     class Trie:

#         def __init__(self):
#             self.root = TrieNode()
    

#         def insert(self, word: str) -> None:

#             node = self.root

#             for char in word:

#                 if char not in node.values:
#                     node.values[char] = TrieNode()
                
#                 node = node.values[char]

#             node.is_word = True


#         def search(self, word: str) -> bool:
            
#             node = self.root

#             for char in word:          
                        
#                 if char not in node.values:
#                     return False
                
#                 node = node.values[char]
            
#             return node.is_word


#         def startsWith(self, prefix: str) -> bool:
            
#             node = self.root

#             for char in prefix:

#                 if char not in node.values:
#                     return False
                
#                 node = node.values[char]
            
#             return True

#     # Testing
#     new_trie = Trie()
#     new_trie.insert('Carrot')
#     print(new_trie.startsWith('Car'))  

#     'Done'

'''380. Insert Delete GetRandom O(1)'''
# def x():

#     # Input
#     # Case 1
#     operations = ["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
#     inputs = [[], [1], [2], [2], [], [1], [2], []]
#     # Output: [None, true, false, true, 2, true, false, 2]


#     'My approach'
#     import random

#     class RandomizedSet:

#         def __init__(self):
#             self.set: set = set()
#             print('Set created!')
            

#         def insert(self, val: int) -> bool:

#             if val not in self.set:
#                 self.set.add(val)
#                 return True

#             else:
#                 return False
            

#         def remove(self, val: int) -> bool:

#             if val in self.set:
#                 self.set.remove(val)
#                 return True
            
#             else:
#                 return False
            

#         def getRandom(self) -> int:

#             return random.choice(list(self.set))

#     'Note: While this approach works, it has O(1) time complexity for all the functions, the list casting in the getRandom() function make it go up to O(n) breaking the challenge requirement'


#     'An optimal solution'
#     import random

#     class RandomizedSet:

#         def __init__(self):
#             self.list = []
#             self.dict = {}
                    
#         def insert(self, val: int) -> bool:

#             if val in self.dict:
#                 return False
            
#             self.dict[val] = len(self.list)
#             self.list.append(val)

#             return True

#         def remove(self, val: int) -> bool:

#             if val not in self.dict:
#                 return False
            
#             last_value, idx = self.list[-1], self.dict[val]

#             # Rewrite the list and the dict
#             self.list[idx], self.dict[last_value] = last_value, idx

#             # Update the list to remove the duplicate
#             self.list.pop()

#             # Remove the value entry in the dict
#             del self.dict[val]

#             return True
            
#         def getRandom(self) -> int:

#             return random.choice(self.list)

#     # Testing
#     for i, op in enumerate(operations):

#         if op == 'RandomizedSet':
#             obj = RandomizedSet()
            
#         elif op == 'insert':
#             print(obj.insert(inputs[i][0]))
        
#         elif op == 'remove':
#             print(obj.remove(inputs[i][0]))
        
#         else:
#             print(obj.getRandom())

#     'Done'







'### MATRICES ###'

'54. Spiral Matrix'
# def x():

#     # Input
#     # Case 1
#     matrix = [[1,2,3],[4,5,6],[7,8,9]]
#     # Output: [1,2,3,6,9,8,7,4,5]

#     # Case 2
#     matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
#     # Output: [1,2,3,4,8,12,11,10,9,5,6,7]

#     # Custom Case  / m = 1
#     matrix = [[2,1,3]]
#     # Output: [2,1,3]

#     # Custom Case / n = 1
#     matrix = [[2],[1],[3],[5],[4]]
#     # Output: [2,1,3,5,4]

#     # Custom Case
#     matrix = [[1,2],[3,4]]
#     # Output: [1,2,4,3]

#     # Custom Case
#     matrix = [[2,3,4],[5,6,7],[8,9,10],[11,12,13],[14,15,16]]
#     # Output: [2,3,4,7,10,13,16,15,14,11,8,5,6,9,12]

#     '''
#     My approach

#         Intuition:
#             1. Handle Special Cases: m = 1 / n = 1 

#             2. Make a while loop that runs until the input has no elements:
#                 a. Take all the subelements from the first element and append them individually to the result.
#                 b. Make a for loop and take the last subelement of each element and append them individually to the result
#                 c. Take all the subelements from the last element and append them in a reverse order individually to the result.
#                 d. Make a for loop and take the first subelement of each element and append them in a reverse order individually to the result.

#             3. Return the result
#     '''

#     def spiralOrder(matrix):

#         if len(matrix) == 1:
#             return matrix[0]
        
#         if len(matrix[0]) == 1:
#             return [num for vec in matrix for num in vec]
        

#         result = []

#         while len(matrix) != 0:

#             first_element = matrix.pop(0)
#             result += first_element

#             if len(matrix) == 0:
#                 break

#             second_element = []
#             for elem in matrix:
#                 second_element.append(elem.pop(-1))
#             result += second_element

#             third_element = matrix.pop(-1)
#             result += reversed(third_element)
            
#             if len(matrix) > 0 and len(matrix[0]) == 0:
#                 break

#             fourth_element = []
#             for elem in matrix:
#                 fourth_element.append(elem.pop(0))
#             result += reversed(fourth_element)

#         return result

#     print(spiralOrder(matrix))


#     'Notes: it works up to 76% of the cases, but from here seems more like patching something that could be better designed'


#     'Another Approach'

#     def spiralOrder(matrix):
            
#             result = []

#             while matrix:
#                 result += matrix.pop(0) # 1

#                 if matrix and matrix[0]: # 2 
#                     for line in matrix:
#                         result.append(line.pop())

#                 if matrix: # 3
#                     result += matrix.pop()[::-1]

#                 if matrix and matrix[0]: # 4

#                     for line in matrix[::-1]:
#                         result.append(line.pop(0))

#             return result

#     'Notes: Same logic, better executed'

'79. Word Search'
# def x():

#     # Input
#     # Case 1
#     board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
#     word = 'ABCCED'
#     # Output: True

#     # Case 2
#     board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
#     word = 'SEE'
#     # Output: True

#     # Case 3
#     board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
#     word = 'ABCB'
#     # Output: False

#     '''
#     Intuition:

#         The problem can be solved by traversing the grid and performing a depth-first search (DFS) for each possible starting position. 
#         At each cell, we check if the current character matches the corresponding character of the word. 
#         If it does, we explore all four directions (up, down, left, right) recursively until we find the complete word or exhaust all possibilities.

#         Approach

#             1. Implement a recursive function backtrack that takes the current position (i, j) in the grid and the current index k of the word.
#             2. Base cases:
#                 - If k equals the length of the word, return True, indicating that the word has been found.
#                 - If the current position (i, j) is out of the grid boundaries or the character at (i, j) does not match the character at index k of the word, return False.
#             3. Mark the current cell as visited by changing its value or marking it as empty.
#             4. Recursively explore all four directions (up, down, left, right) by calling backtrack with updated positions (i+1, j), (i-1, j), (i, j+1), and (i, j-1).
#             5. If any recursive call returns True, indicating that the word has been found, return True.
#             6. If none of the recursive calls returns True, reset the current cell to its original value and return False.
#             7. Iterate through all cells in the grid and call the backtrack function for each cell. If any call returns True, return True, indicating that the word exists in the grid. Otherwise, return False.
            
#     '''

#     'Backtracking (Recursive) Approach'
#     def exist(board: list[list[str]], word: str) -> bool:

#         def backtrack(i, j, k):

#             if k == len(word):
#                 return True
            
#             if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[k]:
#                 return False
            
#             temp = board[i][j]
#             board[i][j] = ''
            
#             if backtrack(i+1, j, k+1) or backtrack(i-1, j, k+1) or backtrack(i, j+1, k+1) or backtrack(i, j-1, k+1):
#                 return True
            
#             board[i][j] = temp

#             return False

#         for i in range(len(board)):

#             for j in range(len(board[0])):

#                 if backtrack(i, j, 0):

#                     return True

#         return False
            
#     # Testing
#     print(exist(board, word))

'''130. Surrounded Regions'''
# def x():

#     #Input
#     #Case 1
#     board = [
#         ["X","X","X","X"],
#         ["X","O","O","X"],
#         ["X","X","O","X"],
#         ["X","O","X","X"]
#         ]
#     # output = [
#     #     ["X","X","X","X"],
#     #     ["X","X","X","X"],
#     #     ["X","X","X","X"],
#     #     ["X","O","X","X"]
#     #     ]

#     #Case 2
#     board = [
#         ['X']
#         ]
#     # output = [
#         # ['X']
#         # ]

#     #Custom Case
#     board = [["O","O"],["O","O"]]


#     '''
#     My Approach

#         Intuition:
#             1. Check if there is any 'O' at the boarders.
#             2. Check is there is any 'O' adjacent to the one in the boarder:
#                 - If do, add them to the not-be-flipped ground and re run.
#                 - if doesn't, flip everything to 'X' and return
#             (Do this until there is no 'O' unchecked )
#     '''

#     def solve(board:list[list[str]]) -> None:

#         M = len(board)
#         N = len(board[0])

#         no_flip = []
#         all_os = []


#         # Collect all 'O's
#         for i in range(M):
#             all_os.extend((i,j) for j in range(N) if board[i][j] == 'O')
        

#         #   Check if there is a boarder 'O' within the group
#         for i in range(len(all_os)):

#             if all_os[i][0] in (0, M-1) or all_os[i][1] in (0, N-1):
#                 no_flip.append(all_os[i])


#         # Collect the 'O's near to no_flip 'O' iteratively
#         flipped = None
#         i = 0

#         while True:

#             # Condition to end the loop
#             if len(all_os) == 0 or i == len(all_os) and flipped is False:
#                 break

#             #Collecting the possibilities of an adjacent 'O'
#             adjacents = []

#             for pos in no_flip:
#                 adjacents.extend([(pos[0]-1, pos[1]), (pos[0]+1, pos[1]), (pos[0], pos[1]-1), (pos[0], pos[1]+1)])
            
#             #Check if the current element is adjacent to any no_flip 'O'
#             if all_os[i] in adjacents:
#                 no_flip.append(all_os.pop(i))
#                 flipped = True
#                 i = 0
#                 continue

#             i += 1
#             flipped = False


#         # Rewritting the board
#         #   Resetting the board to all "X"
#         for i in range(M):
#             board[i] = ["X"]*N
        
#         #   preserving the no_flip 'O's
#         for o in no_flip:
#             board[o[0]][o[1]] = 'O'

#     # Testing
#     solve(board=board)

#     'This solution met 98.2% of the cases'


#     'DFS Approach'
#     def solve(board):

#         n,m=len(board),len(board[0])
#         seen=set()

#         def is_valid(i,j):
#             return 0 <= i < n and 0<= j <m and board[i][j]=="O" and (i,j) not in seen
        
#         def is_border(i,j):
#             return i == 0 or i == n-1 or j == 0 or j == m-1
        
#         def dfs(i,j):

#             board[i][j]="y"
#             seen.add((i,j))

#             for dx , dy in ((0,1) ,(0,-1) ,(1,0),(-1,0)):
#                 new_i , new_j = dx + i , dy + j

#                 if is_valid(new_i , new_j):
#                     dfs(new_i , new_j)
            
#         for i in range(n):
#             for j in range(m):
#                 if is_border(i,j) and board[i][j]=="O":
#                     dfs(i,j) 
                    
#         for i in range(n):
#             for j in range(m):
#                 if board[i][j]=="y":
#                     board[i][j]="O"
#                 else:
#                     board[i][j]="X"

#     # Testing
#     solve(board)

#     'Done'

'''200. Number of Islands'''
# def x():

#     # Input
#     # Case 1
#     grid = [
#       ["1","1","1","1","0"],
#       ["1","1","0","1","0"],
#       ["1","1","0","0","0"],
#       ["0","0","0","0","0"]
#     ]
#     # Ouput: 1

#     # Case 2
#     grid = [
#       ["1","1","0","0","0"],
#       ["1","1","0","0","0"],
#       ["0","0","1","0","0"],
#       ["0","0","0","1","1"]
#     ]
#     # Ouput: 3

#     # Custom Case
#     grid = [
#         ["1","0"]
#         ]
#     # Ouput: 1


#     'My BFS Approach'
#     def numIslands(grid:list[list[str]]) -> int:
        
#         if len(grid) == 1:
#             return len([x for x in grid[0] if x =='1'])

#         # Create the 'lands' coordinates
#         coord = []

#         # Collecting the 'lands' coordinates
#         for i, row in enumerate(grid):
#             coord.extend((i, j) for j, value in enumerate(row) if value == '1')


#         # Create the groups holder
#         islands = []
#         used = set()


#         # BFS Definition
#         def bfs(root:tuple) -> list:

#             queue = [root]
#             curr_island = []

#             while queue:

#                 land = queue.pop(0)
#                 x, y = land[0], land[1]
                
#                 if grid[x][y] == '1' and (land not in curr_island and land not in used):

#                     curr_island.append(land)
                
#                     # Define next lands to search
#                     if x == 0:
#                         if y == 0:
#                             next_lands = [(x+1,y),(x,y+1)]
                        
#                         elif y < len(grid[0])-1:
#                             next_lands = [(x+1,y),(x,y-1),(x,y+1)]
                        
#                         else:
#                             next_lands = [(x+1,y),(x,y-1)]
                    
#                     elif x < len(grid)-1:
#                         if y == 0:
#                             next_lands = [(x-1,y),(x+1,y),(x,y+1)]
                        
#                         elif y < len(grid[0])-1:
#                             next_lands = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
                        
#                         else:
#                             next_lands = [(x-1,y),(x+1,y),(x,y-1)]
                    
#                     else:
#                         if y == 0:
#                             next_lands = [(x-1,y),(x,y+1)]
                        
#                         elif y < len(grid[0])-1:
#                             next_lands = [(x-1,y),(x,y-1),(x,y+1)]
                        
#                         else:
#                             next_lands = [(x-1,y),(x,y-1)]
                                    
#                     # List the next lands to visit
#                     for next_land in next_lands:

#                         if next_land not in curr_island:

#                             queue.append(next_land)

#             return curr_island
            

#         # Checking all the 1s in the grid
#         for elem in coord:

#             if elem not in used:

#                 island = bfs(elem)

#                 islands.append(island)
#                 used.update(set(island))
        
#         return len(islands)

#     # Testing
#     print(numIslands(grid=grid))
    
#     'Note: This could be done way simplier'


#     'Simplified & Corrected BFS Approach'
#     def numIslands(grid:list[list[str]]) -> int:

#         if not grid:
#             return 0

#         num_islands = 0
#         directions = [(1,0),(-1,0),(0,1),(0,-1)]

#         for i in range(len(grid)):

#             for j in range(len(grid[0])):

#                 if grid[i][j] == '1':

#                     num_islands += 1

#                     queue = [(i,j)]

#                     while queue:

#                         x, y = queue.pop(0)

#                         if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == '1':

#                             grid[x][y] = '0'    # Mark as visited

#                             for dx, dy in directions:

#                                 queue.append((x + dx, y + dy))
        
#         return num_islands

#     'Done'

'''212. Word Search II'''
# def x():

#     # Input
#     # Case 1
#     board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
#     words = ["oath","pea","eat","rain"]
#     # Output: ["eat","oath"]

#     # Case 2
#     board = [["a","b"],["c","d"]], 
#     words = ["abcb"]
#     # Output: []

#     # Custom Case
#     board = [["a","b"],["c","d"]], 
#     words = ["abcb"]
#     # Output: []


#     '''
#     My Approach
    
#         Intuiton:

#             - Based on the 'Word Seach I' backtracking solution, I will try to emulate the same but
#                 since now there are multiple word to lookout for, I will rely on a Trie implementation
#                 to look out for prefixes to optimize the process.

#                 And to try to make it work, I will pull the first letter of each word and only start
#                 the searches from those positions, so, roughly the plan is:

#                 1. Collect the coordinates of the first letter from each of the word and store them in a dict
#                     as {'word': coordinates[(x,y)]}, if a word has no coordinates and it means it won't be found
#                     in the matrix, so it won't be in Trie.
                
#                 2. Initiate the Trie with the words with coordinates.

#                 3. Iterate through each of the words, and iterate for each pair of coordinates to look out for that word,
#                     if found, add it to a result list if don't pass to the next pair of coordinates, and so on for each word.
                
#                 4. Return the found words.
#     '''

#     'ACTUAL CODE'
#     # TRIE IMPLEMENTATION

#     # TrieNode Definition
#     class TrieNode:

#         def __init__(self):
#             self.values = {}
#             self.is_word = False


#     # Trie DS Definition
#     class Trie:

#         def __init__(self):
#             self.root = TrieNode()
        
#         def insert(self, word:str) -> None:

#             curr_node = self.root

#             for char in word:

#                 if char not in curr_node.values:
#                     curr_node.values[char] = TrieNode()
                
#                 curr_node = curr_node.values[char]
            
#             curr_node.is_word = True

#         def search(self, word:str) -> bool:

#             curr_node = self.root

#             for char in word:

#                 if char not in curr_node.values:
#                     return False
                
#                 curr_node = curr_node.values[char]

#             return curr_node.is_word

#         def stars_with(self, prefix:str) -> bool:

#             curr_node = self.root

#             for char in prefix:

#                 if char not in curr_node.values:
#                     return False
                
#                 curr_node = curr_node.values[char]

#             return True

#     'Actual Solution'
#     def findWords(board: list[list[str]], words: list[str]) -> list[str]:

#         import copy

#         #AUX BACKTRACK FUNC DEF
#         def backtrack(i:int, j:int, k:str) -> bool:

#             if new_trie.search(k):
#                 return True
                    
#             if not new_trie.stars_with(k):
#                 return False
            
#             temp = board[i][j]
#             board[i][j] = '.'

#             #1
#             if 0<i<len(board)-1 and 0<j<len(board[0])-1:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True
            
#             #2
#             elif 0 == i and 0 == j:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i, j+1, k+board[i][j+1]):        
#                     return True
                
#             #3
#             elif 0 == i and 0<j<len(board[0])-1:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i, j+1, k+board[i][j+1]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True
            
#             #4
#             elif len(board)-1 == i and len(board[0])-1 == j:
#                 if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True
            
#             #5
#             elif 0<i<len(board)-1 and 0 == j:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]):        
#                     return True
                
#             #6
#             elif 0<i<len(board)-1 and len(board[0])-1 == j:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True
            
#             #7
#             elif len(board)-1 == i and 0 == j:
#                 if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]):        
#                     return True
            
#             #8
#             elif len(board)-1 == i and 0<j<len(board[0])-1:
#                 if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True

#             #9
#             elif len(board)-1 == i and len(board[0])-1 == j:
#                 if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True


#             board[i][j] = temp

#             return False 
        

#         # COLLECT FIRST LETTER COORDINATES FOR EACH WORD
#         words_dict = {}

#         for word in words:

#             coordinates = []

#             for i,row in enumerate(board):
#                 coordinates.extend([(i,j) for j,elem in enumerate(row) if board[i][j] == word[0]])

#             if coordinates:
#                 words_dict[word] = coordinates


#         # INITIATE THE TRIE
#         new_trie = Trie()

#         for word in words_dict.keys():
#             new_trie.insert(word)

#         x = 0

#         result = []

#         # ITERATE THE DICT
#         for word in words_dict:

#             temp_board = copy.deepcopy(board)

#             for i,j in words_dict[word]:

#                 if backtrack(i, j, word[0]):

#                     result.append(word)
#                     board = temp_board

#         return result

#     # Testing
#     print(findWords(board=board, words=words))

#     '''
#     Notes:
#         My solution and approach wasn't that far. The logic was correct, the execution was the one to fail.
#         My version of the solution tends to get redundant and can't handle efficiently larger inputs
#     '''

#     # TrieNode Definition
#     class TrieNode:

#         def __init__(self):
#             self.values = {}
#             self.is_word = False

#     # Trie DS Definition
#     class Trie:

#         def __init__(self):
#             self.root = TrieNode()
        
#         def insert(self, word:str) -> None:

#             curr_node = self.root

#             for char in word:
#                 if char not in curr_node.values:
#                     curr_node.values[char] = TrieNode()            
#                 curr_node = curr_node.values[char]
            
#             curr_node.is_word = True

#     'Actual Solution'
#     def findWords(board: list[list[str]], words: list[str]) -> list[str]:

#         # Build the Trie
#         trie = Trie()

#         for word in words:
#             trie.insert(word)
        
#         # Auxiliary vars
#         rows, cols = len(board), len(board[0])
#         result = set()
#         visited = set()

#         #Aux DFS Func
#         def dfs(node:TrieNode, i:int, j:str, path:str) -> None:

#             if i<0 or i>=rows or j<0 or j>=cols or (i,j) in visited or board[i][j] not in node.values:
#                 return
            
#             visited.add((i,j))
#             node = node.values[board[i][j]]
#             path += board[i][j]

#             if node.is_word:
#                 result.add(path)
#                 node.is_word = False    # To avoid duplicate results

#             # Explore neighbors in 4 directions (up, down, left, right)
#             for x, y in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
#                 dfs(node, x, y, path)
            
#             visited.remove((i,j))        

#         # Traverse the board
#         for i in range(rows):
#             for j in range(cols):
#                 dfs(trie.root, i, j, '')        

#         return result

#     'Done'

'''240. Search a 2D Matrix II'''
# def x():

#     # Input
#     # Case 1
#     matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
#     target = 5
#     # Output: True

#     # Case 2
#     matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
#     target = 20
#     # Output: False


#     '''
#     My Approach

#         Intuition:

#             - Iterativelly search in the first row of the matrix if the value is in there by a belonging test
#                 - If the value is in the element, break and return True / else, pop that element from the matrix
#                 - Transpose the matrix and start over until there's no more elements in the matrix
#             - If the loop reaches the last element of the matrix, return False
#     '''

#     def search_matrix(matrix:list[list[int]], target: int) -> bool:

#         m = len(matrix)
#         n = len(matrix[0])

#         # Handle the corner case
#         if n == m == 1:
#             return target == matrix[0][0]

#         while matrix:

#             # print(matrix)

#             element = matrix.pop(0)

#             if target in element:
#                 return True
            
#             matrix = [list(x) for x in zip(*matrix)]
        
#         return False

#     # Testing
#     print(search_matrix(matrix=matrix,target=target))

#     '''Note: This approach doesn't worked because dinamically changing the data structure mess in how python checks membership'''


#     'Binary search approach'
#     def search_matrix(matrix:list[list[int]], target: int) -> bool:

#         m = len(matrix)
#         n = len(matrix[0])

#         # Handle the corner case
#         if n == m == 1:
#             return target == matrix[0][0]

#         row, col = m-1, 0   # Start the search from the bottom left corner

#         while row >= 0 and col < n:

#             element = matrix[row][col]

#             if element == target:
#                 return True
            
#             elif element > target:
#                 row -= 1
                    
#             else:
#                 col += 1
        
#         return False

#     # Testing
#     print(search_matrix(matrix=matrix,target=target))

#     'done'

'''289. Game of Life'''
# def x():

#     # Input
#     # Case 1
#     board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
#     # Output: [[0,0,0],[1,0,1],[0,1,1],[0,1,0]]

#     # Case 2
#     board = [[1,1],[1,0]]
#     # Output: [[1,1],[1,1]]
    

#     '''
#     My Approach

#         Intuition:
#             - Create a 'result matrix' filled with 0s with the size of the original one.
#             - Create a auxiliary function that evaluates all the neighbors to collect the number of 1s to apply the rule.
#                 + In the neighbors evaluation since there are 1s and 0s, to avoid more looping I will sum up to get the number of living cells.
#             - Populate the result matrix according to the rules.
#     '''

#     def game_of_life(board:list[list[int]]) -> None:

#         # Set the matrix dimentions
#         m,n = len(board),len(board[0])

#         # Define the holder matrix
#         holder = [[0]*n for _ in range(m)]

#         # Define the directions of the neighbors
#         directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

#         # Iterate to evaluate each of the cells of the original board
#         for i in range(m):

#             for j in range(n):

#                 # Define the actual neighbors
#                 neighbors = [ board[i+dx][j+dy] for dx, dy in directions if 0 <= i+dx < m and 0 <= j+dy < n ]

#                 # Evalue the number of live neighbors
#                 neighbors = sum(neighbors)

#                 # APPLY THE RULES
#                 if board[i][j] == 1:
                    
#                     # 1. Any live cell with fewer than two live neighbors dies as if caused by under-population.
#                     if neighbors < 2:
#                         holder[i][j] = 0    # Update the holder matrix in the exact position with the result of the rule apply
                    
#                     # 2. Any live cell with two or three live neighbors lives on to the next generation.
#                     elif 2 <= neighbors <= 3:
#                         holder[i][j] = 1
                    
#                     # 3. Any live cell with more than three live neighbors dies, as if by over-population.
#                     elif neighbors > 3:
#                         holder[i][j] = 0

#                 else:
#                     # 4. Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
#                     if neighbors == 3:
#                         holder[i][j] = 1


#         # Prepare the output: Modify the original board according to the result of the game
#         for i in range(m):
#             for j in range(n):
#                 board[i][j] = holder[i][j] 


#     # Testing
#     for i in range(len(board)):
#         print(board[i])

#     game_of_life(board=board)
#     print()

#     for i in range(len(board)):
#         print(board[i])

#     'Done: My approach worked and beated 72% of submissions in runtime and 35% in space.'








'### OTHERS ###'

'7. Reverse Integer'
# def x():

#     # Input
#     x = -15

#     # My Solution
#     raw_res = str(x)[::-1]

#     if '-' in raw_res:
#         res = int(raw_res[:-1])
#         res = int('-'+str(res))

#     else:
#         res = int(raw_res)


#     min_32bit = -2**31
#     max_32bit = 2**31-1

#     if res > max_32bit or res < min_32bit:
#         print(0)

#     else:
#         print(res)

'8. String to Integer (atoi)'
# def x():

#     # Input
#     s = "   -43.25"


#     # My approach

#     # Cleaning leading blank spaces
#     s = s.strip()

#     # In case of a sign present
#     sign = None

#     if s[0] == '-' or s[0] == '+':    
#         sign = s[0]
#         s = s[1:]


#     num = ''

#     # Reviewing each valid character
#     for char in s:    
#         if char not in '0123456789.':
#             break    
#         num += char


#     decimals = None

#     # In case of a decimals
#     if '.' in num:
#         decimals = num[num.find('.')+1:] #35
#         num = num[:num.find('.')]   #42
#         decimal_break = 5 * 10**(len(decimals)-1)

#         decimals = int(decimals)
        
#         #in case no number befor '.'
#         if not num:
#             num = 0
#         else:
#             num = int(num)
        
#         if decimals >= decimal_break:
#             num += 1

#     elif num:
#         num = int(num)



#     # In case is negative
#     if sign == '-':
#         num = int('-'+str(num))


#     max_32bit = 2**31-1
#     min_32bit = -2**31


#     #Outputting the result
#     if not num:
#         print(0)

#     else:
    
#         if num < min_32bit:
#             print(min_32bit)

#         elif num > max_32bit:
#             print(max_32bit)
        
#         else:      
#             print(num)

#     ''' 
#     Note:
#         It left cases unhandled. I also don't have the time to keep building the solution.
#     '''

#     # ChatGPT approach

#     def atoi(s: str) -> int:
#         s = s.strip()  # Remove leading and trailing whitespace
#         if not s:
#             return 0
        
#         sign = 1
#         i = 0
        
#         # Check for sign
#         if s[i] in ['+', '-']:
#             if s[i] == '-':
#                 sign = -1
#             i += 1
        
#         # Iterate through characters and build the number
#         num = 0
#         while i < len(s) and s[i].isdigit():
#             num = num * 10 + int(s[i])
#             i += 1
        
#         # Apply sign and handle overflow
#         num *= sign
#         num = max(-2**31, min(2**31 - 1, num))
        
#         return num

#     print(atoi(s))

'14. Longest Common Prefix'
# def x():

#     '''
#     Approach:

#         1. Order the array alphabetically 
#             & separate in different lists the words starting with each letter.

#         2. Order each array with the longest word upfront.

#         3. Build a dict with *Preffix as key and *Count as value.
#             *Count: will be how many words start with the first letter of the first word, the first two letters of the first word, 
#             and so on until exhauting the first (longest) word
#             *Preffix: the actual first, two first and so on substrings.

#         4. Merge all the resulting dict, order them descendingly, and return the first value if the count > 2, else: return an empty string.

#     '''

#     # Input

#     #   Custom input for testing
#     strs = ["flower", "flow", "flight", "dog", "racecar", "door", "fleet", "car", "racer"]

#     #   Real input
#     strs = ["a"]




#     # My approach

#     def longestCommonPrefix(strs):

#         strs = sorted(strs, reverse=True)

#         # Here will be stored each list
#         lists = {}

#         for str in strs:

#             first_letter = str[0]

#             if first_letter in lists:
#                 lists[first_letter].append(str)
            
#             else:
#                 lists[first_letter] = [str]


#         # Converting the dict into a list to facilitate the logic
#         groups = list(lists.values())

#         # Ordering each sublist by len
#         groups = list(map(lambda x: sorted(x, key=len, reverse=True), groups))

#         # Here will be the counting and the prefixes
#         results = dict()


#         for group in groups:

#             for i in range(1, len(group[0])):
                    
#                 prefix = ''
#                 count = 0

#                 for j in range(len(group)):

#                     if group[0][:i] in group[j]:
#                         count += 1
                    
#                 if count > 1:

#                     prefix = group[0][:i]
#                     results[prefix] = count


#         results = sorted(results.items(), key = lambda x: (x[1], x[0]), reverse=True)

#         # print(results)


#         if results:
#             return results[0][0]

#         else:
#             return ''
        

#     print(longestCommonPrefix(strs))

#     '''
#     Note:
#         My solution appears to be functional but is not working as expected with unexpected input.
#     '''

#     # ChatGPT's approach

#     def longestCommonPrefix(strs):

#         if not strs:
#             return ''

#         strs.sort()
#         first = strs[0]
#         last = strs[-1]

#         prefix = ''

#         for i in range(len(first)):

#             if i < len(last) and first[i] == last[i]:
                
#                 prefix += first[i]
            
#             else:
#                 break
        
#         return prefix


#     print(longestCommonPrefix(strs))

#     '''
#     Conclusion: The difference between my implementation and this is that the problem didn't state that the prefix must be present in all the strings, I assumed it wasn't going to be.
#     '''

'29. Divide Two Integers'
# def x():

#     # Input

#     # Case 1
#     dividend = 10
#     divisor = 3

#     # Case 2
#     dividend = 7
#     divisor = -3

#     # My approach

#     '''
#     Rationale:
#         1. Count how many times the divisor could be substracted from the dividend before reaching something smalle than the divisor
#         2. if only one between dividend and the divisor is less than 0, the result would return a negative number 
#     '''


#     def divide(dividend, divisor):
        
#         # case where 0 is divided by something
#         if dividend == 0:
#             return 0
        

#         # setting variables to operate
#         count = 0
#         div = abs(divisor)
#         dvnd = abs(dividend)


#         # case where the dividend is 1
#         if div == 1 and dvnd != 0:

#             if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
#                 return -dvnd
            
#             else:
#                 return dvnd
        

#         # case where the absolute divisor is greater than the dividend
#         if div > dvnd:
#             return 0
        
#         # case where both are the same number
#         if div == dvnd:
                
#             if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
#                 return -1
            
#             else:
#                 return 1
        
#         # case where is possible to divide iteratively
#         while dvnd >= div:

#             dvnd -= div
#             count += 1
        
#         # In case any is negative, the result will also be negative
#         if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
#             return -count

#         # Otherwise, just return
#         return count


#     print(divide(dividend, divisor))

#     'Notes: My solution actually worked, theres nitpicking cases where it wont, but still '


#     # Another Approach

#     def divide(dividend, divisor):

#         if (dividend == -2147483648 and divisor == -1): return 2147483647
                
#         a, b, res = abs(dividend), abs(divisor), 0

#         for x in range(32)[::-1]:

#             if (a >> x) - b >= 0:
#                 res += 1 << x
#                 a -= b << x
        
#         return res if (dividend > 0) == (divisor > 0) else -res

#     'Notes: This challenge is solved with bitwise operations '

'38. Count and Say'
# def x():

#     # Input

#     # Case 1
#     n = 1   # Exp. Out: "1" (Base Case)

#     # Case 2
#     n = 4   # Exp. Out: "1211" (Base Case)


#     'My Approach - Iterative suboptimal solution' 
#     def countAndSay(n):

#         if n == 1:
#             return '1'
            
#         res = '1'

#         for _ in range(1, n):
        
#             pairs = []
#             count = 0
#             char = res[0]

#             for i in range(len(res)+1):

#                 if i == len(res):
#                     pairs.append(str(count)+char)

#                 elif res[i] == char:
#                     count += 1

#                 else:       
#                     pairs.append(str(count)+char)
#                     char = res[i]
#                     count = 1

#             res = ''.join(pairs)
        
#         return res

#     # Testing
#     print(countAndSay(6))

#     'Notes: It works'

#     'Recursive Approach'
#     def countAndSay(n):
#         if n == 1:
#             return '1'
#         return aux_countAndSay(countAndSay(n - 1))

#     def aux_countAndSay(s):
    
#         if not s:
#             return ''
        
#         result = []
#         count = 1

#         for i in range(1, len(s)):

#             if s[i] == s[i - 1]:
#                 count += 1

#             else:
#                 result.append(str(count) + s[i - 1])
#                 count = 1

#         result.append(str(count) + s[-1])

#         return ''.join(result)

#     # Testing
#     print(countAndSay(6))

#     'Done'

'66. Plus One'
# def x():

#     # Input
#     # Case 1
#     digits = [1,2,3]
#     # Output: [1,2,4]

#     # Case 2
#     digits = [4,3,2,1]
#     # Output: [4,3,2,2]

#     # Case 3
#     digits = [9]
#     # Output: [1,0]

#     # Custom Case
#     digits = [9,9,9]
#     # Output: [1,0,0,0]

#     '''
#     My Approach
#         Intuition:
#             - The case is simple, the catch is to handle the case "[9,9,9]"
#     '''

#     def plusOne(digits: list[int]) -> list[int]:

#         idx = -1

#         while abs(idx) <= len(digits):
            
#             if abs(idx) == len(digits) and digits[idx] == 9:

#                 digits[idx] = 1
#                 digits.append(0)
#                 break

#             if digits[idx] != 9:

#                 digits[idx] += 1
#                 break

#             digits[idx] = 0
#             idx -= 1

#         return digits

#     # Testing
#     print(plusOne(digits=digits))

#     '''
#     Notes: 

#         While this code works, there was an even cleverer approach - To convert the digits into a int, add 1 and return as a list of ints
#         this way, is avoided the handling of cases
#     '''


#     'A different Approach'
#     def plusOne(digits: list[int]) -> list[int]:

#         number = int(''.join([str(x) for x in digits]))
#         number += 1
        
#         return [int(x) for x in str(number)]

#     # Testing
#     print(plusOne(digits=digits))

'69. Sqrt(x)'
# def x():

#     # Input
#     # Case 1
#     x = 4
#     # Output: 2

#     # Case 2
#     x = 8
#     # Output: 2

#     # Custom Case
#     x = 399
#     # Output: ..

#     'My Approach'

#     limit = 46341

#     # Auxiliary Eratosthenes' sieve function
#     def primes(cap):  

#         primes = []
#         not_primes = []

#         for i in range(2, cap+1):

#             if i not in not_primes:
#                 primes.append(i)
#                 not_primes.extend([x for x in range(i*i, cap+1, i)])

#         return primes

#     def mySqrt(x:int) -> int:

#         #Setting a limit for calculating primes
#         limit = x//2

#         prime_nums = primes(limit)

#         squares = list(map(lambda x: x*x, prime_nums))


#         #proning in the squares the correct range to make a range to evaluate
#         root_range = []
#         for idx, v in enumerate(squares):

#             if x <= v:
#                 root_range = [prime_nums[idx-1], prime_nums[idx]]
#                 break

#         #Calculating manually the square of each root in range to select the floor-root for the value
#         for root in range(root_range[0], root_range[1]+1):
            
#             if root*root >= x:
#                 return result
            
#             result = root

#     # Testing
#     print(mySqrt(x))

#     'Notes: This approach was too complicated and actually not as efficient. Apparently with the notion of binary search is easier to solve'


#     'Binary Search Approach'
#     def mySqrt(x):

#         left = 0
#         right = x

#         while left <= right:

#             mid = (left + right)//2

#             if mid*mid < x:
#                 left = mid + 1

#             elif mid*mid > x: 
#                 right = mid -1

#             else:
#                 return mid
        
#         return right

#     # Testing
#     print(mySqrt(x))

'''149. Max Points on a Line'''
# def x():

#     '''
#     Revision

#         The problem could be pretty hard if no math knowledge is acquired beforehand.
#         By definition, if several points share the same 'slope' with one single point,
#         it'd mean that they are all included in the same line.

#         So the problem reduces to (brut force) check for each point if the rest share the same
#         slope and the biggest group with common slope will be the answer
#     '''

#     def maxPoints(points:list[list[int]]):

#         # if there is no more than a pair of point in the plane, well, that's the answer
#         if len(points) < 3:
#             return len(points)
        
#         # Initializing with the lowest possible answer
#         result = 2

#         # Since we are counting on pairs, we're iterating up to the second last point of the group
#         for i, point1 in enumerate(points[:-1]):

#             slopes = {} # The keys will be the slopes and the values the count of points with the same slope

#             for point2 in points[i+1:]:
                
#                 slope = None
#                 x_comp = point2[0] - point1[0]

#                 if x_comp:  # The bool of 0 is False
                    
#                     # Calculate the slope
#                     slope = (point2[1] - point1[1]) / x_comp

#                 # If that slope already exist, add one point to the count
#                 if slope in slopes:

#                     slopes[slope] += 1
#                     new = slopes[slope]

#                     result = max(result, new)
                
#                 # else, create a new dict entry
#                 else:
#                     slopes[slope] = 2

#         return result

#     'Done'

'''171. Excel Sheet Column Number'''
# def x():

#     # Input
#     # Case 1
#     columnTitle = 'A'
#     # Output: 1

#     # Case 2
#     columnTitle = 'AB'
#     # Output: 28

#     # Case 3
#     columnTitle = 'ZY'
#     # Output: 701

#     # Custom Case
#     columnTitle = 'ASY'
#     # Output: 1195

#     'My Approach'
#     def titleToNumber(columnTitle: str) -> int:
    
#         dic = {v:k for k,v in list(enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 1))}
#         res = 0

#         for idx, letter in enumerate(reversed(columnTitle)):
#             res += dic[letter]*pow(26, idx)
        
#         return res

#     # Testing
#     print(titleToNumber(columnTitle=columnTitle))

#     'Done'

'''172. Factorial Trailing Zeroes'''
# def x():

#     # Input
#     # Case 1
#     n = 3
#     # Output: 0 (3! = 6, no trailing zero).

#     # Case 2
#     n = 5
#     # Output: 1 (5! = 120).

#     # Case 3
#     n = 0
#     # Output: 0 (0! = 1).

#     # Custom case
#     n = 1574
#     # Output: 390 


#     'My Approach'
#     def trailingZeroes(n: int) -> int:

#         res = 1

#         for i in range(2, n+1):
#             res *= i
        
#         zeros = 0

#         while True:

#             if  res % 10 != 0:
#                 break
            
#             zeros += 1
#             res //= 10    
            
#         return zeros

#     # Testing
#     print(trailingZeroes(n=1574))

#     'Note: While my approach works and passed, is not as efficient, is O(n)'


#     '''
#     Optimized approach

#         Taking advantage of the fact that every factor of 5 contributes to trailing zeros
#         the problem simplifies greatly since no factorials are needed to be calculated
#     '''

#     def trailingZeroes(n: int) -> int:

#         zeros = 0

#         while n >= 5:

#             n //= 5
#             zeros += n          
            
#         return zeros

#     # Testing
#     print(trailingZeroes(n=1574))

#     'Done'

'''326. Power of Three'''
# def x():

#     # Input
#     # Case 1
#     n = 45
#     # Output: True

#     # Custom Case
#     n = -1
#     # Output: True


#     'Iterative approach'
#     def is_power_of_three(n:int) -> bool:

#         powers = [3**x for x in range(21)]

#         return n in powers


#     'Recursive apporach'
#     def is_power_of_three(n:int) -> bool:

#         # Base case: if n is 1, it's a power of three
#         if n == 1:
#             return True

#         # If n is less than 1, it can't be a power of three
#         if n < 1:
#             return False

#         # Recursive case: check if n is divisible by 3 and then recurse with n divided by 3
#         if n % 3 == 0:
#             return is_power_of_three(n // 3)

#         # If n is not divisible by 3, it's not a power of three
#         return False

#     'Done'



























