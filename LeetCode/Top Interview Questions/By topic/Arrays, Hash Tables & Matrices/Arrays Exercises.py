'''
CHALLENGES INDEX

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


(39)
'''


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














